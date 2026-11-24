"""Small, explicit Gemma mobile-QAT INT2/4/8 unpack and fixed-scale repack helpers."""

from __future__ import annotations

import torch


def unpack_signed(packed: torch.Tensor, original_width: int, num_bits: int) -> torch.Tensor:
    """Decode the exact Gemma U8 packing convention into signed integer weights."""
    if num_bits == 8:
        if packed.dtype != torch.int8:
            raise TypeError("Gemma INT8 weights are stored as int8")
        return packed[..., :original_width]
    if num_bits not in (2, 4):
        raise ValueError(f"unsupported Gemma weight width: {num_bits}")
    factor = 8 // num_bits
    packed = packed.to(torch.uint8)
    mask = (1 << num_bits) - 1
    offset = 1 << (num_bits - 1)
    parts = [((packed >> (index * num_bits)) & mask).to(torch.int16) - offset for index in range(factor)]
    return torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], -1)[..., :original_width].to(torch.int8)


def pack_signed(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Encode signed integer values into Gemma's low-value-first U8 packing."""
    if num_bits == 8:
        return values.to(torch.int8)
    if num_bits not in (2, 4):
        raise ValueError(f"unsupported Gemma weight width: {num_bits}")
    factor = 8 // num_bits
    offset = 1 << (num_bits - 1)
    low, high = -offset, offset - 1
    values = values.to(torch.int16)
    if bool(((values < low) | (values > high)).any()):
        raise ValueError(f"values exceed signed INT{num_bits} range [{low}, {high}]")
    encoded = values + offset
    width = encoded.shape[-1]
    padded_width = ((width + factor - 1) // factor) * factor
    if padded_width != width:
        encoded = torch.nn.functional.pad(encoded, (0, padded_width - width))
    groups = encoded.reshape(*encoded.shape[:-1], -1, factor)
    packed = torch.zeros(groups.shape[:-1], dtype=torch.uint8, device=values.device)
    for index in range(factor):
        packed |= (groups[..., index].to(torch.uint8) << (index * num_bits))
    return packed


def expand_scale(scale: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    """Expand per-row or evenly grouped per-row scales across input columns."""
    scale = scale.to(dtype)
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    if scale.shape[-1] == 1:
        return scale.expand(*scale.shape[:-1], width)
    if width % scale.shape[-1]:
        raise ValueError(f"scale groups ({scale.shape[-1]}) do not evenly divide width ({width})")
    return scale.repeat_interleave(width // scale.shape[-1], dim=-1)


def dequantize(packed: torch.Tensor, scale: torch.Tensor, original_width: int, num_bits: int,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct the values represented by one packed weight tensor."""
    integers = unpack_signed(packed, original_width, num_bits).to(dtype)
    return integers * expand_scale(scale, original_width, dtype)


def quantize_fixed_scales(weight: torch.Tensor, scale: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Round updated weights with existing per-row/group scales and repack them."""
    scale = expand_scale(scale, weight.shape[-1], torch.float32)
    normalized = weight.float() / scale
    if num_bits == 8:
        low, high = -128, 127
    elif num_bits in (2, 4):
        low, high = -(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1
    else:
        raise ValueError(f"unsupported Gemma weight width: {num_bits}")
    integers = torch.round(normalized).clamp(low, high).to(torch.int8)
    return pack_signed(integers, num_bits)


def quantize_recomputed_scales(
    weight: torch.Tensor, scale_template: torch.Tensor, num_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute symmetric max-absolute scales per existing row/group, then pack.

    The source checkpoint's scale tensor supplies the number of groups per row;
    no scale shape or grouping changes. The scale is max(abs(group))/qmax where
    qmax is the positive signed integer limit, so values do not clip on either
    side. Returns (packed weights, recomputed scales).
    """
    if weight.ndim != 2:
        raise ValueError("expected a 2D output-channel by input-channel weight")
    if num_bits not in (2, 4, 8):
        raise ValueError(f"unsupported Gemma weight width: {num_bits}")
    if scale_template.ndim == 1:
        groups = 1
    elif scale_template.ndim == 2:
        groups = scale_template.shape[-1]
    else:
        raise ValueError("expected a rowwise or row-group scale tensor")
    if weight.shape[0] != scale_template.shape[0] or weight.shape[1] % groups:
        raise ValueError("scale template shape is incompatible with weight shape")
    qmax = (1 << (num_bits - 1)) - 1
    grouped = weight.float().reshape(weight.shape[0], groups, -1)
    recomputed = grouped.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / qmax
    if scale_template.ndim == 1:
        new_scale = recomputed[:, 0].to(scale_template.dtype)
    else:
        new_scale = recomputed.to(scale_template.dtype).reshape_as(scale_template)
    return quantize_fixed_scales(weight, new_scale, num_bits), new_scale
