"""Narrow loader compatibility patch for the checkpoint's grouped PLE scales."""

from __future__ import annotations

import torch
from torch import nn


def resize_embedding_scale_groups(module: nn.Module, groups: int) -> None:
    """Resize only the scale parameter shape; checkpoint loading supplies its values."""
    if groups < 1:
        raise ValueError("scale groups must be positive")
    scale = getattr(module, "embedding_scale", None)
    if not isinstance(scale, nn.Parameter) or scale.ndim != 2:
        raise TypeError("expected QuantizedEmbedding.embedding_scale parameter")
    if scale.shape[1] == groups:
        return
    module.embedding_scale = nn.Parameter(
        torch.empty((scale.shape[0], groups), dtype=scale.dtype, device=scale.device),
        requires_grad=False,
    )


def install_per_layer_scale_loader_patch() -> None:
    """Resize Gemma4 PLE scale slots after quant-layer replacement and before weight loading."""
    from transformers.integrations.gemma_quant import QuantizedEmbedding
    from transformers.quantizers.quantizer_gemma import GemmaQuantizer

    if getattr(GemmaQuantizer, "_forge_ple_scale_patch", False):
        return
    original = GemmaQuantizer._process_model_before_weight_loading

    def process_before_weight_loading(self, model, **kwargs):
        original(self, model, **kwargs)
        text_config = model.config.get_text_config()
        groups = int(text_config.num_hidden_layers)
        module = next(
            (
                item
                for name, item in model.named_modules()
                if name.endswith("embed_tokens_per_layer") and isinstance(item, QuantizedEmbedding)
            ),
            None,
        )
        if module is None:
            raise RuntimeError("Gemma mobile-QAT PLE embedding was not replaced with QuantizedEmbedding")
        resize_embedding_scale_groups(module, groups)

    GemmaQuantizer._process_model_before_weight_loading = process_before_weight_loading
    GemmaQuantizer._forge_ple_scale_patch = True
