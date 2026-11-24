"""Stream the real QAT weights through dequantize/repack using their original scales."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

from quantization import dequantize, quantize_fixed_scales


def scale_key(weight_key: str) -> str:
    if weight_key.endswith(".embedding_quantized"):
        return weight_key.removesuffix(".embedding_quantized") + ".embedding_scale"
    if weight_key.endswith(".weight"):
        return weight_key.removesuffix(".weight") + ".weight_scale"
    raise ValueError(f"not a quantized Gemma weight key: {weight_key}")


def round_trip(weights: Path, manifest_path: Path, output: Path, rows_per_chunk: int = 128) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_headers = manifest["safetensors"]["tensor_headers"]
    result_rows = []
    bit_counts: Counter[int] = Counter()
    total_bytes = 0
    changed_bytes = 0
    max_reconstruction_error = 0.0

    with safe_open(str(weights), framework="pt", device="cpu") as f:
        for key, spec in tensor_headers.items():
            bits = spec.get("assigned_bits")
            if bits is None:
                continue
            scale_name = scale_key(key)
            shape = spec["shape"]
            packed_width = shape[-1]
            original_width = packed_width if bits == 8 else packed_width * (8 // bits)
            packed_slice = f.get_slice(key)
            scale_slice = f.get_slice(scale_name)
            rows = shape[0]
            tensor_changed = 0
            tensor_bytes = 0
            tensor_error = 0.0
            for start in range(0, rows, rows_per_chunk):
                end = min(rows, start + rows_per_chunk)
                packed = packed_slice[start:end]
                scale = scale_slice[start:end]
                represented = dequantize(packed, scale, original_width, bits, dtype=torch.float32)
                repacked = quantize_fixed_scales(represented, scale, bits)
                if repacked.shape != packed.shape:
                    raise RuntimeError(f"repacked shape mismatch for {key}: {repacked.shape} != {packed.shape}")
                different = int(torch.count_nonzero(repacked != packed).item())
                tensor_changed += different
                tensor_bytes += packed.numel()
                if different:
                    recon = dequantize(repacked, scale, original_width, bits, dtype=torch.float32)
                    tensor_error = max(tensor_error, float((recon - represented).abs().max().item()))
            result_rows.append({
                "name": key,
                "bits": bits,
                "packed_shape": shape,
                "scale_shape": tensor_headers[scale_name]["shape"],
                "packed_bytes": tensor_bytes,
                "changed_packed_bytes": tensor_changed,
                "max_reconstruction_error": tensor_error,
            })
            bit_counts[bits] += 1
            total_bytes += tensor_bytes
            changed_bytes += tensor_changed
            max_reconstruction_error = max(max_reconstruction_error, tensor_error)
            if len(result_rows) % 50 == 0:
                print(f"checked {len(result_rows)} tensors", flush=True)

    result = {
        "checkpoint_revision": manifest["revision"],
        "method": "source packed tensor -> float32 represented values -> fixed original scales -> same Gemma packing",
        "chunk_rows": rows_per_chunk,
        "quantized_weight_tensors": len(result_rows),
        "weight_tensors_by_bits": {str(k): v for k, v in sorted(bit_counts.items())},
        "packed_bytes": total_bytes,
        "changed_packed_bytes": changed_bytes,
        "changed_fraction": changed_bytes / total_bytes if total_bytes else 0.0,
        "bit_exact": changed_bytes == 0,
        "max_reconstruction_error": max_reconstruction_error,
        "tensor_results": result_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    args = parser.parse_args()
    result = round_trip(args.weights, args.manifest, args.output, args.rows_per_chunk)
    print(json.dumps({k: v for k, v in result.items() if k != "tensor_results"}, indent=2))


if __name__ == "__main__":
    main()
