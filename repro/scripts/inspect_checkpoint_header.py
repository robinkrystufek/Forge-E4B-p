"""Fetch only public Gemma mobile-QAT metadata and a safetensors header."""

from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPO = "google/gemma-4-E4B-it-qat-mobile-transformers"
DEFAULT_REVISION = "8e3ac78f1a4a97c19f5f194ab362a2206f71d961"
MAX_HEADER_BYTES = 16 * 1024 * 1024


def parse_safetensors_header(prefix: bytes) -> dict[str, Any]:
    """Parse a header from an 8-byte length prefix plus JSON header bytes."""
    if len(prefix) < 8:
        raise ValueError("safetensors prefix must contain the 8-byte header length")
    (header_size,) = struct.unpack("<Q", prefix[:8])
    if header_size > MAX_HEADER_BYTES:
        raise ValueError(f"unreasonably large safetensors header: {header_size}")
    if len(prefix) < 8 + header_size:
        raise ValueError(f"incomplete header: need {8 + header_size} bytes, got {len(prefix)}")
    parsed = json.loads(prefix[8 : 8 + header_size])
    if not isinstance(parsed, dict):
        raise ValueError("safetensors header must be a JSON object")
    return parsed


def quant_bits_for_tensor(name: str, spec: dict[str, Any], quantization_config: dict[str, Any]) -> int | None:
    """Apply the checkpoint's ordered module regex overrides to packed weight tensors."""
    if spec.get("dtype") not in ("U8", "I8"):
        return None
    if name.endswith(".weight"):
        module_name = name.removesuffix(".weight")
    elif name.endswith(".embedding_quantized"):
        module_name = name.removesuffix(".embedding_quantized")
    else:
        return None
    module_name = module_name.removeprefix("model.")
    bits = int(quantization_config["num_bits"])
    for pattern, override in quantization_config.get("module_quant_configs", {}).items():
        if re.search(pattern, module_name):
            bits = int(override["num_bits"])
            break
    return bits


def _get(url: str, byte_range: tuple[int, int] | None = None) -> tuple[bytes, dict[str, str]]:
    headers = {"User-Agent": "Forge-GM-mobile-QAT-feasibility/0.1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        if byte_range is not None:
            content_range = response_headers.get("content-range", "")
            if response.status != 206 or not content_range.startswith(f"bytes {byte_range[0]}-"):
                raise RuntimeError(f"server did not honor byte range {byte_range}: {response.status}, {content_range!r}")
        return body, response_headers


def inspect(repo: str, revision: str) -> dict[str, Any]:
    config_url = f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
    config_bytes, _ = _get(config_url)
    config = json.loads(config_bytes)
    model_url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors"
    prefix, headers = _get(model_url, (0, 7))
    (header_size,) = struct.unpack("<Q", prefix)
    if header_size > MAX_HEADER_BYTES:
        raise ValueError(f"unreasonably large safetensors header: {header_size}")
    header_bytes, _ = _get(model_url, (8, 7 + header_size))
    header = json.loads(header_bytes)
    tensors = {name: spec for name, spec in header.items() if name != "__metadata__"}
    quantization_config = config.get("quantization_config", {})
    assignment_counts: Counter[int] = Counter()
    for name, spec in tensors.items():
        bits = quant_bits_for_tensor(name, spec, quantization_config)
        if bits is not None:
            spec["assigned_bits"] = bits
            assignment_counts[bits] += 1
    dtype_counts = Counter(spec["dtype"] for spec in tensors.values())
    content_range = headers.get("content-range", "")
    total = content_range.rsplit("/", 1)[-1] if "/" in content_range else "0"
    return {
        "repo_id": repo,
        "revision": revision,
        "config": config,
        "safetensors": {
            "filename": "model.safetensors",
            "etag": headers.get("etag", "").strip('"'),
            "content_length_bytes": int(total),
            "header_bytes": header_size,
            "tensor_count": len(tensors),
            "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
            "quantized_weight_tensor_counts_by_bits": {str(k): v for k, v in sorted(assignment_counts.items())},
            "weight_scale_tensor_count": sum(name.endswith((".weight_scale", ".embedding_scale")) for name in tensors),
            "activation_scale_tensor_count": sum(name.endswith((".input_activation_scale", ".output_activation_scale")) for name in tensors),
            "tensor_headers": tensors,
            "metadata": header.get("__metadata__", {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.repo, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "tensors": result["safetensors"]["tensor_count"],
        "dtypes": result["safetensors"]["dtype_tensor_counts"],
        "model_bytes": result["safetensors"]["content_length_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
