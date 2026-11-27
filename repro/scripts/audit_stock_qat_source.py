"""Compare retained source-QAT packed tensors to mapped official graph payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open

from patch_official_qat_graph import _decode_tflite, _graph_schema, _sha256


def unpack_signed_np(packed: np.ndarray, width: int, bits: int) -> np.ndarray:
    packed = np.asarray(packed)
    if bits == 8:
        return packed.astype(np.int8, copy=False)[..., :width]
    if bits not in (2, 4):
        raise ValueError(f"unsupported quantization width: {bits}")
    factor = 8 // bits
    mask = (1 << bits) - 1
    offset = 1 << (bits - 1)
    u = packed.astype(np.uint8, copy=False)
    parts = [((u >> (i * bits)) & mask).astype(np.int16) - offset for i in range(factor)]
    return np.stack(parts, axis=-1).reshape(*u.shape[:-1], -1)[..., :width].astype(np.int8)


def scale_key(weight_key: str) -> str:
    if weight_key.endswith(".embedding_quantized"):
        return weight_key.removesuffix(".embedding_quantized") + ".embedding_scale"
    return weight_key.removesuffix(".weight") + ".weight_scale"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_candidates(graph: Path) -> tuple[dict, dict]:
    import mmap
    from ai_edge_litert import schema_py_generated as schema

    type_names = {getattr(schema.TensorType, n): n for n in dir(schema.TensorType) if not n.startswith("_")}
    candidates = {}
    locations = {}
    with graph.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAsModel(mm, 0)
        for gi in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(gi)
            for ti in range(subgraph.TensorsLength()):
                tensor = subgraph.Tensors(ti)
                bid = int(tensor.Buffer())
                q = tensor.Quantization()
                type_name = type_names.get(tensor.Type(), "")
                if bid == 0 or q is None or not type_name.startswith("INT"):
                    continue
                bits = int(type_name[3:])
                shape = tuple(int(x) for x in tensor.ShapeAsNumpy())
                scales = np.array(q.ScaleAsNumpy(), dtype=np.float32, copy=True)
                sig = (bits, shape, _sha256(scales))
                candidates.setdefault(sig, set()).add(bid)
                locations.setdefault(bid, tensor.Name().decode(errors="replace") if tensor.Name() else "")
        del model, subgraph, tensor, q, scales
        mm.close()
    return candidates, locations


def audit(source_dir: Path, graphs: dict[str, Path], provenance: Path, output: Path) -> dict:
    provenance_rows = json.loads(provenance.read_text(encoding="utf-8"))["source_tensor_matches_by_graph"]
    source_file = source_dir / "model.safetensors"
    source_names = set()
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))["text_config"]
    layer_count = int(config["num_hidden_layers"])
    shared_kv_layers = int(config["num_kv_shared_layers"])
    shared_start = layer_count - shared_kv_layers
    matches_by_graph = {}
    missing_source = []
    skipped_graph = {}
    with safe_open(str(source_file), framework="numpy") as source:
        source_names = set(source.keys())
        for graph_name, graph in graphs.items():
            matches = []
            skipped = []
            graph_ids = set()
            for row in provenance_rows[graph_name]:
                base = [m for m in row.get("matches", []) if m.get("revision_match") == "base"]
                if len(base) != 1:
                    skipped.append(row["source_tensor"])
                    continue
                name = row["source_tensor"]
                match = base[0]
                parsed = re.search(r":buffer(\d+):INT([248]):([0-9x]+):", match["graph_tensor_id"])
                if not parsed:
                    raise ValueError(f"unrecognized graph tensor ID: {match['graph_tensor_id']}")
                buffer_id, bits = int(parsed.group(1)), int(parsed.group(2))
                shape = tuple(int(x) for x in parsed.group(3).split("x"))
                if bits != int(row["source_bits"]) or list(shape) != row["logical_shape"]:
                    raise ValueError(f"source/graph quantization metadata mismatch: {name}")

                slab = re.search(r"#slab(\d+)$", name)
                tensor_name = name[:slab.start()] if slab else name
                if tensor_name not in source_names:
                    missing_source.append(name)
                    continue
                packed = source.get_tensor(tensor_name)
                scales = source.get_tensor(scale_key(tensor_name))
                if slab:
                    slab_index = int(slab.group(1))
                    if packed.ndim != 2 or scales.ndim != 2 or slab_index >= scales.shape[1]:
                        raise ValueError(f"unexpected per-layer embedding slab layout: {name}")
                    packed = packed[:, slab_index * 64:(slab_index + 1) * 64]
                    scales = scales[:, slab_index]
                logical = unpack_signed_np(packed, shape[-1], bits)
                source_integer_hash = _sha256(logical)
                source_scale_hash = _sha256(np.asarray(scales, dtype=np.float32))
                if source_integer_hash != row["base_integer_values_sha256"]:
                    raise ValueError(f"pinned source packed values differ from provenance: {name}")
                if source_scale_hash != match["graph_scales_sha256"] or not match["scale_exact"]:
                    raise ValueError(f"source scale differs from official graph: {name}")
                graph_ids.add(buffer_id)
                matches.append({
                    "source_tensor": name,
                    "bits": bits,
                    "logical_shape": list(shape),
                    "official_buffer": buffer_id,
                    "source_integer_sha256": source_integer_hash,
                    "official_integer_sha256_expected": row["base_integer_values_sha256"],
                    "scale_sha256": source_scale_hash,
                })

            metadata, _ = _graph_schema(graph, graph_ids)
            with graph.open("rb") as f:
                import mmap
                from ai_edge_litert import schema_py_generated as schema
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                model = schema.Model.GetRootAsModel(mm, 0)
                for item in matches:
                    info = metadata[item["official_buffer"]]
                    if (info["bits"], list(info["shape"]), info["scale_sha256"], info["quantized_dimension"]) != (
                        item["bits"], item["logical_shape"], item["scale_sha256"], 0
                    ):
                        raise ValueError(f"official graph metadata mismatch: {item['source_tensor']}")
                    buffer = model.Buffers(item["official_buffer"])
                    offset, size = int(buffer.Offset()), int(buffer.Size())
                    raw = mm[offset:offset + size] if offset > 0 else np.asarray(buffer.DataAsNumpy(), dtype=np.uint8)
                    actual = _sha256(_decode_tflite(raw, item["bits"], int(np.prod(item["logical_shape"]))))
                    del raw, buffer
                    if actual != item["official_integer_sha256_expected"]:
                        raise ValueError(f"official graph integer values differ from source: {item['source_tensor']}")
                    item["official_integer_sha256"] = actual
                    item["integer_values_exact"] = True
                del model
                mm.close()
            matches_by_graph[graph_name] = {"exact_tensor_count": len(matches), "distinct_buffer_count": len(graph_ids), "matches": matches}
            skipped_graph[graph_name] = skipped

        # Reconcile source tensors omitted by name-based provenance mapping when
        # their exact quantization signature identifies a packed graph buffer.
        graph = graphs.get("official_main")
        if graph is not None:
            candidates, locations = graph_candidates(graph)
            recovered = []
            for row in provenance_rows["official_main"]:
                name = row["source_tensor"]
                if name not in skipped_graph["official_main"] or name not in source_names or "#slab" in name:
                    continue
                bits = int(row["source_bits"])
                shape = tuple(int(x) for x in row["logical_shape"])
                if bits not in (2, 4, 8) or len(shape) != 2:
                    continue
                packed = source.get_tensor(name)
                scales = source.get_tensor(scale_key(name))
                source_scale_hash = _sha256(np.asarray(scales, dtype=np.float32))
                if source_scale_hash != row["base_scale_sha256"]:
                    raise ValueError(f"source scale does not match provenance for unmapped tensor: {name}")
                source_hash = _sha256(unpack_signed_np(packed, shape[-1], bits))
                if source_hash != row["base_integer_values_sha256"]:
                    raise ValueError(f"source values do not match provenance for unmapped tensor: {name}")
                possible = candidates.get((bits, shape, source_scale_hash), set())
                if not possible:
                    continue
                exact = []
                for bid in possible:
                    buf = None
                    with graph.open("rb") as f:
                        import mmap
                        from ai_edge_litert import schema_py_generated as schema
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        model = schema.Model.GetRootAsModel(mm, 0)
                        buf = model.Buffers(bid)
                        offset, size = int(buf.Offset()), int(buf.Size())
                        raw = mm[offset:offset + size] if offset > 0 else np.asarray(buf.DataAsNumpy(), dtype=np.uint8)
                        actual = _sha256(_decode_tflite(raw, bits, int(np.prod(shape))))
                        del raw, buf, model
                        mm.close()
                    if actual == source_hash:
                        exact.append({"buffer": bid, "tensor_name": locations.get(bid, "")})
                if exact:
                    recovered.append({"source_tensor": name, "bits": bits, "logical_shape": list(shape),
                                      "integer_sha256": source_hash, "scale_sha256": source_scale_hash,
                                      "exact_official_buffers": exact})
            matches_by_graph["official_main"]["reconciled_unmapped_tensor_count"] = len(recovered)
            matches_by_graph["official_main"]["reconciled_unmapped_tensors"] = recovered

    shared_kv = []
    for name in skipped_graph.get("official_main", []):
        match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.self_attn\.([kv])_proj\.weight", name)
        if match and int(match.group(1)) >= shared_start:
            shared_kv.append(name)
    expected_shared_kv = {
        f"model.language_model.layers.{layer}.self_attn.{proj}_proj.weight"
        for layer in range(shared_start, layer_count) for proj in ("k", "v")
    }
    if set(shared_kv) != expected_shared_kv:
        raise ValueError(f"unexpected source tensors absent from official graph: shared-KV candidates {len(shared_kv)} != {len(expected_shared_kv)}")

    logical_source_tensors = {}
    for graph_result in matches_by_graph.values():
        for item in graph_result["matches"]:
            logical_name = re.sub(r"#slab\d+$", "", item["source_tensor"])
            logical_source_tensors[logical_name] = item["bits"]
    retained_by_bits = {}
    for bits in logical_source_tensors.values():
        retained_by_bits[str(bits)] = retained_by_bits.get(str(bits), 0) + 1

    result = {
        "status": "mapped_retained_source_tensors_exact",
        "source_checkpoint_revision": "8e3ac78f1a4a97c19f5f194ab362a2206f71d961",
        "source_safetensors_sha256": file_sha256(source_file),
        "official_graph_sha256": {name: file_sha256(path) for name, path in graphs.items()},
        "provenance": str(provenance),
        "mapped_exact_tensor_count": sum(x["exact_tensor_count"] + x.get("reconciled_unmapped_tensor_count", 0) for x in matches_by_graph.values()),
        "unique_retained_source_tensor_count": len(logical_source_tensors),
        "unique_retained_source_tensors_by_bits": retained_by_bits,
        "shared_kv_config": {"num_hidden_layers": layer_count, "num_kv_shared_layers": shared_kv_layers,
                             "shared_layer_range_zero_based": [shared_start, layer_count - 1]},
        "unretained_source_kv_tensors_explained_by_shared_kv_topology": sorted(shared_kv),
        "mapped_by_graph": matches_by_graph,
        "source_tensors_without_official_mapping": skipped_graph,
        "mapped_source_tensors_missing_from_checkpoint_file": missing_source,
        "matches": matches,
        "scope_note": "This verifies exact packed integer values and scales for provenance-mapped source tensors. It does not establish a mapping for unmatched source tensors.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--official-graph", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graphs = {}
    for item in args.official_graph:
        name, sep, path = item.partition("=")
        if not sep or name not in ("official_main", "official_embedder", "official_per_layer_embedder"):
            parser.error(f"invalid --official-graph {item!r}")
        graphs[name] = Path(path)
    print(json.dumps(audit(args.source_dir, graphs, args.provenance, args.output), indent=2))


if __name__ == "__main__":
    main()
