"""Patch changed mobile-QAT payloads into an official GPU-capable TFLite graph.

The graph topology and all metadata remain byte-for-byte from the official
LiteRT-LM graph. Only source-audited packed weight payload ranges are changed.
The input graph and package are never modified.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(data: bytes | memoryview | np.ndarray) -> str:
    h = hashlib.sha256()
    if isinstance(data, np.ndarray):
        h.update(memoryview(np.ascontiguousarray(data)).cast("B"))
    else:
        h.update(data)
    return h.hexdigest()


def _decode_tflite(raw: bytes | memoryview | np.ndarray, bits: int, count: int) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.uint8)
    if bits == 8:
        values = data.view(np.int8)
    elif bits in (2, 4):
        per_byte = 8 // bits
        mask = (1 << bits) - 1
        sign = 1 << (bits - 1)
        values = np.empty(data.size * per_byte, dtype=np.int8)
        for i in range(per_byte):
            part = ((data >> (i * bits)) & mask).astype(np.int16)
            part[part >= sign] -= 1 << bits
            values[i::per_byte] = part.astype(np.int8)
    else:
        raise ValueError(f"unsupported bit width: {bits}")
    if values.size < count:
        raise ValueError(f"short packed tensor: {values.size} values for {count}")
    return np.asarray(values[:count])


def _buffer_from_id(graph_tensor_id: str) -> tuple[int, int, tuple[int, ...], str]:
    match = re.search(r":buffer(\d+):INT([248]):([0-9x]+):([0-9a-f]+)$", graph_tensor_id)
    if match is None:
        raise ValueError(f"unrecognized graph tensor id: {graph_tensor_id}")
    return (int(match.group(1)), int(match.group(2)),
            tuple(int(x) for x in match.group(3).split("x")), match.group(4))


def _graph_schema(path: Path, buffer_ids: set[int]):
    from ai_edge_litert import schema_py_generated as schema

    with path.open("rb") as fp:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAsModel(mm, 0)
        found: dict[int, dict[str, Any]] = {}
        operator_uses: dict[int, dict[str, dict[str, Any]]] = {bid: {} for bid in buffer_ids}
        type_names = {getattr(schema.TensorType, n): n for n in dir(schema.TensorType)
                      if not n.startswith("_")}
        builtin_names = {getattr(schema.BuiltinOperator, n): n for n in dir(schema.BuiltinOperator)
                         if not n.startswith("_")}
        for sgi in range(model.SubgraphsLength()):
            sg = model.Subgraphs(sgi)
            tensors = [sg.Tensors(i) for i in range(sg.TensorsLength())]
            for ti in range(sg.TensorsLength()):
                tensor = tensors[ti]
                buffer_id = int(tensor.Buffer())
                if buffer_id not in buffer_ids:
                    continue
                type_name = type_names.get(tensor.Type(), str(tensor.Type()))
                q = tensor.Quantization()
                scales = (np.asarray(q.ScaleAsNumpy(), dtype=np.float32)
                          if q is not None else np.empty(0, dtype=np.float32))
                info = {
                    "bits": int(type_name[3:]) if type_name.startswith("INT") else None,
                    "shape": tuple(int(x) for x in tensor.ShapeAsNumpy()),
                    "scale_sha256": _sha256(scales),
                    "quantized_dimension": int(q.QuantizedDimension()) if q is not None else None,
                    "tensor_index": ti,
                    "subgraph_index": sgi,
                }
                prior = found.get(buffer_id)
                if prior is not None and (prior["bits"], prior["shape"], prior["scale_sha256"],
                                           prior["quantized_dimension"]) != (
                        info["bits"], info["shape"], info["scale_sha256"], info["quantized_dimension"]):
                    raise ValueError(f"buffer {buffer_id} is referenced with inconsistent tensor metadata")
                found.setdefault(buffer_id, info)
            for oi in range(sg.OperatorsLength()):
                op = sg.Operators(oi)
                code = model.OperatorCodes(op.OpcodeIndex())
                opname = builtin_names.get(code.BuiltinCode(), f"CUSTOM_{code.CustomCode()}")
                inputs = [int(x) for x in op.InputsAsNumpy()]
                if opname != "FULLY_CONNECTED" or len(inputs) < 2 or inputs[1] < 0:
                    continue
                weight = tensors[inputs[1]]
                buffer_id = int(weight.Buffer())
                if buffer_id not in buffer_ids:
                    continue

                def qinfo(index: int) -> dict[str, Any] | None:
                    if index < 0 or index >= len(tensors):
                        return None
                    tensor = tensors[index]
                    q = tensor.Quantization()
                    scales = (np.asarray(q.ScaleAsNumpy(), dtype=np.float32)
                              if q is not None else np.empty(0, dtype=np.float32))
                    return {
                        "type": type_names.get(tensor.Type(), str(tensor.Type())),
                        "shape": [int(x) for x in tensor.ShapeAsNumpy()],
                        "scale_sha256": _sha256(scales),
                        "scale_count": int(scales.size),
                        "zero_points": (np.asarray(q.ZeroPointAsNumpy(), dtype=np.int64).tolist()
                                        if q is not None else []),
                        "quantized_dimension": int(q.QuantizedDimension()) if q is not None else None,
                    }

                use = {
                    "operator": opname,
                    "operator_version": int(code.Version()),
                    "weight_tensor_name": weight.Name().decode(errors="replace") if weight.Name() else "",
                    "activation_input": qinfo(inputs[0]) if inputs else None,
                    "additional_inputs": [qinfo(i) for i in inputs[2:]],
                    "outputs": [qinfo(int(i)) for i in op.OutputsAsNumpy()],
                }
                key = json.dumps(use, sort_keys=True, separators=(",", ":"))
                if key in operator_uses[buffer_id]:
                    operator_uses[buffer_id][key]["occurrence_count"] += 1
                else:
                    operator_uses[buffer_id][key] = {
                        **use,
                        "representative_subgraph": sgi,
                        "representative_operator_index": oi,
                        "occurrence_count": 1,
                    }
        try:
            mm.close()
        except BufferError:
            pass
    missing = buffer_ids - found.keys()
    if missing:
        raise ValueError(f"weight buffers have no tensor metadata in {path}: {sorted(missing)}")
    return found, {bid: list(uses.values()) for bid, uses in operator_uses.items()}


def _buffer_ranges(path: Path, buffer_ids: set[int]):
    from ai_edge_litert import schema_py_generated as schema

    ranges = {}
    with path.open("rb") as fp:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAsModel(mm, 0)
        for buffer_id in buffer_ids:
            buffer = model.Buffers(buffer_id)
            size, offset = int(buffer.Size()), int(buffer.Offset())
            if size <= 0 or offset <= 0 or offset + size > len(mm):
                raise ValueError(f"invalid external buffer range {buffer_id}: offset={offset} size={size}")
            ranges[buffer_id] = (offset, size)
        try:
            mm.close()
        except BufferError:
            pass
    return ranges


def build_patch_plan(report: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint = report["checkpoint"]
    changed_names = set(checkpoint["modified_weight_tensors"])
    rows = {}
    for graph_name in ("official_main", "qat_main"):
        graph_rows = {row["source_tensor"]: row
                      for row in report["source_tensor_matches_by_graph"][graph_name]}
        rows[graph_name] = graph_rows
    plan = []
    for name in sorted(changed_names):
        off = rows["official_main"].get(name)
        qat = rows["qat_main"].get(name)
        if off is None or qat is None:
            raise ValueError(f"modified source tensor missing from graph audit: {name}")
        if not off["source_changed"] or off["status"] != "base_only_update_not_retained":
            raise ValueError(f"official graph is not an exact base for {name}: {off['status']}")
        if not qat["source_changed"] or qat["status"] != "modified_values_and_scales_exact":
            raise ValueError(f"QAT export did not retain the modified tensor exactly: {name}: {qat['status']}")
        base_matches = [m for m in off["matches"] if m["revision_match"] == "base"]
        mod_matches = [m for m in qat["matches"] if m["revision_match"] == "modified"]
        if len(base_matches) != 1 or len(mod_matches) != 1:
            raise ValueError(f"expected one unambiguous tensor match per graph for {name}")
        om, qm = base_matches[0], mod_matches[0]
        if not om["scale_exact"] or not qm["scale_exact"]:
            raise ValueError(f"checkpoint scale mismatch for {name}")
        if om["graph_scales_sha256"] != qm["graph_scales_sha256"]:
            raise ValueError(f"official and QAT graph scales differ for {name}")
        ob, obits, oshape, _ = _buffer_from_id(om["graph_tensor_id"])
        qb, qbits, qshape, _ = _buffer_from_id(qm["graph_tensor_id"])
        if (obits, oshape) != (qbits, qshape) or obits != int(qat["source_bits"]):
            raise ValueError(f"weight type/shape mismatch for {name}")
        plan.append({
            "source_tensor": name,
            "bits": obits,
            "logical_shape": list(oshape),
            "official_buffer": ob,
            "qat_buffer": qb,
            "official_scale_sha256": om["graph_scales_sha256"],
            "qat_scale_sha256": qm["graph_scales_sha256"],
            "base_integer_values_sha256": off["base_integer_values_sha256"],
            "modified_integer_values_sha256": qat["modified_integer_values_sha256"],
            "official_operator_uses": None,
            "qat_operator_uses": None,
        })
    return plan


def patch_graph(official_graph: Path, qat_graph: Path, output_graph: Path,
                report_path: Path, manifest_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plan = build_patch_plan(report)
    official_ids = {x["official_buffer"] for x in plan}
    qat_ids = {x["qat_buffer"] for x in plan}
    official_meta, official_ops = _graph_schema(official_graph, official_ids)
    qat_meta, qat_ops = _graph_schema(qat_graph, qat_ids)
    for item in plan:
        item["official_operator_uses"] = official_ops[item["official_buffer"]]
        item["qat_operator_uses"] = qat_ops[item["qat_buffer"]]
    official_ranges = _buffer_ranges(official_graph, official_ids)
    qat_ranges = _buffer_ranges(qat_graph, qat_ids)

    with official_graph.open("rb") as foff, qat_graph.open("rb") as fqat:
        omm = mmap.mmap(foff.fileno(), 0, access=mmap.ACCESS_READ)
        qmm = mmap.mmap(fqat.fileno(), 0, access=mmap.ACCESS_READ)
        for item in plan:
            shape = tuple(item["logical_shape"])
            for meta, bid, expected_scale in (
                (official_meta, item["official_buffer"], item["official_scale_sha256"]),
                (qat_meta, item["qat_buffer"], item["qat_scale_sha256"]),
            ):
                actual = meta[bid]
                if (actual["bits"], actual["shape"], actual["scale_sha256"],
                        actual["quantized_dimension"]) != (
                        item["bits"], shape, expected_scale, 0):
                    raise ValueError(f"TFLite tensor metadata changed since audit: {item['source_tensor']} {actual}")
            oo, osz = official_ranges[item["official_buffer"]]
            qo, qsz = qat_ranges[item["qat_buffer"]]
            if osz != qsz:
                raise ValueError(f"packed payload size mismatch for {item['source_tensor']}")
            count = int(np.prod(shape))
            base_hash = _sha256(_decode_tflite(omm[oo:oo + osz], item["bits"], count))
            qat_hash = _sha256(_decode_tflite(qmm[qo:qo + qsz], item["bits"], count))
            if base_hash != item["base_integer_values_sha256"]:
                raise ValueError(f"official payload no longer matches the exact base for {item['source_tensor']}")
            if qat_hash != item["modified_integer_values_sha256"]:
                raise ValueError(f"QAT payload no longer matches the modified source for {item['source_tensor']}")
            item["official_payload_sha256_before"] = _sha256(omm[oo:oo + osz])
            item["qat_payload_sha256"] = _sha256(qmm[qo:qo + qsz])
            item["_official_range"] = (oo, osz)
            item["_qat_range"] = (qo, qsz)
        qmm.close()
        omm.close()

    if output_graph.exists():
        raise FileExistsError(f"refusing to overwrite patched graph: {output_graph}")
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(official_graph, output_graph)
    try:
        with output_graph.open("r+b") as fp:
            mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_WRITE)
            base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            for item in plan:
                start, size = item.pop("_official_range")
                qat_start, qat_size = item.pop("_qat_range")
                with qat_graph.open("rb") as fq:
                    fq.seek(qat_start)
                    payload = fq.read(qat_size)
                if len(payload) != size:
                    raise IOError(f"short payload read for {item['source_tensor']}")
                mm[start:start + size] = payload
            mm.flush()
            mm.close()
        with output_graph.open("rb") as fp:
            mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            for item in plan:
                start, size = official_ranges[item["official_buffer"]]
                count = int(np.prod(item["logical_shape"]))
                actual = _sha256(_decode_tflite(mm[start:start + size], item["bits"], count))
                if actual != item["modified_integer_values_sha256"]:
                    raise ValueError(f"patched output failed read-back for {item['source_tensor']}")
            mm.close()
    except BaseException:
        # Keep this task-created file for diagnosis if patch/write verification fails.
        raise

    digest = hashlib.sha256()
    with output_graph.open("rb") as fp:
        for block in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(block)
    result = {
        "official_graph": str(official_graph),
        "qat_graph": str(qat_graph),
        "patched_graph": str(output_graph),
        "patched_graph_sha256": digest.hexdigest(),
        "official_graph_sha256_before_patch": _sha256_file(official_graph),
        "audit_report": str(report_path),
        "provenance_manifest": str(manifest_path),
        "patched_tensor_count": len(plan),
        "patched_tensors": plan,
        "all_nonpayload_graph_bytes_preserved": True,
        "validation": "source hashes, TFLite types/shapes/scales/axes, original values, and patched read-back values passed",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-graph", type=Path, required=True)
    parser.add_argument("--qat-graph", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(patch_graph(args.official_graph, args.qat_graph, args.output_graph,
                                 args.audit_report, args.manifest), indent=2))


if __name__ == "__main__":
    main()
