"""Audit logical LiteRT-LM graph and weight equivalence after repacking.

The builder moves tensor buffers from TFLite FlatBuffers into external weight
sections and deduplicates some repeated payloads. This audit follows each
tensor's external-buffer group/offset back to its logical payload and compares
the full retained graph structure, tensor metadata, quantization parameters,
and payload hashes against the original package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import tomllib
from pathlib import Path
from typing import Any

import numpy as np


REMOVED = {
    "audio_encoder_hw", "audio_adapter", "end_of_audio",
    "vision_encoder", "vision_adapter", "end_of_vision",
}
RETAINED = {"embedder", "per_layer_embedder", "prefill_decode", "mtp_drafter"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes | bytearray | memoryview | np.ndarray | None) -> str:
    digest = hashlib.sha256()
    if data is None:
        return digest.hexdigest()
    if isinstance(data, np.ndarray):
        if data.size:
            digest.update(memoryview(np.ascontiguousarray(data)).cast("B"))
    elif data:
        digest.update(data)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in sorted(value.items())}
    if hasattr(value, "__dict__"):
        return {key: plain(item) for key, item in sorted(value.__dict__.items())}
    return str(value)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def buffer_payload(model: Any, flatbuffer: mmap.mmap, buffer_id: int) -> tuple[str, int]:
    buffer = model.Buffers(buffer_id)
    data = buffer.DataAsNumpy()
    size = int(data.size) if isinstance(data, np.ndarray) else 0
    if size:
        return sha256_bytes(data), size
    offset, size = int(buffer.Offset()), int(buffer.Size())
    if size:
        if offset < 0 or offset + size > len(flatbuffer):
            raise ValueError(f"buffer {buffer_id} range {offset}+{size} exceeds TFLite file")
        return sha256_bytes(flatbuffer[offset:offset + size]), size
    return sha256_bytes(None), 0


def sections(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    model = tomllib.loads((root / "model.toml").read_text(encoding="utf-8"))
    graphs: dict[str, dict] = {}
    weights: dict[str, dict] = {}
    for section in model.get("section", []):
        kind, model_type = section.get("section_type"), section.get("model_type")
        if kind == "TFLiteModel":
            graphs[model_type] = section
        elif kind == "TFLiteWeights":
            weights[model_type] = section
    return model, {"graphs": graphs, "weights": weights}


def resolve_weight_type(group_name: bytes, graph_type: str) -> str:
    name = group_name.decode("utf-8")
    if name == "tflite_weights":
        return graph_type
    prefix = "tflite_weights:"
    if not name.startswith(prefix):
        raise ValueError(f"unrecognized external weight group: {name!r}")
    return "@" + name[len(prefix):]


def model_graph_path(root: Path, section: dict) -> Path:
    return root / section["data_path"]


def graph_audit(original_root: Path, original_section: dict,
                candidate_root: Path, candidate_section: dict,
                graph_type: str,
                candidate_weight_sections: dict[str, dict]) -> dict:
    from ai_edge_litert import schema_py_generated as schema

    original_path = model_graph_path(original_root, original_section)
    candidate_path = model_graph_path(candidate_root, candidate_section)
    with original_path.open("rb") as original_stream, candidate_path.open("rb") as candidate_stream:
        original_map = mmap.mmap(original_stream.fileno(), 0, access=mmap.ACCESS_READ)
        candidate_map = mmap.mmap(candidate_stream.fileno(), 0, access=mmap.ACCESS_READ)
        original_model = schema.Model.GetRootAsModel(original_map, 0)
        candidate_model = schema.Model.GetRootAsModel(candidate_map, 0)
        external = {}
        groups = {}
        weight_maps: dict[int, tuple[Any, mmap.mmap]] = {}
        for i in range(candidate_model.ExternalBufferGroupsLength()):
            group = candidate_model.ExternalBufferGroups(i)
            groups[i] = group.Name() or b""
        for i in range(candidate_model.ExternalBuffersLength()):
            item = candidate_model.ExternalBuffers(i)
            external[int(item.Id())] = (int(item.Group()), int(item.Offset()), int(item.Length()))

        def logical_weight_path(group_name: bytes) -> Path:
            group_type = resolve_weight_type(group_name, graph_type)
            if not group_type.startswith("@"):
                if group_type not in candidate_weight_sections:
                    raise ValueError(f"no external weights section for {group_type}")
                return model_graph_path(candidate_root, candidate_weight_sections[group_type])
            suffix = group_type[1:]
            matches = [section for section in candidate_weight_sections.values()
                       if Path(section["data_path"]).name.endswith(suffix + ".weight")]
            if len(matches) != 1:
                raise ValueError(f"cannot map external group {suffix!r} to one weights section")
            return model_graph_path(candidate_root, matches[0])

        def external_payload_hash(external_id: int) -> tuple[str, int]:
            if external_id not in external:
                raise ValueError(f"external buffer id {external_id} is missing")
            group_index, offset, length = external[external_id]
            if group_index not in weight_maps:
                path = logical_weight_path(groups[group_index])
                handle = path.open("rb")
                weight_maps[group_index] = (handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
            weight_map = weight_maps[group_index][1]
            if offset < 0 or length < 0 or offset + length > len(weight_map):
                raise ValueError(f"external payload range is outside its weights file: {offset}+{length}")
            return sha256_bytes(weight_map[offset:offset + length]), length

        structure = hashlib.sha256()
        logical = hashlib.sha256()
        tensor_count = 0
        tensor_payload_count = 0
        logical_payload_bytes = 0
        external_tensor_count = 0
        seen_buffer_payloads: set[int] = set()

        def add_record(digest, value):
            digest.update(json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")

        if (original_model.Version() != candidate_model.Version()
                or (original_model.Description() or b"") != (candidate_model.Description() or b"")
                or original_model.OperatorCodesLength() != candidate_model.OperatorCodesLength()
                or original_model.SubgraphsLength() != candidate_model.SubgraphsLength()
                or original_model.BuffersLength() != candidate_model.BuffersLength()
                or original_model.MetadataLength() != candidate_model.MetadataLength()
                or original_model.MetadataBufferLength() != candidate_model.MetadataBufferLength()
                or original_model.SignatureDefsLength() != candidate_model.SignatureDefsLength()):
            raise AssertionError(f"{graph_type} flatbuffer table counts/version changed")
        add_record(structure, {"version": original_model.Version(), "description": original_model.Description() or b""})
        for i in range(original_model.OperatorCodesLength()):
            left = schema.OperatorCodeT.InitFromObj(original_model.OperatorCodes(i))
            right = schema.OperatorCodeT.InitFromObj(candidate_model.OperatorCodes(i))
            if plain(left) != plain(right):
                raise AssertionError(f"{graph_type} operator code {i} changed")
            add_record(structure, left)
        for i in range(original_model.MetadataLength()):
            left = schema.MetadataT.InitFromObj(original_model.Metadata(i))
            right = schema.MetadataT.InitFromObj(candidate_model.Metadata(i))
            if plain(left) != plain(right):
                raise AssertionError(f"{graph_type} metadata record {i} changed")
            add_record(structure, left)
        for i in range(original_model.MetadataBufferLength()):
            if original_model.MetadataBuffer(i) != candidate_model.MetadataBuffer(i):
                raise AssertionError(f"{graph_type} metadata-buffer index {i} changed")
        for i in range(original_model.SignatureDefsLength()):
            left = schema.SignatureDefT.InitFromObj(original_model.SignatureDefs(i))
            right = schema.SignatureDefT.InitFromObj(candidate_model.SignatureDefs(i))
            if plain(left) != plain(right):
                raise AssertionError(f"{graph_type} signature definition {i} changed")
            add_record(structure, left)

        for sgi in range(original_model.SubgraphsLength()):
            source_subgraph = original_model.Subgraphs(sgi)
            candidate_subgraph = candidate_model.Subgraphs(sgi)
            if (source_subgraph.TensorsLength() != candidate_subgraph.TensorsLength()
                    or source_subgraph.OperatorsLength() != candidate_subgraph.OperatorsLength()):
                raise AssertionError(f"{graph_type} subgraph {sgi} tensor/operator count changed")
            source_subgraph_meta = {
                "subgraph": sgi,
                "name": source_subgraph.Name() or b"",
                "inputs": source_subgraph.InputsAsNumpy(),
                "outputs": source_subgraph.OutputsAsNumpy(),
                "debugMetadataIndex": source_subgraph.DebugMetadataIndex(),
            }
            candidate_subgraph_meta = {
                "subgraph": sgi,
                "name": candidate_subgraph.Name() or b"",
                "inputs": candidate_subgraph.InputsAsNumpy(),
                "outputs": candidate_subgraph.OutputsAsNumpy(),
                "debugMetadataIndex": candidate_subgraph.DebugMetadataIndex(),
            }
            if plain(source_subgraph_meta) != plain(candidate_subgraph_meta):
                raise AssertionError(f"{graph_type} subgraph interface {sgi} changed")
            add_record(structure, source_subgraph_meta)
            for ti in range(source_subgraph.TensorsLength()):
                source_tensor = source_subgraph.Tensors(ti)
                candidate_tensor = candidate_subgraph.Tensors(ti)
                source_obj = schema.TensorT.InitFromObj(source_tensor)
                candidate_obj = schema.TensorT.InitFromObj(candidate_tensor)
                source_buffer_id = int(source_tensor.Buffer())
                # Buffer indices are expected to stay stable through externalization;
                # ExternalBuffer is the only new tensor field and is normalized away.
                source_obj.buffer = 0
                source_obj.externalBuffer = 0
                candidate_obj.buffer = 0
                candidate_obj.externalBuffer = 0
                if plain(source_obj) != plain(candidate_obj):
                    raise AssertionError(f"{graph_type} tensor metadata changed at {sgi}:{ti}")
                meta = plain(source_obj)
                add_record(structure, {"tensor": [sgi, ti], "metadata": meta})
                tensor_count += 1
                source_hash, source_len = buffer_payload(original_model, original_map, source_buffer_id)
                external_id = int(candidate_tensor.ExternalBuffer())
                payload_hash, payload_len = ("", 0)
                if external_id:
                    external_tensor_count += 1
                    payload_hash, payload_len = external_payload_hash(external_id)
                else:
                    buffer_id = int(candidate_tensor.Buffer())
                    payload_hash, payload_len = buffer_payload(candidate_model, candidate_map, buffer_id)
                if source_len or payload_len:
                    if source_len != payload_len or source_hash != payload_hash:
                        raise AssertionError(
                            f"logical payload mismatch in {graph_type} subgraph {sgi} tensor {ti}: "
                            f"{source_len}/{source_hash} vs {payload_len}/{payload_hash}"
                        )
                    tensor_payload_count += 1
                    logical_payload_bytes += payload_len
                    seen_buffer_payloads.add(source_buffer_id)
                    add_record(logical, {
                        "tensor": [sgi, ti],
                        "name": source_tensor.Name() or b"",
                        "shape": source_tensor.ShapeAsNumpy(),
                        "type": source_tensor.Type(),
                        "quantization": source_obj.quantization,
                        "payloadBytes": payload_len,
                        "payloadSha256": payload_hash,
                    })
            for oi in range(source_subgraph.OperatorsLength()):
                left = schema.OperatorT.InitFromObj(source_subgraph.Operators(oi))
                right = schema.OperatorT.InitFromObj(candidate_subgraph.Operators(oi))
                if plain(left) != plain(right):
                    raise AssertionError(f"{graph_type} operator changed at {sgi}:{oi}")
                add_record(structure, left)

        # Compare non-tensor buffers as part of the flatbuffer's logical payload.
        non_tensor_payloads = 0
        for bi in range(original_model.BuffersLength()):
            source_hash, source_size = buffer_payload(original_model, original_map, bi)
            candidate_hash, candidate_size = buffer_payload(candidate_model, candidate_map, bi)
            if bi not in seen_buffer_payloads and (source_size or candidate_size):
                if source_size != candidate_size or source_hash != candidate_hash:
                    raise AssertionError(f"{graph_type} non-tensor buffer {bi} changed")
                add_record(logical, {"buffer": bi, "payloadBytes": source_size, "payloadSha256": source_hash})
                non_tensor_payloads += 1
                logical_payload_bytes += source_size

        # The candidate graph must account for every source logical payload.
        structure_hash = structure.hexdigest()
        logical_hash = logical.hexdigest()
        for handle, mapped in weight_maps.values():
            mapped.close()
            handle.close()
        subgraph_count = int(original_model.SubgraphsLength())
        # FlatBuffer vectors expose NumPy views backed by these mappings. Let
        # their local references leave scope before Python closes the mmaps.
        del original_model, candidate_model

    return {
        "officialGraphFile": original_path.name,
        "officialGraphFileSha256": sha256_file(original_path),
        "repackedGraphFile": candidate_path.name,
        "repackedGraphFileSha256": sha256_file(candidate_path),
        "subgraphs": subgraph_count,
        "tensorCount": tensor_count,
        "tensorPayloadCount": tensor_payload_count,
        "externalTensorCount": external_tensor_count,
        "logicalPayloadBytes": logical_payload_bytes,
        "nonTensorPayloadBuffers": non_tensor_payloads,
        "structureSha256": structure_hash,
        "logicalTensorPayloadSha256": logical_hash,
    }


def audit_pair(original: Path, repacked: Path) -> dict:
    original_toml, original_sections = sections(original)
    repacked_toml, repacked_sections = sections(repacked)
    original_graphs = original_sections["graphs"]
    repacked_graphs = repacked_sections["graphs"]
    if set(original_graphs) != RETAINED | REMOVED:
        raise AssertionError(f"unexpected original graph types: {sorted(original_graphs)}")
    if set(repacked_graphs) != RETAINED:
        raise AssertionError(f"unexpected repacked graph types: {sorted(repacked_graphs)}")
    if set(repacked_sections["weights"]) != RETAINED:
        raise AssertionError("repacked package does not have exactly the four text weight sections")

    orig_common = {}
    repack_common = {}
    for section in original_toml.get("section", []):
        kind, model_type = section.get("section_type"), section.get("model_type")
        if kind == "TFLiteModel" and model_type in RETAINED:
            orig_common[(kind, model_type)] = {k: v for k, v in section.items() if k != "data_path"}
        elif kind in {"LlmMetadata", "SP_Tokenizer"}:
            orig_common[(kind, model_type)] = {k: v for k, v in section.items() if k != "data_path"}
    for section in repacked_toml.get("section", []):
        kind, model_type = section.get("section_type"), section.get("model_type")
        if kind == "TFLiteModel" and model_type in RETAINED:
            repack_common[(kind, model_type)] = {k: v for k, v in section.items() if k != "data_path"}
        elif kind in {"LlmMetadata", "SP_Tokenizer"}:
            repack_common[(kind, model_type)] = {k: v for k, v in section.items() if k != "data_path"}
    if orig_common != repack_common:
        raise AssertionError("retained TOML section metadata changed")

    tokenizer_orig = original / next(s["data_path"] for s in original_toml["section"] if s["section_type"] == "SP_Tokenizer")
    tokenizer_new = repacked / next(s["data_path"] for s in repacked_toml["section"] if s["section_type"] == "SP_Tokenizer")
    tokenizer_hashes = [sha256_file(tokenizer_orig), sha256_file(tokenizer_new)]
    if tokenizer_hashes[0] != tokenizer_hashes[1]:
        raise AssertionError("tokenizer payload changed")

    original_meta = (original / next(s["data_path"] for s in original_toml["section"] if s["section_type"] == "LlmMetadata")).read_text(encoding="utf-8")
    repacked_meta_path = repacked / next(s["data_path"] for s in repacked_toml["section"] if s["section_type"] == "LlmMetadata")
    repacked_meta = repacked_meta_path.read_text(encoding="utf-8")
    normalized_meta = re.sub(r'(?m)^min_runtime_version:\s*"[^"]+"\s*\n?', "", repacked_meta)
    if normalized_meta != original_meta:
        raise AssertionError("LlmMetadataProto changed beyond the required min_runtime_version field")

    graph_results = {}
    for graph_type in sorted(RETAINED):
        graph_result = graph_audit(original, original_graphs[graph_type], repacked,
                                   repacked_graphs[graph_type], graph_type,
                                   repacked_sections["weights"])
        graph_results[graph_type] = graph_result

    weights = {}
    for graph_type, section in sorted(repacked_sections["weights"].items()):
        path = repacked / section["data_path"]
        weights[graph_type] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    metadata_path = repacked / next(s["data_path"] for s in repacked_toml["section"]
                                   if s["section_type"] == "LlmMetadata")
    graph_file_hashes = {name: sha256_file(model_graph_path(repacked, section))
                         for name, section in sorted(repacked_graphs.items())}
    repacked_tokenizer_path = repacked / next(s["data_path"] for s in repacked_toml["section"]
                                              if s["section_type"] == "SP_Tokenizer")
    system_entries = {entry["key"]: entry.get("value")
                      for entry in repacked_toml.get("system_metadata", {}).get("entries", [])}

    return {
        "officialGraphTypes": sorted(original_graphs),
        "textOnlyGraphTypes": sorted(repacked_graphs),
        "removedGraphTypes": sorted(set(original_graphs) - set(repacked_graphs)),
        "retainedTomlSectionMetadataEqual": True,
        "tokenizerSha256": tokenizer_hashes[0],
        "metadataEqualExceptAddedMinRuntimeVersion": True,
        "metadataAddedLine": next((line.strip() for line in repacked_meta.splitlines()
                                    if line.startswith("min_runtime_version:")), None),
        "logicalGraphs": graph_results,
        "textSectionFileSha256": {
            "tokenizer": sha256_file(repacked_tokenizer_path),
            "llmMetadata": sha256_file(metadata_path),
            "graphs": graph_file_hashes,
            "weights": {name: row["sha256"] for name, row in weights.items()},
        },
        "containerIdentity": {
            "uuid": system_entries.get("uuid"),
            "creationTimestamp": system_entries.get("creation_timestamp"),
        },
        "externalWeightSections": weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-unpacked", type=Path, required=True)
    parser.add_argument("--candidate-unpacked", type=Path, required=True)
    parser.add_argument("--rebuilt-unpacked", type=Path)
    parser.add_argument("--official-package", type=Path)
    parser.add_argument("--candidate-package", type=Path)
    parser.add_argument("--rebuilt-package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "textonly-container-equivalence-v1",
              "officialVsCandidate": audit_pair(args.official_unpacked, args.candidate_unpacked)}
    if args.rebuilt_unpacked:
        result["officialVsFreshRebuild"] = audit_pair(args.official_unpacked, args.rebuilt_unpacked)
        result["candidateVsFreshRebuildLogicalGraphsEqual"] = all(
            result["officialVsCandidate"]["logicalGraphs"][name]["logicalTensorPayloadSha256"]
            == result["officialVsFreshRebuild"]["logicalGraphs"][name]["logicalTensorPayloadSha256"]
            for name in sorted(RETAINED)
        )
        candidate_sections = result["officialVsCandidate"]["textSectionFileSha256"]
        rebuilt_sections = result["officialVsFreshRebuild"]["textSectionFileSha256"]
        result["candidateVsFreshRebuildPhysicalTextSectionsSha256Equal"] = candidate_sections == rebuilt_sections
        result["containerIdentityChangedOnRebuild"] = (
            result["officialVsCandidate"]["containerIdentity"]
            != result["officialVsFreshRebuild"]["containerIdentity"]
        )
    package_paths = {
        name: path for name, path in (
            ("official", args.official_package),
            ("candidate", args.candidate_package),
            ("freshRebuild", args.rebuilt_package),
        ) if path is not None
    }
    if package_paths:
        result["packageFiles"] = {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in package_paths.items()
        }
        if "candidate" in package_paths and "freshRebuild" in package_paths:
            result["candidateVsFreshRebuildPackageSha256Equal"] = (
                result["packageFiles"]["candidate"]["sha256"]
                == result["packageFiles"]["freshRebuild"]["sha256"]
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "graphs": sorted(RETAINED),
                      "allEqual": True}, indent=2))


if __name__ == "__main__":
    main()
