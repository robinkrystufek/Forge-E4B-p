# Stock QAT text-only package results

Artifact: `Forge-E4B-Stock-QAT-TextOnly.litertlm` (3,150,102,528 bytes; SHA-256 `8a9f67c07c43a9fb16945dbae90a92e9a35e5b3bfef50686ddc42a3c8094b759`). It remains outside Git at `D:\Forge-GM\experiment\output\stock-qat-textonly-20261126\`.

Source pins: Google mobile-QAT `8e3ac78f1a4a97c19f5f194ab362a2206f71d961`, source safetensors SHA-256 `391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4`; official LiteRT-LM `f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`, package SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.

The text-only package physically removes the audio and vision model sections and linked weights, retains the shared text embedders, decoder, tokenizer and MTP drafter. Source-to-graph audit verifies 345 unique retained quantized source tensors exactly: INT2 3, INT4 258, INT8 84. The 36 additional INT4 K/V tensors in shared-KV layers 24–41 do not exist as separate weights in the official exported graph. The 613 source tensors versus 654 provenance rows are reconciled by the single per-layer embedding tensor represented as 42 graph slabs.

| Device | MTP off median | MTP on median | Decode gain |
| --- | ---: | ---: | ---: |
| RTX 4090, official | 110.26 tok/s | 153.27 tok/s | 39.0% |
| RTX 4090, text-only | 111.61 tok/s | 144.32 tok/s | 29.3% |
| Pixel 10 Pro, official | 7.763 tok/s | 13.345 tok/s | 71.9% |
| Pixel 10 Pro, text-only | 7.641 tok/s | 13.257 tok/s | 73.5% |

Host tests used the shared LiteRT-LM 0.19.2 runtime and 3 repeated narrator generations per mode (4096 context, 453-token prompt, 500-token ceiling). Pixel tests used probe app v113 / LiteRT-LM 0.19.1 / LiteRT 2.3.2, 441-token outputs, dynamic 4096 maximum context and 1024 requested context. Full filtered measurements are in the adjacent JSON files; per-tensor source proof and packed round-trip details are in `source_tensor_audit.json` and `source_weight_round_trip.json`.

Reproduction uses the installed Python 3.11.9 environment at `D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv`. Run `python -m unittest discover -s repro/tests -p test_text_only_repack.py -v`. Repack the pinned official unpacked tree by passing its original prefill/decode graph to `repro/scripts/repack_official_litertlm.py` with `--text-only`; write outputs into a new task output directory. Then run `repro/scripts/audit_stock_qat_source.py` with `--source-dir`, three `--official-graph NAME=PATH` arguments for the decoder/embedder/per-layer embedder, and the provenance fixture. The invocation used for this result is recorded in `docs/TEXT_ONLY_PACKAGING_20261126.md`.

## Final container and reproduction audit

`container_equivalence.json` compares the final text-only package directly with the official package. It verifies the retained TOML section metadata, tokenizer, text metadata (apart from the added minimum runtime version), all four retained logical graphs, every tensor's structure and quantization parameters, and every logical tensor payload. The comparison follows external-buffer references and compensates for payload externalization and deduplication. All retained payload fingerprints match, including the MTP drafter; six audio/vision graphs are the only removed graph types.

A clean rebuild from the pinned official unpacked inputs is recorded in `fresh_rebuild_manifest.json`. The rebuilt package has SHA-256 `cf8398a857d7c22fa9e120c3fdac4e745171bf6c90eaeefc5c2a3f4946c52858` versus candidate `8a9f67c07c43a9fb16945dbae90a92e9a35e5b3bfef50686ddc42a3c8094b759`. The four logical graphs and all physical text sections are equivalent; full container hashes differ because the builder regenerates system metadata identity/timestamp.

The prior short-context MTP issue was reproduced with the official and text-only packages using LiteRT-LM 0.19.2, CPU backend, greedy sampling (`top_k=1`, `temperature=0`), and prompt `Hi`. At KV-cache capacities 32 and 128, both packages fail engine creation at `DYNAMIC_UPDATE_SLICE`; both initialize and generate at 1024. See `short_context_mtp.json`. This establishes a runtime/cache-capacity failure shared by the official package.

## Remaining validation

The user clarified deterministic decoding means greedy. The requested paired greedy run over the frozen 12-prompt panel, including generated-token-ID divergence and logits where accessible, has not yet been completed. The existing Python API exposes decoded text and token counts; its token-trace debugger is disabled in the installed runtime build, so direct generated token IDs/logits are not currently exposed through that API.

The prior Pixel tests were GPU runs with `maxContext=4096` but `requestedContext=1024`. The app diagnostics confirm this for both package hashes and both MTP states. The requested sustained Pixel run at an actual 4096 requested context, with MTP on/off and sampled peak memory for both packages, remains outstanding. These prior runs do not satisfy that requirement, and peak device memory was not sampled. The model binary remains local and is not included in Git.
