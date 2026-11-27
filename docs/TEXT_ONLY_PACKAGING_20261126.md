# Stock mobile-QAT Gemma 4 E4B text-only LiteRT-LM

Date: 2026-11-26 (America/Los_Angeles)

## Status

The requested text-only artifact is built and validated on host and Pixel. Tensor-level auditing proves the official text graphs already contain the pinned original stock-QAT weights for every weight used by those graphs, so no patched copy is needed: comparison A=C and B=D. The artifact is comparison D. Its binary remains in the local experiment output because the repository has no established model-artifact publication mechanism documented or configured. Peak host/device memory and broad logit/quality-suite measurements were not captured; these are reported as limitations below.

## Inputs and method

- Official package: `litert-community/gemma-4-E4B-it-litert-lm@f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`, pinned SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.
- QAT source revision: `google/gemma-4-E4B-it-qat-mobile-transformers@8e3ac78f1a4a97c19f5f194ab362a2206f71d961`, pinned source SHA-256 `391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4`.
- Packaging used the pinned official bundle's pristine text `model.toml`, graphs and weights; no weight patch or training was applied. Source-to-graph tensor auditing below proves all retained source-QAT text weights are exact, so the official bundle is also the stock-QAT package (A=C). The text-only repack is both B and D.
- `repro/scripts/repack_official_litertlm.py --text-only` removes the complete six section records for audio encoder/adapter/end marker and vision encoder/adapter/end marker. The official builder then omits their associated external weight sections. It retains shared text embedders, tokenizer, prefill/decode graph and MTP drafter.

## Artifact and structural result

- Artifact: `D:\Forge-GM\experiment\output\stock-qat-textonly-20261126\Forge-E4B-Stock-QAT-TextOnly.litertlm`
- Size: 3,150,102,528 bytes; SHA-256 `8a9f67c07c43a9fb16945dbae90a92e9a35e5b3bfef50686ddc42a3c8094b759`.
- Official source package: 3,659,530,240 bytes. Reduction: 509,427,712 bytes (13.92%). Removed original audio section payloads total 109,795,848 bytes (3.00% of the original package); vision sections total 232,037,460 bytes (6.34%). The builder also removed 167,772,160 bytes of duplicated shared text-embedding data while externalizing weights. Container/header overhead means these values do not add exactly to the observed package delta.
- 10 sections remain: metadata, tokenizer, 4 TFLite graphs (two shared text embedders, decoder, MTP drafter), and their 4 external weight sections. All audio/vision model types and their external weight sections are absent.
- Text generation signatures and `mtp_drafter` remain present. Tokenizer section is preserved.

## Quantized source-to-graph parity

- The pinned source safetensors SHA-256 is `391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4`. The existing round-trip audit reconstructed all 613 source packed quantized tensors exactly: 111 INT2, 306 INT4 and 196 INT8, with zero packed bytes changed and zero reconstruction error.
- The apparent 613-versus-654 provenance count is reconciled: provenance expands the single per-layer embedding tensor into 42 graph slabs, adding 41 records. Those slabs were separately audited against their slices in the pinned source tensor.
- New `repro/scripts/audit_stock_qat_source.py` verifies source integer payload hashes, per-channel scales, TFLite dtype/shape/axis, and graph payload hashes across the official decoder, shared embedder and per-layer embedder. It verified 388 graph records representing 345 unique retained source tensors: INT2 3, INT4 258, INT8 84. This covers all 345 text-QAT source tensors represented by the official graph. The 36 additional source INT4 K/V tensors in layers 24–41 are intentionally absent from the official graph; the source config declares 18 shared-KV layers, and those layers have no separate K/V weight payloads in the exported graph.
- Therefore the official text graphs are exact at every retained stock-QAT tensor; A=C, and removal of only audio/vision sections gives B=D. The original official MTP drafter section is retained and its required weights are externalized by the official builder; MTP was separately exercised at runtime.
- Full per-tensor evidence is retained in `repro/results/stock_qat_textonly_20261126/source_tensor_audit.json`; the source packed-round-trip audit and package manifest are in the same result bundle.

## Reproduction commands

The run reused Python 3.11.9 / LiteRT-LM 0.19.2 from `D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv`. Inputs below are the pinned, already-unpacked official and source checkpoints retained in the existing disposable reproduction tree; output paths must be new and empty.

```powershell
$py = 'D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv\Scripts\python.exe'
$stock = 'D:\Forge-GM\experiment\output\DISPOSABLE_FORGE_E4B_P_REPRO_20261124\models\stock-e4b\unpacked'
$out = 'D:\Forge-GM\experiment\output\stock-qat-textonly-rebuild'
& $py repro\scripts\repack_official_litertlm.py `
  --unpacked-dir $stock `
  --patched-graph (Join-Path $stock 'Section10_TFLiteModel_tf_lite_prefill_decode.tflite') `
  --output (Join-Path $out 'Forge-E4B-Stock-QAT-TextOnly.litertlm') `
  --manifest (Join-Path $out 'package_manifest.json') `
  --text-only
```

Then run the source audit using the matching `models\qat-e4b` source directory and official `Section10`, `Section2`, and `Section3` graph paths with `--official-graph official_main=PATH`, `--official-graph official_embedder=PATH`, and `--official-graph official_per_layer_embedder=PATH`. The exact source audit JSON is included in the results bundle. Unpack the official source package with the pinned LiteRT-LM CLI before running these commands when reproducing from scratch.

## Host inference checks

Reused Python 3.11.9 / LiteRT-LM 0.19.2 at `D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv`. Known-good official package was run before each candidate mode with identical prompt and flags.

| Mode | Official baseline | Text-only candidate |
|---|---|---|
| CPU, greedy smoke | `42`, exit 0 | `42`, exit 0 |
| RTX 4090 GPU, greedy smoke | `42`, exit 0; Direct3D 12 adapter | `42`, exit 0; Direct3D 12 adapter |
| CPU, MTP enabled, 1024 max tokens | `42`; draft success 1.0 | `42`; draft success 1.0 |
| RTX 4090 GPU, MTP enabled, 1024 max tokens | `42`; draft success 0.666667 | `42`; draft success 0.666667 |

GPU resource gate recorded 23,394 MiB free, 0–1% utilization, and no compute PIDs before the RTX 4090 benchmark phases; the 20 GiB estimate plus 2 GiB reserve fit. A 128-token MTP smoke failed with `DYNAMIC_UPDATE_SLICE` preparation error; the established 1024-token MTP smoke completed. Draft acceptance was 0.666667 on GPU and 1.0 on CPU for both official and candidate packages.

RTX 4090 paired long-generation test: fixture `benchmark.narrator.v1`, 4096 context, 453 input tokens, 500 output ceiling, 3 runs per mode; GPU resource gate passed exclusively at 23,394 MiB free, 0–1% utilization, no compute PIDs. Median decode/prefill rates were official MTP off 110.26/4078 tok/s, official MTP on 153.27/4098 tok/s, candidate MTP off 111.61/4092 tok/s, candidate MTP on 144.32/4097 tok/s. MTP gains were 39.0% official and 29.3% text-only. Median TTFT was 0.120 s off and 0.117 s on. Each mode generated 472 tokens (off) or 405 tokens (on); all runs passed the 200-token floor.

Pixel test used the fixed app v1 narrator prompt (458 input tokens), 4096 maximum context with dynamic context and 1024 requested context, 500 output ceiling, Pixel 10 Pro / Android 17 / GPU / model-default precision, probe app v113 / LiteRT-LM 0.19.1 / LiteRT 2.3.2. Candidate MTP on: 5 runs, each 441 tokens, median 13.257 tok/s, 2.412 s TTFT, 35.615 s total. Candidate MTP off: 4 runs, each 441 tokens, median 7.641 tok/s, 2.432 s TTFT, 60.032 s total (73.5% decode gain). Official package MTP off/on: 3 runs per mode, 441 tokens each; median 7.763/13.345 tok/s, median TTFT 2.847/2.313 s, total 59.536/35.869 s (71.9% decode gain). The official model load medians were 31.274 s off and 23.116 s on; candidate medians were 25.023 s off and 24.189 s on. The official and candidate package hashes matched the host. MTP requested/effective state was recorded in diagnostics. Filtered metrics only (no prompt or response text) are retained in `repro/results/stock_qat_textonly_20261126/`.

Full host logs, resource gate records, repeated benchmark results and filtered Pixel metrics are retained beside the package. The first MTP failure log is retained for reproducibility.

## Tests and next acceptance work

`python -m unittest discover -s repro/tests -p test_text_only_repack.py -v` passed 2 tests: exact multimodal section removal with MTP retention, and fail-closed behavior for an incomplete official schema. `audit_stock_qat_source.py` completed against all three official text graphs and matched every retained tensor as described above.

Remaining publication and measurement notes:

1. Peak host VRAM and Pixel process memory were not sampled during the performance runs; only pre-phase host free VRAM (23,394 MiB) and app model-load times are available.
2. Broad logits/quality suites were not run. The unchanged retained graph bytes, exact source quantized values/scales, and deterministic `42` smokes characterize package integrity; they are not a large behavioral evaluation.
3. The local project docs/repository contain no configured or established model-binary publishing destination. The package is retained locally with its SHA; the binary was not uploaded. The code, tests and compact audit/benchmark evidence can be submitted in a dedicated PR.
