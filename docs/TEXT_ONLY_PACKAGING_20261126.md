# Stock mobile-QAT Gemma 4 E4B text-only LiteRT-LM

Date: 2026-11-26 (America/Los_Angeles)

## Status

The requested text-only artifact is built and validated on host and Pixel. Tensor-level auditing proves the official text graphs already contain the pinned original stock-QAT weights for every weight used by those graphs, so no patched copy is needed: comparison A=C and B=D. The artifact is comparison D. Its binary remains in the local experiment output because the repository has no established model-artifact publication mechanism documented or configured. Additional final-container and fresh-rebuild audits are included in the results bundle. The remaining validations and limitations are listed below.

## Inputs and method

- Official package: `litert-community/gemma-4-E4B-it-litert-lm@f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`, pinned SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.
- QAT source revision: `google/gemma-4-E4B-it-qat-mobile-transformers@8e3ac78f1a4a97c19f5f194ab362a2206f71d961`, pinned source SHA-256 `391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4`.
- Packaging used the pinned official bundle's pristine text `model.toml`, graphs and weights; no weight patch or training was applied. Source-to-graph tensor auditing below proves all retained source-QAT text weights are exact, so the official bundle is also the stock-QAT package (A=C). The text-only repack is both B and D.
- `repro/scripts/repack_official_litertlm.py --text-only` removes the complete six section records for audio encoder/adapter/end marker and vision encoder/adapter/end marker. The official builder then omits their associated external weight sections. It retains shared text embedders, tokenizer, prefill/decode graph and MTP drafter.

## Artifact and structural result

- Artifact: `Forge-E4B-Stock-QAT-TextOnly.litertlm` (local, not distributed)
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

Run from a fresh repository checkout on Windows with Python 3.11 x64. No training environment, GPU training, or adapter is needed. The export lock pins the published packages; the validation run reused Python 3.11.9 and LiteRT-LM/builder 0.19.2. Allow disk space for the official download, unpacked inputs, the candidate, builder staging, and optionally a second rebuild. Do not reuse output directories. These portable commands are derived from the verified run; a separate fresh-venv reproduction of this exact command sequence has not been performed.

### Fetch and repack

~~~powershell
$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$work = Join-Path $root 'DISPOSABLE_TEXT_ONLY'
if (Test-Path $work) { throw "Choose a new workspace" }
New-Item -ItemType Directory $work | Out-Null
py -3.11 -m venv "$work\export-venv"
$py = "$work\export-venv\Scripts\python.exe"
$cli = "$work\export-venv\Scripts\litert-lm.exe"
$s = Join-Path $root 'repro\scripts'
& $py -m pip install -r "$root\repro\requirements-export.lock"
& $py "$s\download_stock_litertlm.py" --output "$work\gemma-4-E4B-it.litertlm"
if ($LASTEXITCODE -ne 0) { throw "Official download failed" }
$official = "$work\gemma-4-E4B-it.litertlm"
if ((Get-FileHash $official -Algorithm SHA256).Hash.ToLower() -ne '0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0') { throw "Official hash mismatch" }
$stock = "$work\official-unpacked"
& $cli unpack $official --output-dir $stock
if ($LASTEXITCODE -ne 0) { throw "Unpack failed" }
$out = "$work\candidate"
$candidate = "$out\Forge-E4B-Stock-QAT-TextOnly.litertlm"
& $py "$s\repack_official_litertlm.py" `
  --unpacked-dir $stock `
  --patched-graph "$stock\Section10_TFLiteModel_tf_lite_prefill_decode.tflite" `
  --output $candidate --manifest "$out\package_manifest.json" --text-only
if ($LASTEXITCODE -ne 0) { throw "Repack failed" }
& $py -m unittest discover -s repro/tests -p test_text_only_repack.py -v
~~~

The argument named --patched-graph is a historical interface: pass the original official prefill/decode graph unchanged. Expected package size in the tested builder is 3,150,102,528 bytes, ten sections, with four retained text/MTP graphs. Full package SHA changes on rebuild because of regenerated UUID/timestamp.

### Verify source and final container

The optional source-to-official audit requires the pinned mobile-QAT checkpoint; it is not required to build the text-only package.

~~~powershell
$qat = "$work\qat-e4b"
& $py "$s\download_checkpoint.py" --output "$qat\model.safetensors" --metadata "$work\source-download.json"
& $py "$s\download_checkpoint_metadata.py" --output-dir $qat
if ((Get-FileHash "$qat\model.safetensors" -Algorithm SHA256).Hash.ToLower() -ne '391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4') { throw "QAT source hash mismatch" }
& $py "$s\audit_stock_qat_source.py" --source-dir $qat `
  --official-graph "official_main=$stock\Section10_TFLiteModel_tf_lite_prefill_decode.tflite" `
  --official-graph "official_embedder=$stock\Section2_TFLiteModel_tf_lite_embedder.tflite" `
  --official-graph "official_per_layer_embedder=$stock\Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite" `
  --provenance "$root\repro\fixtures\quantization_provenance.json" --output "$work\source_tensor_audit.json"
& $cli unpack $candidate --output-dir "$work\candidate-unpacked"
& $py "$s\audit_textonly_container_equivalence.py" `
  --official-unpacked $stock --candidate-unpacked "$work\candidate-unpacked" `
  --official-package $official --candidate-package $candidate `
  --output "$work\container_equivalence.json"
~~~

For rebuild verification, repeat the repack into a new output directory, unpack it, and add --rebuilt-unpacked and --rebuilt-package to the container audit command. Compare logical payload fingerprints and physical text-section hashes to the [recorded audit](../repro/results/stock_qat_textonly_20261126/container_equivalence.json), rather than requiring a bit-identical outer container.

### Inference and evidence

Use the repository's public narrator benchmark fixture and [benchmark procedure](MOBILE_QAT_LORA_TO_MIXED_BIT_LITERT_LM.md#functional-check-and-performance-acceptance) for host throughput, substituting each package in turn. The CPU 12-prompt comparison script requires the original frozen panel JSON (cases with case_id, serialized_prompt, prompt_sha256), the unpacked SentencePiece tokenizer, and MTP off. The original panel text is not tracked in this PR: its hashes and comparison results are available, but exact independent replay requires the original fixture. Pixel results were obtained with isolated probe v113/v116; the probe build and full automation procedure are not included in this repository. See the [results index](../repro/results/stock_qat_textonly_20261126/RESULTS.md) for the available evidence.

## Host inference checks

Reused Python 3.11.9 / LiteRT-LM 0.19.2. Known-good official package was run before each candidate mode with identical prompt and flags.

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

## Completed acceptance checks and limitations

`python -m unittest discover -s repro/tests -p test_text_only_repack.py -v` passed 2 tests: exact multimodal section removal with MTP retention, and fail-closed behavior for an incomplete official schema. `audit_stock_qat_source.py` completed against all three official text graphs and matched every retained tensor as described above.

Completed checks:

1. Pixel 10 Pro strict 4096-context GPU validation is complete for both packages, MTP off/on. The isolated v116 probe reported `requestedContext=4096` and `maxContext=4096` for every generation. Each used the same 3499-token prompt and generated 500 tokens (502 with MTP), with greedy sampling (`top_k=1`). Exact per-run settings, package hashes, timings, and rates are in `repro/results/stock_qat_textonly_20261126/pixel_memory_4096_matched.json`.
2. The frozen 12-prompt panel was run paired with greedy sampling on CPU, MTP off, 4096-token KV capacity, and a 128-token output ceiling. Decoded output matched byte-for-byte in all cases; re-tokenized SentencePiece ID sequences also matched, with no divergence. `greedy_panel.json` contains hashes and per-case metrics, without prompt or response text. These re-tokenized IDs are not the native sampled ID stream. LiteRT-LM 0.19.2's Python API returns generated text and token counts; native token traces and logits are unavailable because this runtime build has its debugger disabled. The 128-token ceiling limits the comparison to response prefixes.
3. Matched full-window memory sampling at requested context 4096 is complete. Observed PSS / GL mtrack peaks were official MTP-off 4074.8 / 3128.4 MiB, official MTP-on 4172.1 / 3187.1 MiB, text-only MTP-off 4146.6 / 3130.8 MiB, and text-only MTP-on 5049.5 / 3716.0 MiB. Raw traces and exact KiB values are in `pixel_memory_4096_*.csv` and `pixel_memory_4096_matched.json`. These are sampled process and graphics-accounting maxima, not allocator-level absolute peaks; do not sum PSS and GL mtrack. The earlier partial/uneven samples remain in the result bundle as historical records.
4. A matched short-context MTP check on the official and text-only packages reproduced `DYNAMIC_UPDATE_SLICE` preparation failures at KV-cache capacities 32 and 128, with both completing at 1024. This points to the shared runtime/cache path. See `repro/results/stock_qat_textonly_20261126/short_context_mtp.json`.
5. The final-container and clean-rebuild comparison is in `repro/results/stock_qat_textonly_20261126/container_equivalence.json`; the rebuilt package manifest is `repro/results/stock_qat_textonly_20261126/fresh_rebuild_manifest.json`. Logical tensor payloads and physical text sections match. Full container hashes differ because the builder regenerates system metadata identity/timestamp.
### Remaining limitations

- All four retained logical graphs and tensor payloads match, but physical graph bytes differ from the original because of externalization/deduplication. The builder adds `min_runtime_version: "0.19.0"`. The fresh rebuild matches physical text-section hashes, not the outer UUID/timestamp-bearing container hash.
- Native token IDs, logits, GPU/MTP-on paired greedy output parity, broad quality evaluations and peak host VRAM were not measured.
- The original requested-context-4096 text-only MTP-on run exhibited increased PSS/native heap and reduced throughput, but neither substantial difference recurred in a counterbalanced repeat with clean app processes and cold compiled caches (two runs per package). Mean peak PSS differed by +0.65%, GL mtrack by 2 KiB, and decode throughput by −1.35%. The original run alone had a compiled cache present, shared a process with preceding runs, and coincided with a system-wide background-process freeze/unfreeze wave; no causal attribution is established. Warm-cache/long-lived-process behavior remains uncharacterized by the repeat. See the [investigation](../repro/results/stock_qat_textonly_20261126/pixel_memory_anomaly_investigation.md).
- The original frozen 12-prompt fixture and Pixel probe build/automation are not included. The package binary remains local; no configured publication destination was found.
