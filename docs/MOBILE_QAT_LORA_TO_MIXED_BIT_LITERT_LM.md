# Gemma 4 E4B mobile QAT → LoRA → mixed-bit LiteRT-LM with MTP

This is a reproducible implementation record for adapting the pinned mobile-QAT checkpoint with LoRA, preserving its trained mixed INT2/4/8 representation, and packaging it in the official LiteRT-LM bundle while retaining the MTP drafter. It separates a lightweight correctness preflight from the resource-intensive full experiment.

## Reproduction status and scope

**Validated from a clean checkout on November 24, 2026.** The [clean POC reproduction record](../repro/results/POC_REPRODUCTION_20261124.md) documents a fresh clone, fresh pinned training and export environments, downloads of the pinned public model inputs, bit-exact round-trip of all 613 packed source weights, one synthetic optimizer update, fixed-scale merge, audited graph patch, repack with MTP, and CPU/GPU inference. The resulting package was also tested on a Pixel 10 Pro.

The implementation scripts, synthetic fixtures, provenance mapping and dependency locks are tracked in this repository. No private corpus, pre-existing workstation cache, separate repository, particular machine hostname or internal orchestration service is required to reproduce the clean POC. External model downloads, compatible hardware, Windows tooling and access to the pinned dependencies are required.

**Start with the [canonical clean-checkout procedure](#clean-proof-of-concept-reproduction-repository-tracked-inputs).** Sections 1–6 explain the method and historical experiments; they are not a second runbook. The historical R14-D/P10 results used a different training corpus and are not expected to be reproduced by the one-row public fixture. The POC establishes packaging and inference correctness, not model quality, generalization, or byte-identical output across every rebuild.

Tested environment:
- Source: `google/gemma-4-E4B-it-qat-mobile-transformers`, revision `8e3ac78f1a4a97c19f5f194ab362a2206f71d961`.
- Training: Windows, Python 3.11, RTX 4090 with CUDA, pinned PyTorch and Transformers.
- Export and PC inference: separate LiteRT-LM CLI/API and builder 0.19.2 environment; GPU inference uses WebGPU/Dawn over Direct3D 12, not CUDA.
- Android validation: Pixel 10 Pro, Android 17, Forge GM 1.0-v116 for the long-form benchmark, LiteRT-LM 0.19.1 and LiteRT 2.3.2. The earlier debug-probe build is historical.

## 1. Acquire and validate the exact source

Use the repository downloader and pin the revision. The metadata is part of the model: weights alone do not describe widths, groups, scales or packing.

The canonical procedure below downloads the checkpoint and metadata, inspects its header, and performs the packed-weight round-trip using the tracked scripts. Use an explicit model directory and verify the download hash before training.

Expected packed-weight round-trip inventory: 613 tensors: 111 INT2, 306 INT4 and 196 INT8. The historical provenance fixture separately reports 654 weight tensors (152 INT2, 306 INT4, 196 INT8). These counts differ; the exact scope difference has not been independently reconciled. The 613-tensor round-trip result refers to the packed tensors tested in the clean reproduction. INT2/INT4 use low-value-first offset-binary packing in U8; INT8 is signed. Stop if metadata cannot be reconciled or the no-op round-trip changes packed bytes or decoded values. Never reinterpret these payloads as ordinary BF16 tensors.

## 2. Prepare corpus and splits

Prepare JSONL with one conversation per line. Keep the schema simple, stable, and compatible with the pinned tokenizer and chat template. Synthetic example:

~~~json
{"id":"ex-0001","sourceId":"session-17","lossMask":"last_assistant_only","messages":[{"role":"user","content":"A gate is open and the hall is dark. What should I check first?"},{"role":"assistant","content":"Check the nearest light source and the gate mechanism before moving farther."}]}
~~~

Split by campaign/session/source before shuffling so paraphrases or adjacent turns cannot leak across train and evaluation. Keep examples within the template's context budget; log rejected/oversize rows instead of silently truncating. Pin tokenizer files and chat-template version. Mask user text, system text, role/header tokens and earlier assistant turns to label -100; supervise only the final assistant answer. Preserve the expected end-of-turn behavior. In the audited preparation path token IDs 106 (EOT) and 107 (template newline) were masked from loss; inspect actual token labels because one nearby source comment says newline is omitted although the implementation masks it.

Do not copy private campaign data into a public guide or repo. Store a small synthetic fixture for schema tests and keep real corpora in access-controlled storage. Record source IDs, preprocessing version, hashes, split assignment and deduplication decisions.

## 3. LoRA on the quantized base

Use the original `transformers.integrations.gemma_quant.QuantizedLinear` module semantics. The recorded R14-D target was the 132 original INT4 language-model self-attention projections: 42 q_proj, 24 k_proj, 24 v_proj and 42 o_proj, reflecting shared KV. Freeze packed base bytes and scales. Do not adapt INT2/INT8 tensors, embeddings, or the MTP drafter.

Recorded adapter: rank 16, alpha 32 (scale 2), Kaiming A, zero B. The experiment applies p=0.05 elementwise dropout to the LoRA A factor; this is not standard PEFT input dropout. Its quantized forward adds the LoRA delta to the dequantized base then rounds/clamps at the original width and fixed group scale. Use a weight-rounding STE and exact SRQ forward with identity activation STE for training gradients; plain exact SRQ has no useful gradient and can produce finite loss with zero adapter gradients. Keep original packed representation/scales immutable.

Representative protocol (the full R14-D run used these settings): AdamW lr 1e-4, weight decay .01, effective batch 8, 5% warmup then cosine decay, grad norm clip 1, BF16 autocast, gradient checkpointing, `use_cache=False`, seed 451, one pass over prepared data, max length 4096 without silent truncation. Save optimizer and RNG state to resume exactly. `--max-samples` limits examples but still loads the full model: it is not a cheap preflight.

For a runnable one-update example, see the canonical training command below. Full-corpus experiments require their own corpus, split manifest and training configuration.

Before every GPU phase check available VRAM, utilization, and active processes. Wait/retry until estimated peak plus the agreed 2 GiB reserve fits. Never terminate an existing workload. Keep throughput benchmarks exclusive and separate from functional checks. On the recorded setup the gate estimate was 20 GiB plus 2 GiB reserve.

## 4. Merge without changing quantization

Merge the trained floating adapter delta into the source's quantized grid using the original fixed scales and quantizer. Use the training-produced quantization metrics as the expected merge record:

The canonical procedure calls the tracked merge script with the adapter checkpoint actually emitted by training and the corresponding quantization metrics.

Require `training_merge_packed_exact: true`; fail on mismatched scales or quantization metrics. In R14-D, exactly 77 INT4 tensors and 183,610 packed values changed; INT2=0 and INT8=0. Verify decoded integer values, shapes, widths, scale hashes, q-axis/zero-points and op use. This is still the source INT2/4/8 scheme, not a new uniform INT4 or dynamic quantization.

## 5. Patch official LiteRT-LM graphs and package with MTP

Use the exact official LiteRT-LM conversion/builder version and pristine stock E4B bundle/config. Patch only corresponding prefill/decode packed weight payloads after proving unique tensor matches, base integer hashes, readback, and unchanged bytes outside payload ranges. The official builder externalizes weights and can change section layout; compare tensor payload provenance, not raw section offsets. Retain the official MTP drafter section/config unchanged. Do not rebuild/approximate the drafter or omit it.

Run the patcher and builder with the pinned, separate export environment; it contains `ai_edge_litert` and `litert_lm_builder`. The patch audit JSON must report `merge_packed_hash_match: true` and `outside_payload_bytes_unchanged: true` before you repack.

The canonical patch/repack commands use the tracked provenance mapping and an unpacked copy of the pinned official bundle. Do not use a modified or unverified base package.
Audit package contents: all ten graph structures match expected counts; 1,658 externalized stock weight payloads matched byte-for-byte with zero mismatch; official drafter has 23 quantized buffers (13 INT4, 10 INT8), unchanged. P10 operator provenance maps 77 adapted tensors to 242 FULLY_CONNECTED occurrences. These checks catch accidental graph drift and wrong tensor replacement.

Expected audited P10 artifact: 3,492,003,840 bytes, SHA-256 `18108925d98c552107e2e3d0724c025bb839fa8d70cd6ad5e1aa86ea460b0cfd`. Recompute and compare hashes in a reproduction; do not assume every rebuild is byte-deterministic. The package was created by patching official graph payloads and repacking through official LiteRT-LM builder, not by treating a raw safetensors file as a LiteRT-LM package.

## 6. Lightweight checks and inference validation

Before a candidate run, check the pinned CLI version. Use the functional smoke commands in the canonical procedure with both the stock and candidate packages.

Confirm verbose output selects the NVIDIA RTX 4090 adapter with Direct3D 12, generates a response, and exits successfully. Capture the full initialization log and first native error if it fails. Treat any traceback, runtime error, or missing generated response as failure even if the CLI process exit code is zero. The WebGPU/Dawn backend is not CUDA. For Python API engine creation on Windows, add the LiteRT-LM package folder via `os.add_dll_directory(...)` so `dxcompiler.dll` and `dxil.dll` resolve, and keep that directory handle alive until the engine closes.

Smoke-test a known-good stock package first, then candidate with identical flags. `--cache=no` avoids compiled-model sidecars. Keep functional smoke separate from performance measurements; use an exclusive repeated-run benchmark only for performance comparisons.

Android was tested in Forge GM's isolated debug probe app, not Edge Gallery. On Pixel, verify pushed file SHA, select Model-default precision and GPU, and test MTP on/off in separate runs with identical app/runtime/prompt/context when comparing. The 4096 context with initial FP32 compatibility setting was LMK-killed five times. A 1024 requested context, Model-default precision, GPU run completed with MTP off; this only validates that debug configuration and does not establish 4096-context safety.

The supplied Forge-GM v116 export contains repeated P10 GPU narrator measurements. The corresponding app-aligned PC comparison, including per-run TTFT and decode tok/s, is recorded in section 6a. The separate 1024-context MTP-off run above remains a functional probe and should not be combined with the 4096-context performance results.

## 6a. Forge-GM app-aligned P10 narrator benchmark on the Windows RTX 4090 test host

The previous host table in this section used a short CLI microbenchmark (128 synthetic prefill tokens / 64 forced decode tokens). That microbenchmark is valid only for that short fixed-token workload; its MTP acceptance was 0%, and it does not model Forge-GM's long narrator benchmark. The comparison below replaces it for app-performance conclusions.

The Pixel record and host run use the exact same P10 package SHA-256, `18108925d98c552107e2e3d0724c025bb839fa8d70cd6ad5e1aa86ea460b0cfd`. Forge-GM defines the fixed workload as `benchmark.narrator` version 1, the tracked `repro/fixtures/benchmark.narrator.v1.txt` fixture (derived from Forge GM's registered prompt). Its UTF-8 SHA-256 is `de63332fe826bdaa0ca030662f01f150799186b04218aecd8cd966e3a1abbd38`. The app uses a 500-token output ceiling, 4096-token context, thinking disabled, and its narrator sampler defaults (top-k 64, top-p 0.95, temperature 1.0). Forge-GM accepts long-form performance samples only at 200 or more output tokens.

The host uses LiteRT-LM Python conversation API 0.19.2, reusing the installed export environment and P10 engine cache. The Pixel export is from Forge-GM 1.0-v116 with LiteRT-LM 0.19.1 / LiteRT 2.3.2 on Pixel 10 Pro / Android 17. PC prompt-token count was 453 versus Pixel's 458 for the same prompt string; this runtime/template count difference is recorded as observed. Three conversations were measured per mode. Every host output cleared the 200-token validity floor.

| PC backend | MTP | Decode tok/s, runs | Median decode tok/s | TTFT, runs | Median TTFT | Output tokens |
|---|---|---:|---:|---:|---:|---:|
| CPU | Off | 14.44, 14.31, 14.45 | 14.44 | 2.125, 1.962, 1.975 s | 1.975 s | 463, 463, 463 |
| CPU | On | 19.68, 19.88, 19.75 | 19.75 | 2.083, 1.953, 1.960 s | 1.960 s | 382, 382, 382 |
| RTX 4090 / WebGPU-Dawn / D3D12 | Off | 108.61, 111.86, 111.62 | 111.62 | 1.081, 0.120, 0.120 s | 0.120 s | 431, 431, 431 |
| RTX 4090 / WebGPU-Dawn / D3D12 | On | 111.87, 163.75, 163.85 | 163.75 | 1.180, 0.120, 0.120 s | 0.120 s | 368, 368, 368 |

MTP increased median PC decode throughput about 36.8% on CPU and 46.6% on GPU. The first GPU run in each mode includes cold-start effects; the two warm repeats were stable at about 111.7 tok/s without MTP and 163.8 tok/s with MTP. The output lengths differ because generation ended at different points, so decode rate is the relevant comparison; all runs produced valid long-form outputs.

The supplied Pixel v116 export has five P10 GPU narrator records with the same package hash, prompt version, 458 input tokens, model-default precision, context 4096 and output ceiling 500. Its two MTP-off runs were 7.775 and 7.832 tok/s (median 7.804; median TTFT 2.696 s). Its three MTP-on runs were 13.732, 13.579 and 14.198 tok/s (median 13.732; median TTFT 2.899 s), about 1.76x the MTP-off median. This is repeated diagnostic evidence of P10 speedup on Pixel, not a controlled hardware/runtime comparison against the PC.

The earlier 128/64-token synthetic CLI benchmark reported zero accepted drafts and lower MTP-on throughput; that result does not contradict the long-form app workload above. A separate fixed-token 500-decode API attempt did not emit progress for several minutes and was stopped after confirming it was this task's own process. The app-equivalent conversation API completed all reported runs. No partial or stalled-run metrics are included.

## 6b. Pixel 10 Pro GPU MTP benchmark

This section records a separate on-device benchmark of the clean POC artifact. The exact file selected in Forge GM's Load/Replace model picker was `Gemma4-E4B-POC-INT2-4-8-MTP.litertlm` in Downloads. Its 3,492,003,840-byte device file was SHA-256 checked after ADB transfer and matched the host package manifest: `d9088bd7e39056f9a470332d4a6f9fac309e05de7392e181606382091b225b7f`. The file picker selection, rather than a prior benchmark report's model label, identifies the tested artifact.

The test ran on a Pixel 10 Pro with Android 17 using Forge GM `1.0-v116`, LiteRT-LM `0.19.1`, and LiteRT `2.3.2`. Only GPU inference was measured; no Pixel CPU runs were performed. Four generations were run sequentially with MTP off, then four with MTP on. The fixed workload is Forge GM's registered `benchmark.narrator` version 1; its tracked fixture is `repro/fixtures/benchmark.narrator.v1.txt`, SHA-256 `de63332fe826bdaa0ca030662f01f150799186b04218aecd8cd966e3a1abbd38`. Each mode used the same 4096-token context, 500-token output ceiling, thinking disabled, GPU backend, and model-default precision. The app reported four valid runs, no errors, no short runs, and no missing metrics per mode; all outputs exceeded the 200-token validity floor.

| Pixel GPU mode | TTFT mean (range) | Decode tok/s mean (range) | Total generation time mean (range) | Input tokens | Output tokens |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MTP off | 2.85 s (2.52–3.03) | 7.7 (7.5–7.8) | 59.88 s (59.08–61.49) | 458 | 440 |
| MTP on | 2.77 s (2.42–2.98) | 12.8 (12.7–13.0) | 37.09 s (36.26–37.74) | 458 | 441 |

The app's per-run/profile table reported `MTP requested / effective` as `Off / Off` for the baseline and `On / On` for the drafter run. Mean decode throughput rose 66.2% (12.8 versus 7.7 tok/s); mean TTFT changed from 2.85 s to 2.77 s, and mean generation time fell 38.1%. This is a within-device A/B result for the selected package and fixed prompt, not a comparison against PC hardware.

### Forge GM engine path and metric boundaries

The benchmark uses Forge GM's compiled prompt and native LiteRT-LM conversation path rather than a separate Android inference client:

| Stage | Forge GM implementation | Behavior relevant to this run |
| :--- | :--- | :--- |
| Prompt | `src/main.ts` calls `compilePrompt('benchmark.narrator', {})`; `src/prompts/templates.ts` owns the versioned fixture | Fixed prompt v1; no app-specific duplicate prompt and no retry |
| Test request | `android/app/src/main/java/app/forgegm/local/ForgeBridge.kt`, `probeModel` | Empty system/tools/history; one conversation; 500-token limit; thinking disabled; supplies no sampler override, so LiteRT-LM uses the bundle/runtime default sampler |
| Model inspection and MTP | `ForgeBridge.openEngine` and `android/gpu-runtime/src/main/kotlin/app/forgegm/generation/LiteRtGeneration.kt` | `ModelInfo` detects the bundle's speculative-decoding support; requested MTP is configured ON only when a drafter is present and backend is GPU or CPU. The run's app profile reported requested and effective MTP both ON |
| GPU engine | `LiteRtGeneration` | Uses LiteRT-LM `Backend.GPU` and `EngineConfig`; model-default precision leaves activation data type unset. Context is capped at `min(4096, bundle maximum)` for this narrator path |
| Benchmark instrumentation | `ForgeBridge.probeModel` and `ModelDiagnostics.benchmark` | Enables LiteRT-LM benchmark instrumentation during engine initialization; reads prefill/decode counts, TTFT and decode tok/s from `conversation.benchmarkInfo()` and measures generation total with `SystemClock.elapsedRealtime()` |

The engine reload and cache identity include precision and the MTP setting; model initialization occurs before the generation timer and is not included in TTFT or total generation time. Logcat showed the bundle's `mtp_drafter` signature and the LiteRT GPU delegate for the main subgraph. The speed increase and app-reported effective-MTP profile provide the measured A/B evidence; signature presence alone is not treated as proof of drafter use.

The README summarizes the results. The detailed run record and package identity are in [`repro/results/POC_REPRODUCTION_20261124.md`](../repro/results/POC_REPRODUCTION_20261124.md#pixel-10-pro-gpu-mtp-benchmark).

## 6c. Official stock E4B PC CPU/GPU baseline

This baseline uses the official stock `gemma-4-E4B-it.litertlm` package, not a LoRA-patched POC artifact. The local file used was the pinned official bundle from `litert-community/gemma-4-E4B-it-litert-lm@f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`, 3,659,530,240 bytes, SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.

The PC used the existing LiteRT-LM CLI/API 0.19.2 environment. A separate CLI smoke with `--cache=no --verbose --top-k=1 --temperature=0` generated `42` on CPU and GPU; the GPU log selected the RTX 4090 Direct3D 12 adapter. Performance measurements then used `repro/scripts/benchmark_narrator.py` and its tracked `benchmark.narrator.v1.txt` prompt. Settings were 4096-token context, 500-token output ceiling, 200-token validity floor, top-k 64, top-p 0.95, temperature 1.0, and thinking off. Three serial conversations ran for each CPU/GPU and MTP off/on combination. No performance modes overlapped.

| Backend | MTP | Decode tok/s, runs | Median decode | TTFT, runs (s) | Median TTFT | Total time, runs (s) | Output tokens/run |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU / XNNPACK | Off | 13.86, 13.73, 12.88 | 13.73 | 2.16, 2.14, 2.67 | 2.16 | 31.60, 31.87, 34.35 | 409 each |
| CPU / XNNPACK | On | 17.16, 17.45, 17.46 | 17.45 | 2.16, 2.00, 2.01 | 2.01 | 29.09, 28.48, 28.47 | 463 each |
| RTX 4090 / WebGPU-D3D12 | Off | 108.41, 111.58, 111.45 | 111.45 | 0.84, 0.12, 0.12 | 0.12 | 5.19, 4.34, 4.35 | 472 each |
| RTX 4090 / WebGPU-D3D12 | On | 100.92, 154.91, 154.48 | 154.48 | 1.47, 0.12, 0.12 | 0.12 | 5.48, 2.73, 2.73 | 405 each |

Median MTP decode speedup was 27.1% on CPU and 38.6% on GPU. LiteRT logged MTP drafter success of 41.9% for CPU and 47.6% for GPU; the stock bundle's `mtp_drafter` signature was loaded. GPU gates recorded 23,554 MiB free at 0% utilization before both off and on performance phases. The GPU phase used an 8 GiB peak estimate plus a 2 GiB reserve for MTP off, and a 10 GiB estimate plus the same reserve for MTP on. The first run in each GPU mode showed cold-start cost; all three runs remain in the reported median.

The full result record contains the package and prompt hashes, all settings, resource-gate readings, CLI smoke adapter line, and per-run JSON/log locations: [`repro/results/POC_REPRODUCTION_20261124.md`](../repro/results/POC_REPRODUCTION_20261124.md#stock-official-e4b-pc-baseline). No model was copied and the existing LiteRT-LM environment and official-baseline engine cache were reused.

Original host logs and per-run JSON were retained locally at the time of the experiment; they are not bundled in this repository. The published tables and hashes record the available results. The tracked runner and fixture allow new measurements without those logs.

## Useful references

- [Pinned Google model revision](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers/tree/8e3ac78f1a4a97c19f5f194ab362a2206f71d961)
- [Google AI Edge LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM)
- [LiteRT-LM documentation](https://ai.google.dev/edge/litert-lm)
- [Tracked clean reproduction record](../repro/results/POC_REPRODUCTION_20261124.md)
- [Tracked scripts](../repro/scripts/) and [fixtures](../repro/fixtures/)


## Clean proof-of-concept reproduction (repository tracked inputs)

The `repro/` tree contains the implementation dependencies used by this short proof-of-concept: source-format inspection/download, packed quantization, fixed-scale LoRA trainer and merge, official graph payload patch/repack, prompt and one-row synthetic corpus fixtures, and narrator benchmark runner. The approximately 7 MB `repro/fixtures/quantization_provenance.json` is the audited source-to-official graph mapping; it is model metadata, not training corpus. Its historical path fields are descriptive, not required locations; the patcher consumes the tracked tensor mappings and receives actual input paths explicitly. The POC is one optimizer update and is for pipeline validation only, not a quality claim.

### Disposable setup

Use Windows with Python 3.11 x64, Git, curl and a compatible NVIDIA GPU for training. From a fresh clone, run the following in PowerShell at the repository root. The commands create a disposable working directory beside the checkout; adjust its path if needed. Do not point outputs at an existing model cache. The source and stock downloads are approximately 3.53 GB and 3.66 GB, with roughly 16 GB additional working space for unpacking, merging and packaging; environments and compiled caches require extra space. Preserve useful audit outputs before cleaning up.

~~~powershell
# If starting without a checkout:
# git clone https://github.com/robinkrystufek/Forge-E4B-p.git
# Set-Location Forge-E4B-p

$ErrorActionPreference = 'Stop'
function Invoke-Checked {
    param([string]$Executable)
    & $Executable @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable"
    }
}
$src = (Get-Location).Path
if (-not (Test-Path (Join-Path $src 'repro/scripts'))) {
    throw 'Run from the repository root.'
}
$work = Join-Path (Split-Path -Parent $src) 'DISPOSABLE_FORGE_E4B_POC'
if (Test-Path $work) {
    throw 'Choose a new disposable work directory; do not overwrite an existing run.'
}
$run = Join-Path $work 'run'
$models = Join-Path $work 'models/qat-e4b'
$stock = Join-Path $work 'models/stock-e4b'
New-Item -ItemType Directory -Force $run,$models,$stock | Out-Null

Invoke-Checked py -3.11 -m venv (Join-Path $work 'training-venv')
$py = Join-Path $work 'training-venv/Scripts/python.exe'
Invoke-Checked $py -m pip install --upgrade pip
# Install the pinned CUDA wheel first, then the remaining training lock.
Invoke-Checked $py -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126
Invoke-Checked $py -m pip install --index-url https://pypi.org/simple -r (Join-Path $src 'repro/requirements-training.lock')

Invoke-Checked py -3.11 -m venv (Join-Path $work 'export-venv')
$ep = Join-Path $work 'export-venv/Scripts/python.exe'
Invoke-Checked $ep -m pip install -r (Join-Path $src 'repro/requirements-export.lock')
~~~

The export lock records the validated LiteRT-LM 0.19.2 / builder 0.19.2 and LiteRT nightly versions. The training lock pins Transformers to commit `1097d1570e6208063fa49575bb2b6e756a50c2f6` because stock Gemma mobile-QAT loading depends on its `QuantizedLinear` integration. If the CUDA wheel is unavailable for the host, stop rather than silently installing CPU Torch: this trainer targets the GPU and its memory gate.

### Fetch, validate, train, merge

~~~powershell
$s = Join-Path $src 'repro\scripts'
Invoke-Checked $py "$s\download_checkpoint.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output "$models\model.safetensors" --metadata "$run\source-download.json"
if ((Get-FileHash "$models\model.safetensors" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4') {
    throw 'Source checkpoint SHA-256 mismatch'
}
Invoke-Checked $py "$s\download_checkpoint_metadata.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output-dir $models
Invoke-Checked $py "$s\inspect_checkpoint_header.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output "$run\checkpoint-manifest.json"
Invoke-Checked $py "$s\round_trip_checkpoint.py" --weights "$models\model.safetensors" --manifest "$run\checkpoint-manifest.json" --output "$run\round-trip.json"
Invoke-Checked $py "$s\prepare_chat_jsonl.py" --model-dir $models --input "$src\repro\fixtures\poc_chat.jsonl" --output "$run\prepared.jsonl"
# Before GPU training, monitor nvidia-smi until estimated peak plus 2 GiB fits; never stop another process.
Invoke-Checked $py "$s\r14d_qat_lora.py" --arm P10 --model-dir $models --prepared "$run\prepared.jsonl" --output "$run\adapter" --max-samples 1 --estimate-gib 20
Invoke-Checked $py "$s\merge_r14d_qat_lora.py" --source-model $models --manifest "$run\checkpoint-manifest.json" --adapter-dir "$run\adapter\checkpoints\epoch_100" --output "$run\merged" --expected-metrics "$run\adapter\quantization_metrics.jsonl"
~~~

The preprocessing fixture intentionally trains on one supervised answer token. The preparer checks that the rendered full conversation begins with the user-only generation prefix; it masks headers, prompt, template suffix and end-of-turn, and labels only assistant body tokens. Real corpora should use the same schema, source/session split before shuffle, deduplication, and a recorded private data manifest; do not commit private conversations. The training script uses an effective batch of 8, so this one-row POC is one partial-batch optimizer update. Full protocol and quantization invariants are described above.

### Stock LiteRT-LM package, patch, repack

Download the official bundle from `litert-community/gemma-4-E4B-it-litert-lm`, revision `f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`, file `gemma-4-E4B-it.litertlm`; verify SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`. Unpack using the pinned CLI (`litert-lm unpack --help` for syntax), then patch and repack:

~~~powershell
Invoke-Checked $ep "$s\download_stock_litertlm.py" --output "$stock\gemma-4-E4B-it.litertlm"
Invoke-Checked $ep -m pip show litert-lm litert-lm-builder ai-edge-litert
$cli = Join-Path $work 'export-venv\Scripts\litert-lm.exe'
Invoke-Checked $cli unpack "$stock\gemma-4-E4B-it.litertlm" --output-dir "$stock\unpacked"
$patched = "$run\patched\Section10_TFLiteModel_tf_lite_prefill_decode.tflite"
Invoke-Checked $ep "$s\patch_official_graph_from_checkpoint.py" --official-graph "$stock\unpacked\Section10_TFLiteModel_tf_lite_prefill_decode.tflite" --provenance "$src\repro\fixtures\quantization_provenance.json" --merged-model "$run\merged" --merge-report "$run\merged\fixed_scale_merge_report.json" --source-manifest "$run\checkpoint-manifest.json" --output-graph $patched --manifest "$run\patch-manifest.json"
Invoke-Checked $ep "$s\repack_official_litertlm.py" --unpacked-dir "$stock\unpacked" --patched-graph $patched --output "$run\Gemma4-E4B-POC-INT2-4-8-MTP.litertlm" --manifest "$run\package-manifest.json"
~~~

The patcher refuses mismatched base integer hashes, scales, shapes, quant widths, merge metrics, or byte changes outside target payloads. The repacker uses original section metadata and leaves the official MTP drafter intact. Audit package section counts and MTP section presence before inference. The pinned P10 reference SHA above is a previous full-run package and is not expected for a newly trained one-step POC.

### Functional check and performance acceptance

Run the following stock and candidate smoke tests serially. Before GPU work, satisfy the resource gate described above. Confirm the expected answer `42` and the intended adapter in the native log; a process exit code alone is insufficient.

~~~powershell
$candidate = Join-Path $run 'Gemma4-E4B-POC-INT2-4-8-MTP.litertlm'
Invoke-Checked $cli --version
foreach ($model in @("$stock\gemma-4-E4B-it.litertlm", $candidate)) {
    Invoke-Checked $cli run $model --backend=cpu --speculative-decoding=off --cache=no --verbose --top-k=1 --temperature=0 --max-num-tokens=1024 --prompt 'What is 19 + 23? Answer with the number only.'
    Invoke-Checked $cli run $model --backend=gpu --speculative-decoding=on --cache=no --verbose --top-k=1 --temperature=0 --max-num-tokens=1024 --prompt 'What is 19 + 23? Answer with the number only.'
}
~~~

For throughput comparisons use only the exact `benchmark.narrator.v1.txt` fixture and repo runner below, with 3 valid conversations per mode; each must emit at least 200 tokens. Keep modes and benchmark phases exclusive. The runner records package hash, TTFT, input/output counts and decode tok/s. GPU uses WebGPU/Dawn over Direct3D 12; confirm the NVIDIA adapter in verbose CLI logs. Wait until GPU utilization is idle and estimated memory plus 2 GiB reserve fits. Require median MTP-on decode tok/s > median MTP-off on the same backend; report the actual ratio rather than claiming a result from section presence alone.

~~~powershell
Invoke-Checked $ep "$s\benchmark_narrator.py" --model "$run\Gemma4-E4B-POC-INT2-4-8-MTP.litertlm" --cache "$run\engine-cache" --backend cpu --mtp off --runs 3 --output "$run\cpu-off.json"
Invoke-Checked $ep "$s\benchmark_narrator.py" --model "$run\Gemma4-E4B-POC-INT2-4-8-MTP.litertlm" --cache "$run\engine-cache" --backend cpu --mtp on --runs 3 --output "$run\cpu-on.json"
# Repeat on GPU only after the resource gate and exclusive-idle check.
# Use the same commands with --backend gpu and distinct output filenames.
~~~

Windows Python API inference registers the bundled DLL directory with `os.add_dll_directory` and retains its handle until engine close, so `dxcompiler.dll` and `dxil.dll` are discoverable. Do not reuse short synthetic prompts to infer narrator MTP behavior: the 128/64 CLI microbenchmark produced zero draft acceptance and is not Forge-GM's app workload.

The stock downloader resumes interrupted transfers and verifies SHA-256 before renaming its partial file. The export lock intentionally excludes unrelated editable environments.

`prepare_chat_jsonl.py` consumes the `BatchEncoding['input_ids']` returned by the pinned Transformers tokenizer. Do not assume `apply_chat_template(tokenize=True)` directly returns a Python list; its wrapper type can vary across tokenizer implementations. The preparer should fail closed if the user-only generation prefix does not exactly match the full conversation prefix.


Fresh checkout validation revealed the adapter checkpoint for `--max-samples 1` is named `checkpoints/epoch_100` (not `epoch_100_final`); use the emitted checkpoint path or inspect the run directory. Large Hugging Face Xet downloads can stall with Python's `urllib` on some Windows networks. `download_stock_litertlm.py` therefore uses the built-in Windows `curl.exe` with resume/retry support and hashes the completed file; do not restart a partial 3.6 GB transfer from byte zero.

On clean-run smoke tests, `--max-num-tokens=32` failed during engine initialization in both the stock bundle and patched candidate with `DYNAMIC_UPDATE_SLICE`/tensor-shape preparation errors. Using `--max-num-tokens=1024` initialized and generated `42` on the stock model. Keep CLI functional checks at 1024 or the app's 4096 context; tiny context values are not supported by these fixed prefill graph signatures.
