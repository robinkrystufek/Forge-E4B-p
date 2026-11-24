# Gemma 4 E4B mobile QAT → LoRA → mixed-bit LiteRT-LM with MTP

This is a reproducible implementation record for adapting the pinned mobile-QAT checkpoint with LoRA, preserving its trained mixed INT2/4/8 representation, and packaging it in the official LiteRT-LM bundle while retaining the MTP drafter. It separates a lightweight correctness preflight from the resource-intensive full experiment.

## Reproduction baseline

- Model: `google/gemma-4-E4B-it-qat-mobile-transformers` at revision `8e3ac78f1a4a97c19f5f194ab362a2206f71d961`.
- Tested Windows host: Finnl1, RTX 4090; LiteRT-LM CLI and builder 0.19.2. Windows GPU backend is WebGPU/Dawn on Direct3D 12, not CUDA.
- Pixel: Pixel 10 Pro, Android 17; Forge GM `1.0-v113-mtp-probe`, LiteRT-LM 0.19.1, LiteRT 2.3.2.
- Training environment reused from `D:\Forge-GM\experiment\mobile_qat_feasibility\venv` (Python 3.11.9, Torch 2.14.1+cu126, Transformers 5.24.0.dev0, PEFT/Accelerate stack). Export reuses `litert_export_venv`; LiteRT-LM 0.19.2. Do not make a new environment if these existing environments still fit.
- The actual R14-D report, package audit and Pixel record are linked in the [Forge-GM experiment tree](https://github.com/robinkrystufek/Forge-GM-lab/tree/main/benchmarks/mobile_qat_feasibility). Main report files in the original workspace are `experiment/output/r14d-qat-lora-20261123_0818/reports/DEPLOYMENT_REPORT.md` and `FINAL_VERDICT.md`.

## Repository completeness audit (2026-11-24)

The procedure below is **not yet self-contained in this GitHub repository**. This guide is committed, but the implementation scripts, test fixtures, lock files, source manifests, and model/package binaries it names live in the FINNL1 workspace at the paths listed here; those files are not included in this repo's PR. A clean checkout therefore cannot execute the steps as written until those exact versioned scripts/fixtures and dependency pins are added or published. The linked Forge-GM-lab path currently supplies the experiment record, but is a separate workspace/repository dependency. Treat this document as a location-indexed runbook, not a complete reproducibility bundle, until those dependencies are vendored or linked to immutable source revisions.

To close that gap, the repository needs (at minimum) the exact training/merge/patch/repack/provenance scripts and their imports, a generic JSONL-to-token/label preparation utility, synthetic test fixtures plus lightweight checks, pinned training/export dependency manifests, and instructions with immutable revisions/hashes for both the QAT source checkpoint and official stock LiteRT-LM package. Full model binaries and private corpora should remain external; their download URLs, revisions, expected hashes, and storage requirements belong in the guide. Do not describe the artifact as reproducible from this repository alone until a clean-checkout smoke test uses only repository-tracked code plus those documented external downloads.

## 1. Acquire and validate the exact source

Use the repository downloader and pin the revision. The metadata is part of the model: weights alone do not describe widths, groups, scales or packing.

~~~powershell
$python = 'D:\Forge-GM\experiment\mobile_qat_feasibility\venv\Scripts\python.exe'
$exportPython = 'D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv\Scripts\python.exe'
$scripts = 'D:\Forge-GM\experiment\Forge-GM-lab\benchmarks\mobile_qat_feasibility\scripts'
$cache = 'D:\Forge-GM\experiment\mobile_qat_feasibility\cache'
$run = 'D:\Forge-GM\experiment\output\r14d-qat-lora-20261123_0818'
New-Item -ItemType Directory -Force "$run\audits" | Out-Null
if (-not (Test-Path "$cache\model.safetensors")) {
  & $python "$scripts\download_checkpoint.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output "$cache\model.safetensors" --metadata "$run\audits\download_manifest.json"
}
& $python "$scripts\download_checkpoint_metadata.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output-dir "$cache"
& $python "$scripts\inspect_checkpoint_header.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output "$run\audits\checkpoint_manifest.json"
& $python "$scripts\round_trip_checkpoint.py" --weights "$cache\model.safetensors" --manifest "$run\audits\checkpoint_manifest.json" --output "$run\audits\source_round_trip.json"
~~~

Expected inventory: 613 packed tensors: 111 INT2, 306 INT4 and 196 INT8. INT2/INT4 use low-value-first offset-binary packing in U8; INT8 is signed. Stop if metadata cannot be reconciled or the no-op round-trip changes packed bytes or decoded values. Never reinterpret these payloads as ordinary BF16 tensors.

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

~~~powershell
& $python "$scripts\r14d_qat_lora.py" --arm P10 --prepared "$run\training\prepared\P10.jsonl" --output "$run\training\P10" --estimate-gib 20
~~~

Before every GPU phase check available VRAM, utilization, and active processes. Wait/retry until estimated peak plus the agreed 2 GiB reserve fits. Never terminate an existing workload. Keep throughput benchmarks exclusive and separate from functional checks. On the recorded setup the gate estimate was 20 GiB plus 2 GiB reserve.

## 4. Merge without changing quantization

Merge the trained floating adapter delta into the source's quantized grid using the original fixed scales and quantizer. Use the training-produced quantization metrics as the expected merge record:

~~~powershell
& $python "$scripts\merge_r14d_qat_lora.py" --source-model "$cache" --manifest "$run\audits\checkpoint_manifest.json" --adapter-dir "$run\training\P10\checkpoints\epoch_100_final" --output "$run\export\merged_fixed_scale\P10" --expected-metrics "$run\training\P10\quantization_metrics.jsonl"
~~~

Require `training_merge_packed_exact: true`; fail on mismatched scales or quantization metrics. In R14-D, exactly 77 INT4 tensors and 183,610 packed values changed; INT2=0 and INT8=0. Verify decoded integer values, shapes, widths, scale hashes, q-axis/zero-points and op use. This is still the source INT2/4/8 scheme, not a new uniform INT4 or dynamic quantization.

## 5. Patch official LiteRT-LM graphs and package with MTP

Use the exact official LiteRT-LM conversion/builder version and pristine stock E4B bundle/config. Patch only corresponding prefill/decode packed weight payloads after proving unique tensor matches, base integer hashes, readback, and unchanged bytes outside payload ranges. The official builder externalizes weights and can change section layout; compare tensor payload provenance, not raw section offsets. Retain the official MTP drafter section/config unchanged. Do not rebuild/approximate the drafter or omit it.

Run the patcher and builder with the reused export environment; it contains `ai_edge_litert` and `litert_lm_builder`. The patch audit JSON must report `merge_packed_hash_match: true` and `outside_payload_bytes_unchanged: true` before you repack.

~~~powershell
$official = 'D:\Gemma4-MTP-work\official-reference\unpacked'
$provenance = 'D:\Forge-GM\experiment\Forge-GM-lab\benchmarks\mobile_qat_feasibility\quantization_provenance.json'
$merged = "$run\export\merged_fixed_scale\P10"
$patched = "$run\export\P10\patched\Section10_R14D_P10.tflite"
$pkg = "$run\export\P10\Gemma4-E4B-R14D-P10-QAT-INT2-INT4-INT8.litertlm"
& $exportPython "$scripts\patch_official_graph_from_checkpoint.py" --official-graph "$official\Section10_TFLiteModel_tf_lite_prefill_decode.tflite" --provenance $provenance --merged-model $merged --merge-report "$merged\fixed_scale_merge_report.json" --source-manifest "$run\audits\checkpoint_manifest.json" --output-graph $patched --manifest "$run\export\P10\patch_manifest.json"
if ($LASTEXITCODE -ne 0) { throw 'Official graph payload patch failed' }
& $exportPython "$scripts\repack_official_litertlm.py" --unpacked-dir $official --patched-graph $patched --output $pkg --manifest "$run\export\P10\package_manifest.json"
if ($LASTEXITCODE -ne 0) { throw 'Official LiteRT-LM repack failed' }
~~~
Audit package contents: all ten graph structures match expected counts; 1,658 externalized stock weight payloads matched byte-for-byte with zero mismatch; official drafter has 23 quantized buffers (13 INT4, 10 INT8), unchanged. P10 operator provenance maps 77 adapted tensors to 242 FULLY_CONNECTED occurrences. These checks catch accidental graph drift and wrong tensor replacement.

Expected audited P10 artifact: 3,492,003,840 bytes, SHA-256 `18108925d98c552107e2e3d0724c025bb839fa8d70cd6ad5e1aa86ea460b0cfd`. Recompute and compare hashes in a reproduction; do not assume every rebuild is byte-deterministic. The package was created by patching official graph payloads and repacking through official LiteRT-LM builder, not by treating a raw safetensors file as a LiteRT-LM package.

## 6. Lightweight checks and inference validation

Before a candidate run, check installed CLI version and use the already installed environment. On Finnl1:

~~~powershell
$cli = 'D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv\Scripts\litert-lm.exe'
& $cli --version
& $cli run '<model.litertlm>' --backend=cpu --speculative-decoding=off --cache=no --verbose --top-k=1 --temperature=0 --max-num-tokens=1024 --prompt 'What is 19 + 23? Answer with the number only.'
& $cli run '<model.litertlm>' --backend=gpu --speculative-decoding=on --cache=no --verbose --top-k=1 --temperature=0 --max-num-tokens=1024 --prompt 'What is 19 + 23? Answer with the number only.'
~~~

Confirm verbose output selects the NVIDIA RTX 4090 adapter with Direct3D 12, generates a response, and exits successfully. Capture the full initialization log and first native error if it fails. Treat any traceback, runtime error, or missing generated response as failure even if the CLI process exit code is zero. The WebGPU/Dawn backend is not CUDA. For Python API engine creation on Windows, add the LiteRT-LM package folder via `os.add_dll_directory(...)` so `dxcompiler.dll` and `dxil.dll` resolve, and keep that directory handle alive until the engine closes.

Smoke-test a known-good stock package first, then candidate with identical flags. `--cache=no` avoids compiled-model sidecars. Keep functional smoke separate from performance measurements; use an exclusive repeated-run benchmark only for performance comparisons.

Android was tested in Forge GM's isolated debug probe app, not Edge Gallery. On Pixel, verify pushed file SHA, select Model-default precision and GPU, and test MTP on/off in separate runs with identical app/runtime/prompt/context when comparing. The 4096 context with initial FP32 compatibility setting was LMK-killed five times. A 1024 requested context, Model-default precision, GPU run completed with MTP off; this only validates that debug configuration and does not establish 4096-context safety.

The supplied Forge-GM v116 export contains repeated P10 GPU narrator measurements. The corresponding app-aligned PC comparison, including per-run TTFT and decode tok/s, is recorded in section 6a. The separate 1024-context MTP-off run above remains a functional probe and should not be combined with the 4096-context performance results.

## 6a. Forge-GM app-aligned P10 narrator benchmark on Finnl1

The previous host table in this section used a short CLI microbenchmark (128 synthetic prefill tokens / 64 forced decode tokens). That microbenchmark is valid only for that short fixed-token workload; its MTP acceptance was 0%, and it does not model Forge-GM's long narrator benchmark. The comparison below replaces it for app-performance conclusions.

The Pixel record and host run use the exact same P10 package SHA-256, `18108925d98c552107e2e3d0724c025bb839fa8d70cd6ad5e1aa86ea460b0cfd`. Forge-GM defines the fixed workload as `benchmark.narrator` version 1, the `BENCHMARK_NARRATOR` text in `D:\Forge-GM\experiment\Forge-GM-current\src\prompts\templates.ts`. Its UTF-8 SHA-256 is `de63332fe826bdaa0ca030662f01f150799186b04218aecd8cd966e3a1abbd38`. The app uses a 500-token output ceiling, 4096-token context, thinking disabled, and its narrator sampler defaults (top-k 64, top-p 0.95, temperature 1.0). Forge-GM accepts long-form performance samples only at 200 or more output tokens.

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

Host logs, exact prompt copy, runner, JSON results, and GPU gate records are in `D:\Forge-GM\experiment\output\r14d-qat-lora-20261123_0818\evaluation\litertlm_mtp_host_metrics\rerun_20261124\`. The shared pre-existing LiteRT-LM engine cache was reused and expanded; it remains in place. No model or environment was created for this benchmark.

## Useful references

- [Pinned Google model revision](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers/tree/8e3ac78f1a4a97c19f5f194ab362a2206f71d961)
- [Google AI Edge LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM)
- [LiteRT-LM documentation](https://ai.google.dev/edge/litert-lm)
- Experiment source: `D:\Forge-GM\experiment\Forge-GM-lab\benchmarks\mobile_qat_feasibility`
- Audit outputs: `D:\Forge-GM\experiment\output\r14d-qat-lora-20261123_0818\export\P10\package_manifest.json`, `patch_manifest.json`, `operator_provenance.json`, `evaluation\package_section_audit\PACKAGE_SECTION_AUDIT.md`, and `evaluation\pixel_gpu\pixel_p10_gpu_retry_record.json`.


## Clean proof-of-concept reproduction (repository tracked inputs)

The `repro/` tree contains the implementation dependencies used by this short proof-of-concept: source-format inspection/download, packed quantization, fixed-scale LoRA trainer and merge, official graph payload patch/repack, prompt and one-row synthetic corpus fixtures, and narrator benchmark runner. The 7 MB `fixtures/quantization_provenance.json` is the audited source-to-official graph mapping; it is public model metadata, not training corpus. The POC is one optimizer update and is for pipeline validation only, not a quality claim.

### Disposable setup

Use Python 3.11 x64. These steps intentionally create `DISPOSABLE_FORGE_E4B_POC` and a fresh venv. Do not point outputs at a valuable shared model cache. The source model requires 3.53 GB; the stock bundle is 3.66 GB; unpacked sections, merged weights, patched graph and package require roughly 16 GB working space. Preserve the output package and audit directory before removing the disposable workspace.

~~~powershell
$root = (Get-Location).Path
$work = Join-Path $root 'DISPOSABLE_FORGE_E4B_POC'
$py = Join-Path $work 'venv\Scripts\python.exe'
$src = Join-Path $work 'repo'
$run = Join-Path $work 'run'
$models = Join-Path $work 'models\qat-e4b'
$stock = Join-Path $work 'models\stock-e4b'
New-Item -ItemType Directory -Force $work,$run,$models,$stock | Out-Null
py -3.11 -m venv (Join-Path $work 'venv')
& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $src 'repro\requirements-training.lock')
# Install the CUDA 12.6 Torch wheel from the official PyTorch index before the lock:
# & $py -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126
# Install export pins in a separate venv because LiteRT-LM and training have distinct native stacks.
py -3.11 -m venv (Join-Path $work 'export-venv')
$ep = Join-Path $work 'export-venv\Scripts\python.exe'
& $ep -m pip install -r (Join-Path $src 'repro\requirements-export.lock')
~~~

The export lock records the validated LiteRT-LM 0.19.2 / builder 0.19.2 and LiteRT nightly versions. The training lock pins Transformers to commit `1097d1570e6208063fa49575bb2b6e756a50c2f6` because stock Gemma mobile-QAT loading depends on its `QuantizedLinear` integration. If the CUDA wheel is unavailable for the host, stop rather than silently installing CPU Torch: this trainer targets the GPU and its memory gate.

### Fetch, validate, train, merge

~~~powershell
$s = Join-Path $src 'repro\scripts'
& $py "$s\download_checkpoint.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output "$models\model.safetensors" --metadata "$run\source-download.json"
& $py "$s\download_checkpoint_metadata.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output-dir $models
& $py "$s\inspect_checkpoint_header.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output "$run\checkpoint-manifest.json"
& $py "$s\round_trip_checkpoint.py" --weights "$models\model.safetensors" --manifest "$run\checkpoint-manifest.json" --output "$run\round-trip.json"
& $py "$s\prepare_chat_jsonl.py" --model-dir $models --input "$src\repro\fixtures\poc_chat.jsonl" --output "$run\prepared.jsonl"
# Before GPU training, monitor nvidia-smi until estimated peak plus 2 GiB fits; never stop another process.
& $py "$s\r14d_qat_lora.py" --arm P10 --model-dir $models --prepared "$run\prepared.jsonl" --output "$run\adapter" --max-samples 1 --estimate-gib 20
& $py "$s\merge_r14d_qat_lora.py" --source-model $models --manifest "$run\checkpoint-manifest.json" --adapter-dir "$run\adapter\checkpoints\epoch_100" --output "$run\merged" --expected-metrics "$run\adapter\quantization_metrics.jsonl"
~~~

The preprocessing fixture intentionally trains on one supervised answer token. The preparer checks that the rendered full conversation begins with the user-only generation prefix; it masks headers, prompt, template suffix and end-of-turn, and labels only assistant body tokens. Real corpora should use the same schema, source/session split before shuffle, deduplication, and a recorded private data manifest; do not commit private conversations. The training script uses an effective batch of 8, so this one-row POC is one partial-batch optimizer update. Full protocol and quantization invariants are described above.

### Stock LiteRT-LM package, patch, repack

Download the official bundle from `litert-community/gemma-4-E4B-it-litert-lm`, revision `f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`, file `gemma-4-E4B-it.litertlm`; verify SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`. Unpack using the pinned CLI (`litert-lm unpack --help` for syntax), then patch and repack:

~~~powershell
& $ep -m pip show litert-lm litert-lm-builder ai-edge-litert
$cli = Join-Path $work 'export-venv\Scripts\litert-lm.exe'
& $cli unpack "$stock\gemma-4-E4B-it.litertlm" --output-dir "$stock\unpacked"
$patched = "$run\patched\Section10_TFLiteModel_tf_lite_prefill_decode.tflite"
& $ep "$s\patch_official_graph_from_checkpoint.py" --official-graph "$stock\unpacked\Section10_TFLiteModel_tf_lite_prefill_decode.tflite" --provenance "$src\repro\fixtures\quantization_provenance.json" --merged-model "$run\merged" --merge-report "$run\merged\fixed_scale_merge_report.json" --source-manifest "$run\checkpoint-manifest.json" --output-graph $patched --manifest "$run\patch-manifest.json"
& $ep "$s\repack_official_litertlm.py" --unpacked-dir "$stock\unpacked" --patched-graph $patched --output "$run\Gemma4-E4B-POC-INT2-4-8-MTP.litertlm" --manifest "$run\package-manifest.json"
~~~

The patcher refuses mismatched base integer hashes, scales, shapes, quant widths, merge metrics, or byte changes outside target payloads. The repacker uses original section metadata and leaves the official MTP drafter intact. Audit package section counts and MTP section presence before inference. The pinned P10 reference SHA above is a previous full-run package and is not expected for a newly trained one-step POC.

### Functional check and performance acceptance

Run a short CPU and GPU generation first. For throughput comparisons use only the exact `benchmark.narrator.v1.txt` fixture and repo runner below, with 3 valid conversations per mode; each must emit at least 200 tokens. Keep modes and benchmark phases exclusive. The runner records package hash, TTFT, input/output counts and decode tok/s. GPU uses WebGPU/Dawn over Direct3D 12; confirm the NVIDIA adapter in verbose CLI logs. Wait until GPU utilization is idle and estimated memory plus 2 GiB reserve fits. Require median MTP-on decode tok/s > median MTP-off on the same backend; report the actual ratio rather than claiming a result from section presence alone.

~~~powershell
& $ep "$s\benchmark_narrator.py" --model "$run\Gemma4-E4B-POC-INT2-4-8-MTP.litertlm" --cache "$run\engine-cache" --backend cpu --mtp off --runs 3 --output "$run\cpu-off.json"
& $ep "$s\benchmark_narrator.py" --model "$run\Gemma4-E4B-POC-INT2-4-8-MTP.litertlm" --cache "$run\engine-cache" --backend cpu --mtp on --runs 3 --output "$run\cpu-on.json"
# Repeat on GPU only after the resource gate and exclusive-idle check.
~~~

Windows Python API inference registers the bundled DLL directory with `os.add_dll_directory` and retains its handle until engine close, so `dxcompiler.dll` and `dxil.dll` are discoverable. Do not reuse short synthetic prompts to infer narrator MTP behavior: the 128/64 CLI microbenchmark produced zero draft acceptance and is not Forge-GM's app workload.

Use `download_stock_litertlm.py --output <path>` to fetch the pinned official package; it verifies the SHA before renaming the `.partial` file. The short clean reproduction also needs an explicit `--model-dir` in the trainer. Install the CUDA wheel before the rest of the training lock (`pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126`, then install the lock with PyPI as the index); this avoids resolving CUDA Torch from the general package index. The export lock is intentionally minimal and excludes the unrelated editable `litert-torch` checkout from the shared environment.

The official stock package download command is:

~~~powershell
& $ep "$s\download_stock_litertlm.py" --output "$stock\gemma-4-E4B-it.litertlm"
~~~

`prepare_chat_jsonl.py` consumes the `BatchEncoding['input_ids']` returned by the pinned Transformers tokenizer. Do not assume `apply_chat_template(tokenize=True)` directly returns a Python list; its wrapper type can vary across tokenizer implementations. The preparer should fail closed if the user-only generation prefix does not exactly match the full conversation prefix.


Fresh checkout validation revealed the adapter checkpoint for `--max-samples 1` is named `checkpoints/epoch_100` (not `epoch_100_final`); use the emitted checkpoint path or inspect the run directory. Large Hugging Face Xet downloads can stall with Python's `urllib` on some Windows networks. `download_stock_litertlm.py` therefore uses the built-in Windows `curl.exe` with resume/retry support and hashes the completed file; do not restart a partial 3.6 GB transfer from byte zero.

On clean-run smoke tests, `--max-num-tokens=32` failed during engine initialization in both the stock bundle and patched candidate with `DYNAMIC_UPDATE_SLICE`/tensor-shape preparation errors. Using `--max-num-tokens=1024` initialized and generated `42` on the stock model. Keep CLI functional checks at 1024 or the app's 4096 context; tiny context values are not supported by these fixed prefill graph signatures.
