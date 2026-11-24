# Gemma 4 E4B mobile QAT → LoRA → mixed-bit LiteRT-LM with MTP

This is a reproducible implementation record for adapting the pinned mobile-QAT checkpoint with LoRA, preserving its trained mixed INT2/4/8 representation, and packaging it in the official LiteRT-LM bundle while retaining the MTP drafter. It separates a lightweight correctness preflight from the resource-intensive full experiment.

## Reproduction baseline

- Model: `google/gemma-4-E4B-it-qat-mobile-transformers` at revision `8e3ac78f1a4a97c19f5f194ab362a2206f71d961`.
- Tested Windows host: Finnl1, RTX 4090; LiteRT-LM CLI and builder 0.19.2. Windows GPU backend is WebGPU/Dawn on Direct3D 12, not CUDA.
- Pixel: Pixel 10 Pro, Android 17; Forge GM `1.0-v113-mtp-probe`, LiteRT-LM 0.19.1, LiteRT 2.3.2.
- Training environment reused from `D:\Forge-GM\experiment\mobile_qat_feasibility\venv` (Python 3.11.9, Torch 2.14.1+cu126, Transformers 5.24.0.dev0, PEFT/Accelerate stack). Export reuses `litert_export_venv`; LiteRT-LM 0.19.2. Do not make a new environment if these existing environments still fit.
- The actual R14-D report, package audit and Pixel record are linked in the [Forge-GM experiment tree](https://github.com/robinkrystufek/Forge-GM-lab/tree/main/benchmarks/mobile_qat_feasibility). Main report files in the original workspace are `experiment/output/r14d-qat-lora-20261123_0818/reports/DEPLOYMENT_REPORT.md` and `FINAL_VERDICT.md`.

## 1. Acquire and validate the exact source

Use the repository downloader and pin the revision. The metadata is part of the model: weights alone do not describe widths, groups, scales or packing.

~~~powershell
$root = 'D:\Forge-GM\experiment\Forge-GM-lab\benchmarks\mobile_qat_feasibility'
$cache = 'D:\Forge-GM\experiment\output\r14d-qat-lora-20261123_0818\source'
$run = 'D:\Forge-GM\experiment\output\r14d-qat-lora-20261123_0818'
python "$root\scripts\download_checkpoint.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output "$cache\model.safetensors" --metadata "$cache\metadata_manifest.json"
python "$root\scripts\download_checkpoint_metadata.py" --repo google/gemma-4-E4B-it-qat-mobile-transformers --revision 8e3ac78f1a4a97c19f5f194ab362a2206f71d961 --output-dir "$cache"
python "$root\scripts\round_trip_checkpoint.py" --weights "$cache\model.safetensors" --manifest "$cache\metadata_manifest.json" --output "$run\audits\source_round_trip.json"
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
python "$root\scripts\r14d_qat_lora.py" --arm P10 --prepared "$run\training\prepared\P10.jsonl" --output "$run\training\P10" --estimate-gib 20
~~~

Before every GPU phase check available VRAM, utilization, and active processes. Wait/retry until estimated peak plus the agreed 2 GiB reserve fits. Never terminate an existing workload. Keep throughput benchmarks exclusive and separate from functional checks. On the recorded setup the gate estimate was 20 GiB plus 2 GiB reserve.

## 4. Merge without changing quantization

Merge the trained floating adapter delta into the source's quantized grid using the original fixed scales and quantizer. Use the training-produced quantization metrics as the expected merge record:

~~~powershell
python "$root\scripts\merge_r14d_qat_lora.py" --source-model "$cache" --manifest "$cache\metadata_manifest.json" --adapter-dir "$run\training\P10\checkpoints\epoch_100_final" --output "$run\export\merged_fixed_scale\P10" --expected-metrics "$run\training\P10\quantization_metrics.jsonl"
~~~

Require `training_merge_packed_exact: true`; fail on mismatched scales or quantization metrics. In R14-D, exactly 77 INT4 tensors and 183,610 packed values changed; INT2=0 and INT8=0. Verify decoded integer values, shapes, widths, scale hashes, q-axis/zero-points and op use. This is still the source INT2/4/8 scheme, not a new uniform INT4 or dynamic quantization.

## 5. Patch official LiteRT-LM graphs and package with MTP

Use the exact official LiteRT-LM conversion/builder version and pristine stock E4B bundle/config. Patch only corresponding prefill/decode packed weight payloads after proving unique tensor matches, base integer hashes, readback, and unchanged bytes outside payload ranges. The official builder externalizes weights and can change section layout; compare tensor payload provenance, not raw section offsets. Retain the official MTP drafter section/config unchanged. Do not rebuild/approximate the drafter or omit it.

Audit package contents: all ten graph structures match expected counts; 1,658 externalized stock weight payloads matched byte-for-byte with zero mismatch; official drafter has 23 quantized buffers (13 INT4, 10 INT8), unchanged. P10 operator provenance maps 77 adapted tensors to 242 FULLY_CONNECTED occurrences. These checks catch accidental graph drift and wrong tensor replacement.

Expected audited P10 artifact: 3,492,003,840 bytes, SHA-256 `18108925d98c552107e2e3d0724c025bb839fa8d70cd6ad5e1aa86ea460b0cfd`. Recompute and compare hashes in a reproduction; do not assume every rebuild is byte-deterministic. The package was created by patching official graph payloads and repacking through official LiteRT-LM builder, not by treating a raw safetensors file as a LiteRT-LM package.

## 6. Lightweight checks and inference validation

Before a candidate run, check installed CLI version and use the already installed environment. On Finnl1:

~~~powershell
$cli = 'D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv\Scripts\litert-lm.exe'
& $cli --version
& $cli run '<model.litertlm>' --backend=gpu --cache=no --verbose --top-k=1 --temperature=0 --prompt 'Write a short test response.'
~~~

Confirm verbose output selects the NVIDIA RTX 4090 adapter with Direct3D 12, generates a response, and exits successfully. Capture the full initialization log and first native error if it fails. The WebGPU/Dawn backend is not CUDA. For Python API engine creation on Windows, add the LiteRT-LM package folder via `os.add_dll_directory(...)` so `dxcompiler.dll` and `dxil.dll` resolve, and keep that directory handle alive until the engine closes.

Smoke-test a known-good stock package first, then candidate with identical flags. `--cache=no` avoids compiled-model sidecars. Keep functional smoke separate from performance measurements; use an exclusive repeated-run benchmark only for performance comparisons.

Android was tested in Forge GM's isolated debug probe app, not Edge Gallery. On Pixel, verify pushed file SHA, select Model-default precision and GPU, and test MTP on/off in separate runs with identical app/runtime/prompt/context when comparing. The 4096 context with initial FP32 compatibility setting was LMK-killed five times. A 1024 requested context, Model-default precision, GPU run completed with MTP off; this only validates that debug configuration and does not establish 4096-context safety.

Observed single-run diagnostics on Pixel 10 Pro: MTP off, 458 input/446 output, TTFT 2.86 s, 7.59 token/s, 61.54 s total. MTP on, 458 input/396 output, TTFT 2.91 s, 13.61 token/s, 31.97 s total. These were not a controlled paired benchmark: output lengths differed and MTP-on trace is a diagnostic record, so treat as directional observations, not a guaranteed speedup. The corresponding 5-run cold-start host proxy median was 6.786 s (range 6.759–7.115); it measures startup/graph-init plus a short response, not steady-state token throughput or MTP A/B.

## 7. Lightweight reproduction gate

Before an expensive run, reuse the existing training environment and run its CPU test suite plus tiny synthetic quantizer/LoRA preflight. The preflight should prove: zero-init adapter exactly preserves base output; controlled adapter changes cross the fixed quantizer; A and B receive gradients; base packed weights and scales remain unchanged; pack/depack is exact; and a few optimizer steps lower loss. Do not load the full E4B model or run the training regiment for this check.

For this guide change, 28 existing CPU unit tests passed and the tiny actual trainer preflight passed all six invariants with loss 13.8330 → 12.9280. No full training, package rebuild, GPU inference, or Pixel rerun was performed. The preflight result is in the experiment's task log; reuse the existing scripts/tests rather than adding another environment.

## 8. Executed short proof of concept and package structure

A one-step synthetic-data proof of execution was run with the same QAT-aware trainer before packaging. The single synthetic conversation was `What is two plus two?` → `4`, tokenized with the pinned tokenizer/chat template and last-assistant-only labels. It produced 24 input tokens and one supervised token after masking EOT/newline. This is intentionally only a path check: one optimizer update cannot establish a loss trend, useful adaptation, or quality.

Use the same existing training/export environments and artifacts described above. For a truly short run, prepare a tiny synthetic JSONL row in the trainer's prepared format and pass `--max-samples 1`; that option still loads the whole model. Set a hard run-time budget and stop before ten minutes if a future machine behaves differently. The actual run on the RTX 4090 completed in about ten seconds including model loading, with 20 GiB estimated peak plus 2 GiB reserve checked before launch. The step had nonzero gradients in the selected modules; the fixed-scale merge changed 127 INT4 values in one tensor, and no INT2 or INT8 values changed. Its single loss was 0.00132, so there was no multi-step loss comparison.

Run the production-equivalent fixed-scale merge, audit patch, and official builder in that order. For POC mode, the actual output was `SimpleLoRA-POC-Gemma4-E4B-MixedQAT-MTP.litertlm`, SHA-256 `4d066ef5f5fcb31cb0428571ef6ea7b9d6be9431db418a7ad2be6240aa9478b1`. The merge fake-quant hash matched exactly; patch readback passed and all graph bytes outside the one quantized payload range were identical to the stock graph.

### Package structure changes made for LiteRT-LM

The packaging change is a storage/layout transformation around an official graph, plus a verified payload edit:

1. Start from Google's official E4B `.litertlm`, unpack its sections, and preserve the stock tokenizer, metadata and all unmodified model graphs.
2. Replace only the matched packed INT4 tensor bytes in the official `tf_lite_prefill_decode` graph. Keep that graph's operator topology, buffer metadata and quantization parameters intact. For the POC, the single matched tensor was `model.language_model.layers.22.self_attn.k_proj.weight`; for the full R14-D P10 artifact, 77 INT4 tensors were patched. INT2/INT8, MTP drafter and non-target tensors stay at their source values.
3. Point the original `model.toml` prefill/decode graph path at the patched graph. The LiteRT-LM builder externalizes eligible constants from eight graphs into separate `TFLiteWeights` sections and redirects their buffers. Thus the stock 12-section bundle becomes a 20-section bundle; eight weight sections are added, including the MTP drafter weights. The builder also adjusts package metadata, redirects two buffers and deduplicates 167,772,160 bytes. Do not expect raw section offsets or all graph-section bytes to match stock.
4. Verify semantic/integer tensor provenance, scale hashes, graph readback, byte equality outside patched payload ranges, and package inventory. Confirm the MTP drafter graph and its external weights remain present. The POC package retained the MTP graph (3,839,640 bytes) and its `TFLiteWeights` section (41,287,680 bytes).

This is why patching the audited official graph payloads and then invoking the official builder is the working route; exporting a raw checkpoint or rebuilding only the main graph would omit the multimodal sections, external-weight layout or drafter.

### Short local inference smoke

With LiteRT-LM CLI 0.19.2 on Finnl1/RTX 4090, run the same three short prompts on CPU and GPU, with speculative decoding off and on. For the tested POC package use `--max-num-tokens=1024` (requested context limit); all 12 executions returned the expected answers on both backends. GPU logs selected the NVIDIA RTX 4090 / Direct3D 12 adapter. MTP-on logs loaded the `mtp_drafter` signature and recorded success rates of 1.000, 0.667 and 0.333 on CPU, and 0.667, 0.667 and 0.333 on GPU for the three prompts. These are functional smoke observations, not performance measurements.

An initial attempt with `--max-num-tokens=64` failed: CPU could not allocate tensors (the decode graph's dynamic update slice exceeded the short state) and GPU reported `Chosen prefill work group size exceeds available state entries (64)`. The CLI nevertheless returned process exit code 0 after printing a traceback. Therefore, check logs for exceptions and an actual generated response; an exit code alone is not a pass. The successful 1024-token retry generated `42`, `Paris` and `Green.` for the arithmetic, capital and color prompts. This debug cap is not evidence that larger contexts are safe.

POC training, merge report, patch/repack manifests, full inference logs, and the POC package were retained under `D:\Forge-GM\experiment\output\r14d-qat-lora-20261123_0818\evaluation\simple_lora_poc`. The POC is not a quality-qualified model and must not be presented as a trained release.

## Useful references

- [Pinned Google model revision](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers/tree/8e3ac78f1a4a97c19f5f194ab362a2206f71d961)
- [Google AI Edge LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM)
- [LiteRT-LM documentation](https://ai.google.dev/edge/litert-lm)
- Experiment source: `D:\Forge-GM\experiment\Forge-GM-lab\benchmarks\mobile_qat_feasibility`
- Audit outputs: `D:\Forge-GM\experiment\output\r14d-qat-lora-20261123_0818\export\P10\package_manifest.json`, `patch_manifest.json`, `operator_provenance.json`, `evaluation\package_section_audit\PACKAGE_SECTION_AUDIT.md`, and `evaluation\pixel_gpu\pixel_p10_gpu_retry_record.json`.

