# Clean proof-of-concept reproduction record (2026-11-24)

## Reproduction setup

- Repository: `robinkrystufek/Forge-E4B-p`, branch `docs/e4b-mobile-qat-guide`, commit `28c6909` plus this evidence commit.
- Run root: `D:\Forge-GM\experiment\output\DISPOSABLE_FORGE_E4B_P_REPRO_20261124` (explicitly disposable, created from a fresh clone).
- Fresh `training-venv` and `export-venv`, Python 3.11. Training lock installed Torch 2.14.1+cu126 and Transformers 5.24.0.dev0 at `1097d1570e6208063fa49575bb2b6e756a50c2f6`; export lock installed LiteRT-LM/API/builder 0.19.2, ai-edge-litert-nightly 2.4.0.dev20261122, ai-edge-quantizer-nightly 0.10.1.dev20261113, NumPy 2.4.6, FlatBuffers 25.12.19, safetensors 0.9.0rc1.
- Source checkpoint downloaded from `google/gemma-4-E4B-it-qat-mobile-transformers@8e3ac78f1a4a97c19f5f194ab362a2206f71d961`; 3,525,094,516 bytes; SHA-256 `391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4`.
- Official stock package downloaded from `litert-community/gemma-4-E4B-it-litert-lm@f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`; 3,659,530,240 bytes; SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.
- CPU and GPU are the same RTX 4090 host; GPU backend is WebGPU/Dawn over Direct3D 12. Three 60-second idle samples before performance measurement showed 23,387 MiB free and 0% utilization. Windows hid executable paths for three listed device PIDs; none could be confirmed as a separate model workload, and no process was stopped.

## From source weights to package

- Source inspection found 3,104 tensors; round-trip checked all 613 packed quantized weight tensors: 111 INT2, 306 INT4, 196 INT8. Changed bytes: 0; bit-exact: true.
- `poc_chat.jsonl` was tokenized with the pinned model tokenizer. The prepared sample has 24 token IDs and one supervised assistant-body token. The run was one optimizer update (`--max-samples 1`, partial batch 1), rank 16 / alpha 32 / A-factor dropout 0.05. It loaded and mapped all 132 shared-KV-aware attention target modules, completed with finite loss 0.001321 and nonzero adapter gradients. This is pipeline validation only; a single training example does not measure generalization.
- Fixed-scale merge changed 127 packed values in one INT4 tensor; no INT2 or INT8 values changed. `training_merge_packed_exact=true`.
- Graph patch audit: one tensor patched, `merge_packed_hash_match=true`, `outside_payload_bytes_unchanged=true`.
- Official repack created a 3,492,003,840-byte, 20-section LiteRT-LM package with original config, tokenizer, and MTP drafter section. Package SHA-256: `d9088bd7e39056f9a470332d4a6f9fac309e05de7392e181606382091b225b7f`.
- Candidate CPU smoke generated `42` at 1024 tokens. Candidate GPU smoke generated `42`; verbose output selected `NVIDIA GeForce RTX 4090, arch=lovelace, vendor=nvidia, backend=Direct3D 12, adapterType=Discrete GPU`.

## App-aligned narrator benchmark

Prompt: tracked `benchmark.narrator.v1.txt`; SHA-256 `de63332fe826bdaa0ca030662f01f150799186b04218aecd8cd966e3a1abbd38`. Runtime: LiteRT-LM Python conversation API 0.19.2. Context 4096, output ceiling 500, thinking off, top-k 64, top-p 0.95, temperature 1.0. Input 453 tokens. Three conversations per mode; all emitted >=200 tokens.

| Backend | MTP | Decode tok/s (3 runs) | Median tok/s | TTFT (3 runs, seconds) | Median TTFT | Output tokens |
|---|---|---:|---:|---:|---:|---:|
| CPU | Off | 14.56, 14.70, 14.59 | 14.59 | 2.105, 2.068, 2.476 | 2.105 | 411 each |
| CPU | On | 18.96, 18.63, 18.90 | 18.90 | 2.108, 1.964, 1.973 | 1.973 | 379 each |
| RTX 4090 / WebGPU-D3D12 | Off | 107.54, 110.91, 111.53 | 110.91 | 0.765, 0.120, 0.120 | 0.120 | 371 each |
| RTX 4090 / WebGPU-D3D12 | On | 110.74, 151.62, 153.91 | 151.62 | 1.205, 0.119, 0.117 | 0.119 | 414 each |

MTP median decode speedup: CPU 1.30x (+29.6%); GPU 1.37x (+36.7%). The two warm GPU repeats were 110.91 / 111.53 tok/s with MTP off and 151.62 / 153.91 with MTP on (1.37x comparing warm medians). MTP drafter success rates in native runtime logs were 47.6% on CPU and 46.4% on GPU. The first run in each mode had cold-start TTFT and is retained in the table. Output lengths differ because model generation ends naturally; all runs pass the benchmark's 200-token validity floor.

## Obstacles fixed in the repo

1. Trainer had a host-specific hardcoded cache; added `--model-dir` for clean disposable paths.
2. Tokenizer `apply_chat_template` returned `BatchEncoding`; prep now reads its `input_ids` field and checks the prefix before building labels.
3. A full shared export `pip freeze` included unrelated editable `litert-torch`, required unavailable `litert-converter`, and a conflicting second Torch version. Replaced it with a minimal LiteRT-LM export lock.
4. The patcher imported training-only Torch even though graph auditing can be NumPy-only. Removed that dependency from the patch script.
5. Hugging Face Xet transfer stalled partway through the large stock bundle using Python urllib. The downloader now resumes with Windows `curl.exe` and validates the final SHA.
6. A one-sample run writes `checkpoints/epoch_100`, not `epoch_100_final`; docs corrected to use the emitted checkpoint path.
7. LiteRT-LM rejected `--max-num-tokens=32` on both stock and candidate with `DYNAMIC_UPDATE_SLICE` preparation/tensor-allocation errors. Stock and candidate both generated at 1024; narrator performance uses app context 4096.
8. Benchmark runner now creates its cache directory before engine construction.

## Preserved evidence

The disposable run tree retains manifests, round-trip audit, adapter metrics/checkpoint, merge and patch manifests, package manifest/artifact, stock bundle, CPU/GPU smoke logs, and benchmark JSON under `D:\Forge-GM\experiment\output\DISPOSABLE_FORGE_E4B_P_REPRO_20261124\run`. The full model inputs, unpacked stock sections, two fresh virtual environments, and engine cache also remain under that clearly marked disposable root; they were retained so the package can be inspected and the run audited. No pre-existing shared environment or model was modified.
