# Forge-E4B-p

**Gemma 4 E4B mobile QAT → mixed-bit LiteRT-LM with MTP**

A reproducible experimental project exploring fixed-scale LoRA adaptation and text-only deployment of Google's Gemma 4 E4B mobile-QAT model while preserving the original mixed INT2/INT4/INT8 representation and official multi-token prediction (MTP) drafter.

[Workflow](#workflow) · [Validation](#validation) · [Benchmarks](#mtp-performance) · [Reproduce](#reproduce-it) · [Repository layout](#repository-layout)

---

| **3.49 GB** | **3.15 GB** | **INT2 / INT4 / INT8** | **1.74×** |
| :---: | :---: | :---: | :---: |
| Trial 1 LoRA POC | Trial 2 text-only stock | Original mixed precision | Trial 2 Pixel median MTP speedup |

> **Scope:** Two independent packaging trials. Trial 1 is a one-update synthetic LoRA proof of concept; Trial 2 removes multimodal towers from the untrained official package. Neither is a model-quality or generalization claim.

## Experimental showcase

| Trial | Intervention | Result | Evidence |
| :--- | :--- | :--- | :--- |
| **1 · Fixed-scale LoRA POC** (Nov 24) | One synthetic optimizer update; patch original packed graph | 127 INT4 values changed; 3.49 GB package; MTP retained | [Reproduction](repro/results/POC_REPRODUCTION_20261124.md) |
| **2 · Removal of vision and audio towers** (Nov 26) | No training or weight replacement; remove six multimodal graph sections | 3.15 GB package (13.92% smaller than official); all four retained logical graphs and payloads match | [Method](docs/TEXT_ONLY_PACKAGING_20261126.md) · [Results](repro/results/stock_qat_textonly_20261126/RESULTS.md) |

Trial 2 uses the official stock package, **not** the Trial 1 LoRA artifact. The interventions are independent.

## Workflow

The adaptation retains the original quantization grid. Only audited packed weight payloads in the official LiteRT-LM graph are patched; package metadata and the MTP drafter are preserved.

![Five-step workflow: inspect pinned mixed-bit weights, prepare masked chat examples, adapt and merge at fixed scales, patch and repack the official graph while preserving MTP, and validate CPU/GPU inference and narrator throughput.](docs/assets/workflow.svg)

The workflow supports exact source-weight round-tripping, masked chat preparation, fixed-scale LoRA merging, audited graph patching and inference validation without converting the source checkpoint to uniform precision.

## Validation

The [clean reproduction record](repro/results/POC_REPRODUCTION_20261124.md) documents a run from a fresh checkout on November 24, 2026.

| Check | Observed result |
| :--- | :--- |
| Source round-trip | Bit-exact across all 613 packed weight tensors |
| Synthetic LoRA update | One optimizer update; finite loss and nonzero adapter gradients |
| Fixed-scale merge | 127 INT4 values changed in one tensor; INT2 and INT8 unchanged |
| Graph patch | Packed merge hashes match; bytes outside target payloads unchanged |
| LiteRT-LM package | 20 sections, 3,492,003,840 bytes; original config, tokenizer and MTP drafter retained |
| Functional inference | CPU and RTX 4090 GPU smoke tests generated the expected response |

## MTP performance

### Trial 1 · Fixed-scale LoRA proof of concept

![Decode throughput with MTP off and on for all five benchmark configurations: PC CPU stock and LoRA POC, RTX 4090 stock and LoRA POC, and Pixel 10 Pro GPU LoRA POC. PC medians and Pixel means are distinguished; N is shown per mode. All bars share a zero-based linear scale and error bars show observed min–max ranges.](docs/assets/mtp-throughput.svg)

| Backend | Package | Statistic | N per mode | MTP off (tok/s) | MTP on (tok/s) | Speedup | TTFT off (s) | TTFT on (s) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PC CPU | Official stock | Median | 3 | 13.73 (12.88–13.86) | 17.45 (17.16–17.46) | **1.27×** (+27.1%) | 2.16 (2.14–2.67) | 2.01 (2.00–2.16) |
| PC CPU | LoRA POC | Median | 3 | 14.59 (14.56–14.70) | 18.90 (18.63–18.96) | **1.30×** (+29.6%) | 2.105 (2.068–2.476) | 1.973 (1.964–2.108) |
| RTX 4090 | Official stock | Median | 3 | 111.45 (108.41–111.58) | 154.48 (100.92–154.91) | **1.39×** (+38.6%) | 0.12 (0.12–0.84) | 0.12 (0.12–1.47) |
| RTX 4090 | LoRA POC | Median | 3 | 110.91 (107.54–111.53) | 151.62 (110.74–153.91) | **1.37×** (+36.7%) | 0.120 (0.120–0.765) | 0.119 (0.117–1.205) |
| Pixel 10 Pro GPU | LoRA POC | Mean | 4 | 7.7 (7.5–7.8) | 12.8 (12.7–13.0) | **1.66×** (+66.2%) | 2.85 (2.52–3.03) | 2.77 (2.42–2.98) |

Values are **median (min–max)** of valid generations for the PC configurations. **Pixel values are mean (min–max):** individual run measurements are not included in the reproduction record, so exact medians cannot be reported. N is the number of generations per MTP mode (32 generations total across all configurations). Speedup is the ratio of the reported central statistics; the table ranges and chart whiskers are observed extrema, not confidence intervals or standard deviations.

**Benchmark protocol.** All runs used the registered Forge-GM `benchmark.narrator` v1 fixture, a 4,096-token context, 500-token output ceiling and disabled thinking; every generation passed the 200-token validity floor. TTFT denotes time to first token.

- **PC:** LiteRT-LM 0.19.2, CPU (XNNPACK for the stock baseline) and RTX 4090 (WebGPU/Dawn over Direct3D 12), three runs per package/backend/MTP mode. The explicit sampler used top-k 64, top-p 0.95 and temperature 1.0. Cold-start samples remain included in the medians. Stock drafter success was 41.9% on CPU and 47.6% on GPU; POC drafter success was 47.6% and 46.4%, respectively.
- **Pixel:** Android 17, Forge GM 1.0-v116, LiteRT-LM 0.19.1 / LiteRT 2.3.2, GPU with model-default precision and sampler, four valid serial runs per mode. The transferred POC package was verified by SHA-256 and the app reported MTP effective off/on as configured. Mean generation time decreased from 59.88 s (59.08–61.49) to 37.09 s (36.26–37.74), a **38.1% reduction**, with 440 → 441 output tokens. Total time excludes engine/model initialization.

These are separate within-configuration MTP comparisons. PC results use medians and Pixel results use means; the differing runtimes and samplers preclude a controlled cross-device comparison. Output lengths vary with natural generation endings. No Pixel CPU or stock Pixel results were collected in Trial 1, and these measurements do not establish model quality.

**Full evidence:** [LoRA POC PC](repro/results/POC_REPRODUCTION_20261124.md#app-aligned-narrator-benchmark) · [Official stock PC](repro/results/POC_REPRODUCTION_20261124.md#stock-official-e4b-pc-baseline) · [Pixel GPU](repro/results/POC_REPRODUCTION_20261124.md#pixel-10-pro-gpu-mtp-benchmark) · [Pixel engine and metric collection](docs/MOBILE_QAT_LORA_TO_MIXED_BIT_LITERT_LM.md#6b-pixel-10-pro-gpu-mtp-benchmark). The records retain package hashes, per-run values or available ranges, and exact settings.

### Trial 2 · Removal of vision and audio towers

![Trial 2 MTP throughput: official stock and text-only stock, RTX 4090 and Pixel 10 Pro GPU; medians and observed ranges with sample counts, separate device axes.](docs/assets/mtp-throughput-textonly.svg)

| Backend | Package | N off/on | MTP off (tok/s) | MTP on (tok/s) | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: |
| RTX 4090 GPU | Official stock | 3/3 | 110.26 (107.53–111.56) | 153.27 (99.18–156.19) | **1.39×** |
| RTX 4090 GPU | Text-only stock | 3/3 | 111.61 (106.96–111.68) | 144.32 (109.98–156.21) | **1.29×** |
| Pixel 10 Pro GPU | Official stock | 3/3 | 7.76 (6.45–7.80) | 13.35 (13.11–13.43) | **1.72×** |
| Pixel 10 Pro GPU | Text-only stock | 4/5 | 7.64 (7.55–7.65) | 13.26 (12.94–13.29) | **1.74×** |

All values are median (observed min–max); whiskers are not uncertainty intervals. RTX 4090: LiteRT-LM 0.19.2, 453 input tokens, 4096 context, top-k 64/top-p 0.95/temperature 1. Pixel: probe v113, LiteRT-LM 0.19.1 / LiteRT 2.3.2, GPU/model-default precision, 458 input tokens, 1024 requested context with dynamic maximum 4096, 441 generated tokens. Both use the narrator fixture and a 500-token output ceiling. Trial 1 and Trial 2 were separate sessions with different Pixel profiles and summary statistics; compare MTP modes within each configuration, not the trials as a controlled head-to-head experiment.

**Packaging result:** 3,659,530,240 → 3,150,102,528 bytes (−13.92%); the decrease includes removal of multimodal data and builder deduplication of shared embedding data. 345 unique retained source quantized tensors and all four final logical text/MTP graphs match. The CPU greedy panel (MTP off, up to 128 generated tokens) matched decoded text in 12/12 cases. Native token IDs and logits were unavailable.

**Long-context validation:** both packages completed GPU inference at requested/max context 4096 with MTP off/on. An initial text-only MTP-on memory/throughput anomaly did not recur in a counterbalanced repeat using fresh app processes and absent compiled caches (two runs per package). Mean inference-window peak PSS was 4,152 vs 4,179 MiB (official vs text-only; +0.65%), and decode throughput was 18.45 vs 18.20 tok/s (−1.35%). These are descriptive observations, not a formal equivalence test or evidence of runtime memory savings. See [matched repeat and original anomaly](repro/results/stock_qat_textonly_20261126/RESULTS.md#matched-mtp-on-memory-repeat).

## Reproduce it

**Trial 2:** [Text-only packaging and reproduction](docs/TEXT_ONLY_PACKAGING_20261126.md#reproduction-commands) (no training required).

**Trial 1:** [Clean proof-of-concept reproduction instructions](docs/MOBILE_QAT_LORA_TO_MIXED_BIT_LITERT_LM.md#clean-proof-of-concept-reproduction-repository-tracked-inputs)

The implementation guide also documents the original experiment, quantization invariants, pinned source and stock bundle revisions, label masking, training protocol, GPU requirements and patch/repack steps. Use the clean-reproduction section for repository-tracked inputs rather than the historical workstation-specific paths.

| Environment | Tested configuration |
| :--- | :--- |
| Training | Python 3.11, pinned PyTorch / Transformers stack |
| Export and inference | Separate pinned environment, LiteRT-LM and builder 0.19.2 |
| Host | Windows, NVIDIA RTX 4090; inference via CPU or WebGPU/Dawn over Direct3D 12 |
| Inputs | Pinned mobile-QAT checkpoint and stock LiteRT-LM bundle; external downloads |

### Repository layout

| Path | Purpose |
| :--- | :--- |
| [`docs/MOBILE_QAT_LORA_TO_MIXED_BIT_LITERT_LM.md`](docs/MOBILE_QAT_LORA_TO_MIXED_BIT_LITERT_LM.md) | Implementation guide and clean-checkout instructions |
| [`docs/TEXT_ONLY_PACKAGING_20261126.md`](docs/TEXT_ONLY_PACKAGING_20261126.md) | Stock text-only package, audits and reproduction |
| [`repro/scripts/`](repro/scripts/) | Download, inspect, prepare, train, merge, patch, package and benchmark |
| [`repro/fixtures/`](repro/fixtures/) | Synthetic chat sample, provenance mapping and exact narrator prompt |
| [`repro/requirements-training.lock`](repro/requirements-training.lock) | Pinned training dependencies |
| [`repro/requirements-export.lock`](repro/requirements-export.lock) | Pinned export dependencies |
| [`repro/results/`](repro/results/) | Reproduction evidence, hashes, audits and benchmark results |

Model downloads and environments are intentionally excluded from Git. The tracked training fixture is synthetic; use an access-controlled corpus for private data and follow the guide's split and masking requirements.

---

Upstream references: [Gemma mobile-QAT checkpoint (pinned revision)](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers/tree/8e3ac78f1a4a97c19f5f194ab362a2206f71d961) · [Stock LiteRT-LM bundle (pinned revision)](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/tree/f254b5ba6e36f7d89cc7892e268e9877bc20f0a0) · [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
