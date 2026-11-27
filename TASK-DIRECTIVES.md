# Task directives: stock-QAT text-only LiteRT-LM

Date: 2026-11-26 (America/Los_Angeles)

## User objective

Package the original Google Gemma 4 E4B mobile-QAT weights as a text-only LiteRT-LM model, preserving mixed-bit quantization, official graph behavior, tokenizer, KV cache, and MTP. Compare stock and repackaged variants, audit provenance and removed components, validate host CPU/GPU and Pixel 10 Pro GPU with MTP off/on, benchmark, and prepare a dedicated Forge-E4B-p PR. Do not train or modify source weights.

## Authoritative pins

- QAT source: `google/gemma-4-E4B-it-qat-mobile-transformers@8e3ac78f1a4a97c19f5f194ab362a2206f71d961`.
- Official container: `litert-community/gemma-4-E4B-it-litert-lm@f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`.
- Existing source SHA-256: `391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4`.
- Existing official package SHA-256: `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.

## Constraints and amendments

- Preserve all pre-existing edits in the original checkout; it was on `feat/campaign16-multiepoch` with unrelated Campaign 16 work. Prepare the PR from a dedicated branch/worktree containing only this task.
- Reuse `D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv` (Python 3.11.9, LiteRT-LM 0.19.2); no environment created.
- Starting free space: C: 127,524,777,984 bytes; D: 80,382,464,000 bytes. Do not remove existing model/output artifacts without explicit direction.
- User amendment: device availability is not an early-stage blocker; check the device once a candidate is ready. Candidate and official-device comparisons have now completed.
- Authorized wireless ADB connection was verified for Pixel testing. Discover a current endpoint locally; no private network address is required for reproduction.
- User requested long RTX4090 tests with MTP off/on in parallel and then three official Pixel runs per mode; completed and recorded in results/docs.
- Do not commit model binaries. No configured/established model publishing destination was found.

## Acceptance status

Completed: exact source-to-official retained text tensor audit (345 unique, 388 graph records); text-only physical package; host CPU and RTX4090 GPU smoke, MTP checks and repeated long-generation runs; Pixel 10 Pro GPU baseline/candidate, three official runs per mode, and candidate runs; reproducibility notes and evidence. Reused installed environment only.

Completed: Pixel 10 Pro GPU inference and full-window memory sampling at `requestedContext=4096` for both packages with MTP on/off. The v116 probe confirmed the profile, package SHA, and effective MTP mode on all four runs; each processed 3499 input tokens and generated 500/502 tokens. The controlled MTP-on memory comparison uses two runs per package with fresh processes and absent compiled caches; results and raw traces are in `repro/results/stock_qat_textonly_20261126/pixel_memory_4096_mtp_on_repeat.json` and adjacent CSVs. Other evaluation points are complete: exact container and fresh-rebuild audit; paired 12-prompt greedy comparison (decoded and re-tokenized IDs match, but native IDs/logits are not exposed by the installed LiteRT-LM API); and shared official/candidate short-context MTP failure at cache capacities 32/128 with both working at 1024. The model binary remains local.
