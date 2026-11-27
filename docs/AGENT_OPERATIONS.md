# Agent operations

## LiteRT-LM stock-QAT text-only packaging (verified 2026-11-26)

- Source checkout: `D:\Forge-GM\experiment\Forge-E4B-p`; source branch `feat/campaign16-multiepoch` had unrelated Campaign16 work. Task changes are being ported to dedicated `feat/stock-qat-textonly` worktree based on `origin/main`.
- Reusable runtime: `D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv`, Python 3.11.9, LiteRT-LM/builder 0.19.2. Reused; recheck versions before future use.
- Source pin `google/gemma-4-E4B-it-qat-mobile-transformers@8e3ac78f1a4a97c19f5f194ab362a2206f71d961`; official package pin `litert-community/gemma-4-E4B-it-litert-lm@f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`. Source/package hashes and output hash are in `docs/TEXT_ONLY_PACKAGING_20261126.md`.
- Repack script: `repro/scripts/repack_official_litertlm.py --text-only`. Removes all six audio/vision section records and linked external weights; keeps tokenizer, shared text embedders, decoder and MTP drafter. Model SHA-256 `8a9f67c07c43a9fb16945dbae90a92e9a35e5b3bfef50686ddc42a3c8094b759`, local output on D:.
- Source tensor graph audit: `repro/scripts/audit_stock_qat_source.py` matched 388 graph mappings to 345 unique retained quantized tensors (INT2 3, INT4 258, INT8 84) across official decoder/embedder/per-layer embedder. The source round-trip audit checked 613 packed tensor entries (111/306/196 by bit width); 654 provenance rows are explained by 42 per-layer embedding slabs replacing one tensor (+41 records). 36 shared-KV K/V tensors in layers24–41 have no separate payloads in the official graph.
- Host verification: CPU and NVIDIA RTX4090 GPU (LiteRT WebGPU/Dawn over D3D12) official and candidate smoke; MTP CPU/GPU checks; 3 long benchmark runs per package/mode. Medians in `docs/TEXT_ONLY_PACKAGING_20261126.md` and results JSON.
- Finnl1 ADB: `D:\Gemma4-MTP-work\host-android-toolchain\sdk\platform-tools\adb.exe`, v37.0.3; wireless Pixel10Pro `192.168.4.20:45951`, Android17 `blazer`. Probe app `app.forgegm.trial.mtpprobe` v113, LiteRT-LM0.19.1/LiteRT2.3.2. Official baseline 3 runs per mode; candidate 3 MTP-on and 4 MTP-off, 441 output tokens/run. Official median decode 7.763 tok/s off /13.345 on; candidate 7.641 off /13.257 on. Metrics-only JSON in `repro/results/stock_qat_textonly_20261126/`. Recheck wireless endpoint before future sessions.
- At GPU benchmark gate, free VRAM was 23,394 MiB, utilization 0–1%, no compute PIDs; estimated 20 GiB peak +2 GiB reserve fit. This is pre-run availability only, not measured peak allocation. A 128-token MTP smoke failed with `DYNAMIC_UPDATE_SLICE`; 1024-token MTP checks completed.
- Limits: peak host/device memory not sampled during benchmarks; broad logit/quality suite not run. No model-binary publishing target found in project; binary intentionally remains local.

## Reusable environment and artifact locations

| Absolute path | Purpose | Runtime / owner | Status |
|---|---|---|---|
| `D:\Forge-GM\experiment\mobile_qat_feasibility\litert_export_venv` | LiteRT-LM export/repack/inference | Python3.11.9, LiteRT-LM0.19.2; shared | Reused, verified 2026-11-26 |
| `D:\Forge-GM\experiment\output\DISPOSABLE_FORGE_E4B_P_REPRO_20261124` | Existing pinned source/package, environments and caches | Existing shared task reproduction | Retained; do not delete without owner confirmation |

Confidence: paths and versions were checked 2026-11-26. Device connection, GPU load and free space are transient and must be rechecked before reuse.
