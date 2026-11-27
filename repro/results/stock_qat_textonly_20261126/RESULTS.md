# Stock QAT text-only package results

Artifact: `Forge-E4B-Stock-QAT-TextOnly.litertlm` (3,150,102,528 bytes; SHA-256 `8a9f67c07c43a9fb16945dbae90a92e9a35e5b3bfef50686ddc42a3c8094b759`). The binary is not distributed through Git; reproduce it using the [packaging guide](../../../docs/TEXT_ONLY_PACKAGING_20261126.md).

Source pins: Google mobile-QAT `8e3ac78f1a4a97c19f5f194ab362a2206f71d961`, source safetensors SHA-256 `391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4`; official LiteRT-LM `f254b5ba6e36f7d89cc7892e268e9877bc20f0a0`, package SHA-256 `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.

The text-only package physically removes the audio and vision model sections and linked weights, retains the shared text embedders, decoder, tokenizer and MTP drafter. Source-to-graph audit verifies 345 unique retained quantized source tensors exactly: INT2 3, INT4 258, INT8 84. The 36 additional INT4 K/V tensors in shared-KV layers 24–41 do not exist as separate weights in the official exported graph. The 613 source tensors versus 654 provenance rows are reconciled by the single per-layer embedding tensor represented as 42 graph slabs.

| Device | MTP off median | MTP on median | Decode gain |
| --- | ---: | ---: | ---: |
| RTX 4090, official | 110.26 tok/s | 153.27 tok/s | 39.0% |
| RTX 4090, text-only | 111.61 tok/s | 144.32 tok/s | 29.3% |
| Pixel 10 Pro, official | 7.763 tok/s | 13.345 tok/s | 71.9% |
| Pixel 10 Pro, text-only | 7.641 tok/s | 13.257 tok/s | 73.5% |

Host tests used the shared LiteRT-LM 0.19.2 runtime and 3 repeated narrator generations per mode (4096 context, 453-token prompt, 500-token ceiling). Pixel tests used probe app v113 / LiteRT-LM 0.19.1 / LiteRT 2.3.2, 441-token outputs, dynamic 4096 maximum context and 1024 requested context. Full filtered measurements are in the adjacent JSON files; per-tensor source proof and packed round-trip details are in `source_tensor_audit.json` and `source_weight_round_trip.json`.

Reproduction uses Python 3.11.9 / LiteRT-LM 0.19.2. Follow the [portable commands and pinned inputs](../../../docs/TEXT_ONLY_PACKAGING_20261126.md#reproduction-commands) for the repack and both audits. The fresh rebuild reproduced the physical text-section hashes, not the full container SHA.

## Final container and reproduction audit

`container_equivalence.json` compares the final text-only package directly with the official package. It verifies the retained TOML section metadata, tokenizer, text metadata (apart from the added minimum runtime version), all four retained logical graphs, every tensor's structure and quantization parameters, and every logical tensor payload. The comparison follows external-buffer references and compensates for payload externalization and deduplication. All retained payload fingerprints match, including the MTP drafter; six audio/vision graphs are the only removed graph types.

A clean rebuild from the pinned official unpacked inputs is recorded in `fresh_rebuild_manifest.json`. The rebuilt package has SHA-256 `cf8398a857d7c22fa9e120c3fdac4e745171bf6c90eaeefc5c2a3f4946c52858` versus candidate `8a9f67c07c43a9fb16945dbae90a92e9a35e5b3bfef50686ddc42a3c8094b759`. The four logical graphs and all physical text sections are equivalent; full container hashes differ because the builder regenerates system metadata identity/timestamp.

The prior short-context MTP issue was reproduced with the official and text-only packages using LiteRT-LM 0.19.2, CPU backend, greedy sampling (`top_k=1`, `temperature=0`), and prompt `Hi`. At KV-cache capacities 32 and 128, both packages fail engine creation at `DYNAMIC_UPDATE_SLICE`; both initialize and generate at 1024. See `short_context_mtp.json`. This establishes a runtime/cache-capacity failure shared by the official package.

## Greedy and Pixel 4096-context validation

The paired greedy comparison ran all 12 frozen cases on the CPU backend with LiteRT-LM 0.19.2, MTP off, 4096-token KV-cache capacity, and a 128-token output ceiling. All 12 decoded outputs matched byte-for-byte; re-tokenizing each output with the package SentencePiece tokenizer also produced matching ID sequences, with no first divergence in the compared output. Per-run output hashes and retokenized-ID hashes are in `greedy_panel.json`. These are re-tokenized IDs, not the original sampled stream: the Session API exposes decoded text and token counts, and the installed runtime's token-trace debugger is disabled. Per-step logits were not accessible. The 128-token ceiling limits this comparison to the first part of each response.

The earlier v113 Pixel long-prompt runs used `requestedContext=1024`; they are retained in `pixel_long_context_dynamic.json` as a dynamic-growth check. The acceptance run used the isolated v116 probe, whose GPU narrator configuration requests 4096 tokens. All four package/MTP combinations completed on GPU with the response profile confirming `requestedContext=4096`, `maxContext=4096`, and the expected package SHA and effective MTP state. Each processed 3499 input tokens and generated 500 tokens (502 with MTP), using the same long prompt and greedy sampling (`top_k=1`, `top_p=1`, temperature 0.01; the app API rejects temperature zero). Run-level timings and rates are in `pixel_memory_4096_matched.json`.

For each run, `dumpsys meminfo` sampled process `TOTAL PSS` and `GL mtrack` about every 1.2 seconds over a complete 127.5–129.0 second window, including the generation. The observed PSS / GL mtrack peaks in MiB were official MTP-off 4074.8 / 3128.4, official MTP-on 4172.1 / 3187.1, text-only MTP-off 4146.6 / 3130.8, and text-only MTP-on 5049.5 / 3716.0. Exact KiB peaks, sample counts, per-run profiles, limitations, and four raw traces are in `pixel_memory_4096_matched.json` and the adjacent `pixel_memory_4096_*.csv` files. These are sampled process and graphics-accounting maxima, not allocator instrumentation; PSS and GL mtrack are separate views and must not be summed. One run per combination was measured, so the values are comparable observations rather than statistical distributions.

## Limitations

The v113 dynamic-growth and partial memory traces are historical evidence, superseded for strict 4096-context validation by the v116 matched runs. The text-only MTP-on observation is 21.0% higher in peak PSS and 22.5% lower in decode throughput than official at requested context 4096 (one run per condition). Repeat under matched process, cache and thermal conditions before attributing a regression to the package. Peak host VRAM, native token IDs, logits, and GPU/MTP-on paired greedy output parity were not measured; broad quality evaluation was outside the completed scope. The model binary remains local and is not included in Git.
