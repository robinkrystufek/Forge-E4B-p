# Pixel MTP-on memory trace investigation

Date: 2026-11-26 (device timestamps are America/Los_Angeles, UTC-08:00)

This investigation precedes any repeat benchmark. Neither model package was modified or rebuilt.

## Finding

The text-only MTP-on trace is not evidence of a larger initial model footprint. Its PSS began at 4,218,459 KiB, peaked at 5,170,669 KiB (20:36:27.821), and ended at 5,133,557 KiB. GL mtrack began at 3,252,724 KiB, peaked at 3,805,156 KiB (20:36:17.356), and ended at 3,266,136 KiB, only 13,412 KiB above its initial value. The fourth CSV field is `Native Heap` size: it began at 558,164 KiB, peaked at 1,448,704 KiB at 20:36:55.947, and ended at 1,442,180 KiB. Its retained growth (+884,016 KiB from first to last sample) closely tracks the persistent PSS increase. This is consistent with a native-heap allocation/retention event during inference; it does not identify which runtime component allocated it.

## The four original traces

All four were captured with the same v116 APK, Pixel 10 Pro / Android 17, GPU backend, context 4096, 3,499 input tokens, about 500 generated tokens, and the same greedy-like sampling profile (top-k 1, top-p 1, temperature 0.01). They were sequential runs in the same Android process (PID 28082), not clean-process comparisons. There was no process death/start between them. Each MTP toggle tears down and reopens the engine inside that same process.

| Package / MTP | Request interval (host-derived, local time) | Sample window | PSS first / peak / last (KiB) | GL first / peak / last (KiB) | Native heap first / peak / last (KiB) | Cache before engine init |
|---|---|---|---:|---:|---:|---|
| Official / off | ~20:20:55.570–20:22:12.714 | 20:20:59.112–20:23:06.623; 100 samples | 4,133,703 / 4,172,586 / 4,146,043 | 3,194,772 / 3,203,528 / 3,178,896 | 544,168 / 547,056 / 546,320 | absent |
| Official / on | ~20:24:22.301–20:25:02.254 | 20:24:25.095–20:26:33.680; 100 samples | 4,250,380 / 4,272,228 / 4,246,001 | 3,254,852 / 3,263,612 / 3,238,980 | 463,096 / 464,460 / 463,012 | absent |
| Text-only / off | ~20:33:05.374–20:34:24.003 | 20:33:08.624–20:35:16.535; 100 samples | 4,219,118 / 4,246,146 / 4,220,947 | 3,197,224 / 3,205,976 / 3,181,344 | 567,204 / 567,204 / 557,684 | absent |
| Text-only / on | ~20:35:53.961–20:36:41.488 | 20:35:56.211–20:38:05.211; 105 samples | 4,218,459 / 5,170,669 / 5,133,557 | 3,252,724 / 3,805,156 / 3,266,136 | 558,164 / 1,448,704 / 1,442,180 | **present** |

The request starts are reconstructed as response JSON file-write time minus the script's elapsed HTTP-call time; response times are file-write timestamps and may trail actual response receipt by a small amount. The old script did not save response `metrics.ttftMs` or `metrics.totalMs`, so exact model-side generation start/end cannot be recovered from surviving artifacts. The request window is an approximation, not a timestamped inference lifecycle record. The anomalous PSS and GL maxima occurred during this request window; native heap reached its maximum about 14 seconds after the response file was written. Sampling continued roughly 80 seconds after response, so the retained end values are post-generation.

The sampling script wrote four fields as `timestamp_ms,total_pss_kib,gl_mtrack_kib,native_heap_kib`, sourced from `dumpsys meminfo`'s `TOTAL PSS`, `GL mtrack`, and `Native Heap` rows. Samples were taken about every 0.75 seconds by the script; observed mean spacing was 1.24–1.30 seconds due to ADB/dumpsys overhead.

## APK/source provenance

- v113 pulled from the device before update: SHA-256 `2EA800E033C75D8B284EC7886B00BE0F4636F7D74B48C388196FA302C649C342`; package `app.forgegm.trial.mtpprobe`, version code 113, version name `1.0-v113-mtp-probe`. The APK's local modification time is the pull time, not a proven build time.
- The exact available v113 source snapshot is `D:\Forge-GM\experiment\Forge-GM-r14d-full-20261122_141602`, HEAD `b8b72577953491594d46f58d2d5b871da1489fbe`, package version 113, plus an uncommitted six-line `ForgeBridge.kt` probe patch. That patch hard-coded narrator GPU probe context to 1024 and limited the probe to 1024. It is the material v113 probe behavior; the clean historical commit `0a2ee3107c315f430c8c03bae7c618ee1236633f` alone does not represent the installed v113 APK's probe source.
- v116 source snapshot: `D:\Forge-GM\experiment\Forge-GM-campaign16-source-20261124`, HEAD `c601048065d8443e6651d28123b2a67cf6c0c604`, package version 116. Its GPU probe uses context 4096 without the v113 probe-only 1024 override. A no-index source comparison showed 62 `src` files changed (+4,738/-1,061); seven Android app Java/Kotlin files changed (+521/-90), two Prompt Lab bridge files changed (+149/-8), and `android/gpu-runtime/src/main/kotlin` had no source changes. Package version changed 113→116. The persisted JNI runtime SHA was the same (`1b2a020e948136e5560703ab85600d0b6bc39e34223f6c0fa6fb30d454316bd5`).
- The first v116 build was the regular `app.forgegm.trial` application ID and was rejected by Android with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`; it was not installed. The isolated probe APK was then built with `npm run android` followed by `gradlew :app:assembleDebug -PforgeMtpProbe=true`, and installed successfully as `app.forgegm.trial.mtpprobe`. Build APK and pulled installed `base.apk` both hash to `791DD9847AB2FABC9C508699F8A763817C24FE01B8D2F5B42ADB53B37480E8C9`. Device package state reports update time 2026-11-26 20:18:57 and first install 2026-11-20 14:46:04. Both APKs use the same debug signing certificate SHA-256 `9d8bd0ef4baa57a292123bc8a2c73c4c7ba67bb47b1a345cf46f231ae6a73505`.
- All four memory samples were taken after this v116 install and used that identical APK hash. The probe's single process remained PID 28082 throughout the four configurations. Its later death at about 20:51 followed explicit task removal, after all traces.

## Engine, cache, and app activity

Persisted v116 diagnostics record engine initializations at 20:19:53.923 (official/off, cache absent), 20:23:38.592 (official/on, absent), 20:32:44.657 (text-only/off, absent), and 20:35:33.979 (text-only/on, **cache present**). The text-only package was imported/initialized at 20:31:22.287 with MTP on and cache absent; that run created the cache subsequently observed by the anomalous MTP-on run. Thus the anomalous case alone entered inference with a warm compiled cache, whereas the other three began with no cache artifacts. Cache presence is the persisted diagnostic boolean; the old trace does not preserve a complete cache inventory or content hash.

The MTP setter closes and reopens the engine. The package import path also resets the engine. These are in-process engine lifecycle changes, not process restarts. Prompt Lab used `stream=false`, and inference results were not added to the diagnostics benchmark history; diagnostics retain initialization/import profiles but not per-request lifecycle events. No timestamped evidence shows WebView navigation or UI actions during generation; the same app/WebView remained active. The old artifacts do not include per-process WebView/native allocation attribution, so WebView or another native runtime allocation cannot be ruled in or out.

Android event logs around 20:35:57–20:36:18 also show a system-wide background-process unfreeze/freeze wave. The probe PID is not named as a frozen process. This is an environmental event concurrent with the GL rise and should be noted as a possible measurement confound, not attributed to the model. The app remained the active workload and no other deliberate app activity was recorded.

## Next matched repeat

The repeat must use the same installed v116 APK for both bundles, clean app processes per measured run, identical context/prompt/output/sampling, matched MTP-on compiled-cache state, and alternating package order. For every run capture PID, package SHA, process start time, cache state at engine init, request-send and response timestamps, API `ttftMs` and `totalMs`, and a memory trace including PID. Sample a complete pre-request baseline through generation and at least one minute post-response. Do not delete or alter package bundles. The previous samples were not rerun because the device is presently absent from `adb devices`; after device reconnection this is the remaining experiment, rather than grounds for another model-package claim.

Raw inputs are the four `pixel_memory_4096_*_*.csv` and matching JSON files in this directory, plus scratch copies `matched_*` and scripts under `D:\Forge-GM\experiment\output\stock-qat-textonly-validation-20261126`.
