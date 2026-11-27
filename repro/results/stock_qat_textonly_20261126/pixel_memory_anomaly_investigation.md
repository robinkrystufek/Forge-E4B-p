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
- Exact probe-specific `ForgeBridge.kt` change between the dirty v113 source used for the probe and v116 was:

  ```diff
  - val narratorContextTokens = if (BuildConfig.APPLICATION_ID.endsWith(".mtpprobe") && chosen == "gpu") 1024 else 4096
  + val narratorContextTokens = 4096
  - val mtpProbeGpu = mode == "gpu" && BuildConfig.APPLICATION_ID.endsWith(".mtpprobe")
  - val limit = if (mode == "router" || mtpProbeGpu) 1024 else 4096
  + val limit = if (mode == "router") 1024 else 4096
  ```

  This means v113 exposed/requested 1024 for the GPU probe; v116 requests 4096. The complete normalized source comparison (6,271,409 bytes before gzip; SHA-256 `67D60E0FFA98FE7B1BF6A778208B3BD067A95878CDB6CF609F2C284236918F31`; `src/`, Android app main sources/resources, Prompt Lab bridge, and `package.json`) is archived in [probe_app_v113_to_v116.diff.gz](probe_app_v113_to_v116.diff.gz). It includes the app changes around model-bundle storage/registry and Prompt Lab; the GPU runtime Kotlin source itself did not change. No model package bytes are included.
- The first v116 build was the regular `app.forgegm.trial` application ID and was rejected by Android with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`; it was not installed. The isolated probe APK was then built with `npm run android` followed by `gradlew :app:assembleDebug -PforgeMtpProbe=true`, and installed successfully as `app.forgegm.trial.mtpprobe`. Build APK and pulled installed `base.apk` both hash to `791DD9847AB2FABC9C508699F8A763817C24FE01B8D2F5B42ADB53B37480E8C9`. Device package state reports update time 2026-11-26 20:18:57 and first install 2026-11-20 14:46:04. Both APKs use the same debug signing certificate SHA-256 `9d8bd0ef4baa57a292123bc8a2c73c4c7ba67bb47b1a345cf46f231ae6a73505`.
- All four memory samples were taken after this v116 install and used that identical APK hash. The probe's single process remained PID 28082 throughout the four configurations. Its later death at about 20:51 followed explicit task removal, after all traces.

## Engine, cache, and app activity

Persisted v116 diagnostics record engine initializations at 20:19:53.923 (official/off, cache absent), 20:23:38.592 (official/on, absent), 20:32:44.657 (text-only/off, absent), and 20:35:33.979 (text-only/on, **cache present**). The text-only package was imported/initialized at 20:31:22.287 with MTP on and cache absent; that run created the cache subsequently observed by the anomalous MTP-on run. Thus the anomalous case alone entered inference with a warm compiled cache, whereas the other three began with no cache artifacts. Cache presence is the persisted diagnostic boolean; the old trace does not preserve a complete cache inventory or content hash.

The MTP setter closes and reopens the engine. The package import path also resets the engine. These are in-process engine lifecycle changes, not process restarts. Prompt Lab used `stream=false`, and inference results were not added to the diagnostics benchmark history; diagnostics retain initialization/import profiles but not per-request lifecycle events. No timestamped evidence shows WebView navigation or UI actions during generation; the same app/WebView remained active. The old artifacts do not include per-process WebView/native allocation attribution, so WebView or another native runtime allocation cannot be ruled in or out.

Android event logs around 20:35:57–20:36:18 also show a system-wide background-process unfreeze/freeze wave. The probe PID is not named as a frozen process. This is an environmental event concurrent with the GL rise and should be noted as a possible measurement confound, not attributed to the model. The app remained the active workload and no other deliberate app activity was recorded.

## Controlled MTP-on repeat

After the provenance/cache review, the repeat used the same installed v116 probe APK and alternated package order as official → text-only → text-only → official. Each measured run force-stopped the app, deleted only the selected package's generated MTP-on cache directory, started a new app process, verified the selected package SHA, configured MTP on, loaded the model, then generated with the same long prompt. The runtime initialization profile reported `cacheArtifactsPresentBeforeInit=false` in all four runs. The APK hash was `791DD9847AB2FABC9C508699F8A763817C24FE01B8D2F5B42ADB53B37480E8C9` throughout. The two bundles were checked by SHA-256 before transfer/import; neither package was modified or rebuilt.

| Order | Package | PID | Request → response, local time | API TTFT / total (ms) | PSS peak during request (KiB) | GL mtrack peak (KiB) | Native heap peak during request (KiB) | Cache before init |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 1 | Official | 12172 | 21:25:53.809 → 21:26:32.622 | 11,842 / 38,772 | 4,257,104 | 3,262,992 | 576,036 | absent |
| 2 | Text-only | 14889 | 21:30:32 → 21:31:12 | 11,898 / 39,297 | 4,248,107 | 3,262,992 | 576,268 | absent |
| 3 | Text-only | 17002 | 21:33:52 → 21:34:32 | 11,607 / 39,300 | 4,310,534 | 3,262,996 | 576,580 | absent |
| 4 | Official | 19559 | 21:38:24 → 21:39:03 | 11,916 / 39,323 | 4,246,142 | 3,262,992 | 572,668 | absent |

All runs processed 3,499 input tokens and returned 502 tokens (MTP draft tokens make the output exceed the 500-token request cap slightly). The API profile confirmed requested/max context 4096, dynamic context, GPU, model-default precision, and MTP enabled. Each 180-second memory window began before engine load and continued well after generation; it contains 135–137 samples, with 29–30 during the host-observed inference request. Timestamped request-send/response and load events, profile details, per-sample PID, PSS, GL mtrack, and native heap are in `pixel_memory_4096_mtp_on_repeat.json` and the four adjacent `pixel_memory_4096_mtp_on_repeat_*.csv` files.

The two-run mean inference-window PSS peak was 4,251,623 KiB (4.05 GiB) for official and 4,279,321 KiB (4.08 GiB) for text-only, a 27,698 KiB (27.0 MiB, 0.65%) observed difference. Mean GL mtrack peaks differed by only 2 KiB. The highest repeated text-only PSS peak was 4,310,534 KiB (4.11 GiB), 860,135 KiB (0.82 GiB) below the original anomalous 5,170,669 KiB (4.93 GiB) peak. During inference, native heap peaks were 572,668–576,580 KiB and end-of-window values were 556,680–558,716 KiB; the original trace ended at 1,442,180 KiB. The anomalous sustained growth did not recur under the clean-process/cold-cache conditions.

This is evidence against attributing the original anomaly to the text-only package alone. It is consistent with an allocation/retention event tied to process/runtime state; the warm compiled cache was a confound unique to the original text-only MTP-on run, but these data do not prove that the cache caused the event. With two runs per package and observed text-only PSS variability of about 61 MiB, the small mean difference is descriptive, not a statistical claim. The `Native Heap` value is a separate `dumpsys meminfo` accounting field; it must not be added to PSS or GL mtrack.

Raw inputs are the four `pixel_memory_4096_*_*.csv` and matching JSON files in this directory, plus scratch copies `matched_*` and scripts under `D:\Forge-GM\experiment\output\stock-qat-textonly-validation-20261126`.
