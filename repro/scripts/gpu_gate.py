"""Memory/utilization gate for planned GPU phases; never starts or kills workloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MIB_PER_GIB = 1024
DEFAULT_MARGIN_MIB = 2 * MIB_PER_GIB


@dataclass(frozen=True)
class Sample:
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...] = ()


_DESKTOP_PROCESSES = {
    "explorer.exe", "searchhost.exe", "startexperiencehost.exe",
    "startmenuexperiencehost.exe", "shellexperiencehost.exe", "systemsettings.exe",
    "nvidia share.exe", "phoneexperiencehost.exe", "chatgpt.exe",
    "codex-computer-use.exe", "codex-computer-use-swift.exe", "dwm.exe", "runtimebroker.exe",
    "system", "gcc.exe", "gbt_dl_lib.exe", "windowsterminal.exe", "openwith.exe",
}


def parse_gpu_csv(gpu_line: str, processes: str = "") -> Sample:
    fields = next(csv.reader([gpu_line], skipinitialspace=True))
    if len(fields) != 4:
        raise ValueError(f"expected 4 nvidia-smi GPU fields, got {len(fields)}")
    values = [int(field.strip().split()[0]) for field in fields]
    pids = set()
    for line in processes.splitlines():
        if not line.strip():
            continue
        parts = next(csv.reader([line], skipinitialspace=True))
        pid = int(parts[0].strip())
        # On WDDM, nvidia-smi reports D3D desktop compositors in this table too.
        # Keep likely inference/training processes; utilization is also required low for benchmarks.
        process_name = parts[1].strip().strip('"').replace("\\", "/").rsplit("/", 1)[-1].lower() if len(parts) > 1 else ""
        # psutil may return the Windows image name without its .exe suffix,
        # while some nvidia-smi builds include it. Compare both as one stem.
        process_stem = process_name.removesuffix(".exe")
        desktop_stems = {name.removesuffix(".exe") for name in _DESKTOP_PROCESSES}
        if process_stem not in desktop_stems:
            pids.add(pid)
    return Sample(*values, compute_pids=tuple(sorted(pids)))


def sample_gpu(nvidia_smi: str = "nvidia-smi") -> Sample:
    gpu = subprocess.run(
        [nvidia_smi, "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    processes = subprocess.run(
        [nvidia_smi, "--query-compute-apps=pid,name", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    ).stdout
    # WDDM can return placeholder names for GUI contexts. Resolve PIDs through
    # Windows where possible so desktop rendering apps do not masquerade as compute.
    try:
        import psutil

        resolved = []
        for line in processes.splitlines():
            if not line.strip():
                continue
            pid = line.split(",", 1)[0].strip()
            try:
                name = psutil.Process(int(pid)).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                name = line.split(",", 1)[1].strip() if "," in line else ""
            resolved.append(f"{pid},{name}")
        processes = "\n".join(resolved)
    except ImportError:
        pass
    if os.name == "nt":
        # Protected WDDM desktop processes can deny psutil image-name reads.
        # tasklist still reports their image name; use it only to resolve those
        # inaccessible PIDs, leaving unknown names conservatively classified.
        try:
            tasklist = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"], check=True,
                capture_output=True, text=True,
            ).stdout
            names = {}
            for line in tasklist.splitlines():
                try:
                    row = next(csv.reader([line]))
                    if len(row) >= 2:
                        names[row[1].strip()] = row[0].strip()
                except (StopIteration, csv.Error):
                    continue
            resolved = []
            for line in processes.splitlines():
                if not line.strip():
                    continue
                pid, _, reported_name = line.partition(",")
                reported_name = reported_name.strip()
                if reported_name.lower() in {"[insufficient permissions]", ""}:
                    reported_name = names.get(pid.strip(), reported_name)
                resolved.append(f"{pid.strip()},{reported_name}")
            processes = "\n".join(resolved)
        except (OSError, subprocess.CalledProcessError):
            pass
    return parse_gpu_csv(gpu, processes)


def eligible(sample: Sample, estimated_peak_mib: int, margin_mib: int = DEFAULT_MARGIN_MIB,
             benchmark: bool = False, current_pid: int | None = None) -> bool:
    enough_memory = sample.free_mib >= estimated_peak_mib + margin_mib
    if not enough_memory:
        return False
    if benchmark:
        other_compute = [pid for pid in sample.compute_pids if pid != current_pid]
        return sample.utilization_percent <= 5 and not other_compute
    return True


def wait_for_gpu(estimated_peak_mib: int, *, margin_mib: int = DEFAULT_MARGIN_MIB,
                 consecutive: int = 1, poll_seconds: int = 60, benchmark: bool = False,
                 log_path: Path | None = None, nvidia_smi: str = "nvidia-smi") -> Sample:
    """Wait without timeout; log every sample and return after consecutive eligible polls."""
    if estimated_peak_mib < 0 or margin_mib < 0 or consecutive < 1 or poll_seconds < 1:
        raise ValueError("memory values must be nonnegative and poll/consecutive counts positive")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    good = 0
    while True:
        sample = sample_gpu(nvidia_smi)
        ok = eligible(sample, estimated_peak_mib, margin_mib, benchmark, current_pid=os.getpid())
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "estimated_peak_mib": estimated_peak_mib,
                  "safety_margin_mib": margin_mib, "benchmark_exclusive": benchmark,
                  "eligible": ok, **asdict(sample)}
        line = json.dumps(record)
        print(line, flush=True)
        if log_path:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
        good = good + 1 if ok else 0
        if good >= consecutive:
            return sample
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-gib", type=float, required=True, help="conservative peak VRAM estimate")
    parser.add_argument("--margin-gib", type=float, default=2.0)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--benchmark-exclusive", action="store_true", help="also require <=5%% utilization and no other compute PIDs")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    wait_for_gpu(round(args.estimate_gib * MIB_PER_GIB), margin_mib=round(args.margin_gib * MIB_PER_GIB),
                 consecutive=args.consecutive, poll_seconds=args.poll_seconds,
                 benchmark=args.benchmark_exclusive, log_path=args.log)


if __name__ == "__main__":
    main()
