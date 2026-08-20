from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time

from app_paths import get_runtime_paths

PATHS = get_runtime_paths()
JOBS_DIR = PATHS.jobs_dir
METRICS_CACHE_TTL = 2.0
_cache_lock = threading.Lock()
_metrics_cache: dict[str, tuple[float, dict[str, object]]] = {}


def run_command(command: list[str], timeout: float = 3.0) -> str:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return result.stdout.strip()


def cached_snapshot(cache_key: str, loader) -> dict[str, object]:
    now = time.time()
    with _cache_lock:
        cached = _metrics_cache.get(cache_key)
        if cached and now - cached[0] < METRICS_CACHE_TTL:
            return cached[1]

    snapshot = loader()
    with _cache_lock:
        _metrics_cache[cache_key] = (time.time(), snapshot)
    return snapshot


def _get_cpu_snapshot_uncached() -> dict[str, object]:
    if not shutil.which("vmstat"):
        return {"available": False, "summary": "CPU stats unavailable on this machine."}

    try:
        output = run_command(["vmstat", "1", "2"], timeout=3.5)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return {"available": False, "summary": "CPU stats unavailable right now."}
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 3:
        return {"available": False, "summary": "CPU stats unavailable on this machine."}

    values = lines[-1].split()
    if len(values) < 17:
        return {"available": False, "summary": "CPU stats unavailable on this machine."}

    waiting = int(values[15])
    idle = int(values[14])
    user = int(values[12])
    system = int(values[13])
    cpu_busy = max(0, min(100, 100 - idle))

    return {
        "available": True,
        "cpu_busy_percent": cpu_busy,
        "user_percent": user,
        "system_percent": system,
        "wait_percent": waiting,
        "idle_percent": idle,
        "summary": f"CPU {cpu_busy}% busy",
    }


def get_cpu_snapshot() -> dict[str, object]:
    return cached_snapshot("cpu", _get_cpu_snapshot_uncached)


def _get_gpu_snapshot_uncached() -> dict[str, object]:
    if shutil.which("nvidia-smi"):
        try:
            output = run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return {"available": False, "summary": "GPU stats unavailable right now."}
        first = output.splitlines()[0].strip()
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 6:
            name, gpu_util, memory_util, memory_used, memory_total, temperature = parts[:6]
            gpu_util_int = int(float(gpu_util))
            memory_util_int = int(float(memory_util))
            memory_used_int = int(float(memory_used))
            memory_total_int = int(float(memory_total))
            temperature_int = int(float(temperature))
            return {
                "available": True,
                "vendor": "nvidia",
                "name": name,
                "gpu_util_percent": gpu_util_int,
                "memory_util_percent": memory_util_int,
                "memory_used_mb": memory_used_int,
                "memory_total_mb": memory_total_int,
                "temperature_c": temperature_int,
                "summary": f"GPU {gpu_util_int}% • VRAM {memory_used_int}/{memory_total_int} MB",
            }
    return {"available": False, "summary": "GPU stats unavailable on this machine."}


def get_gpu_snapshot() -> dict[str, object]:
    return cached_snapshot("gpu", _get_gpu_snapshot_uncached)


def print_processes(limit: int) -> None:
    output = run_command(
        ["ps", "-eo", "pid,ppid,stat,etime,%cpu,%mem,comm,args", "--sort=-%cpu"]
    )
    lines = output.splitlines()
    print("\n".join(lines[: max(2, limit + 1)]))


def print_cpu() -> None:
    print(run_command(["vmstat", "1", "2"]))


def print_gpu() -> None:
    snapshot = get_gpu_snapshot()
    if snapshot["available"]:
        print(
            f'{snapshot["name"]}, {snapshot["gpu_util_percent"]} %, {snapshot["memory_util_percent"]} %, '
            f'{snapshot["memory_used_mb"]} MiB, {snapshot["memory_total_mb"]} MiB, {snapshot["temperature_c"]}'
        )
    else:
        print("No supported GPU monitoring command found.")


def summarize_jobs() -> str:
    jobs = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

    active = [job for job in jobs if job.get("state") in {"queued", "processing"}]
    active.sort(key=lambda item: (item.get("state") != "processing", item.get("created_at", 0)))
    if not active:
        return "No queued or processing TTS jobs."

    lines = []
    processing_count = sum(1 for job in active if job.get("state") == "processing")
    queued_count = sum(1 for job in active if job.get("state") == "queued")
    lines.append(f"Active TTS jobs: {processing_count} processing, {queued_count} queued")
    for index, job in enumerate(active, start=1):
        title = job.get("title", "Untitled")
        state = job.get("state", "unknown")
        progress = int(round(float(job.get("progress", 0.0)) * 100))
        lines.append(f"{index}. {title} [{state}] {progress}%")
    return "\n".join(lines)


def watch(interval: float, limit: int) -> None:
    while True:
        print("\033c", end="")
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        print()
        print(summarize_jobs())
        print()
        print("Top processes")
        print_processes(limit)
        print()
        print("CPU")
        print_cpu()
        print()
        print("GPU")
        try:
            print_gpu()
        except subprocess.CalledProcessError as exc:
            print(exc.stderr.strip() or exc)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local monitor for TTS jobs, processes, CPU, and GPU.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    proc_parser = subparsers.add_parser("processes", help="Show top CPU processes.")
    proc_parser.add_argument("--limit", type=int, default=12)

    subparsers.add_parser("cpu", help="Show CPU and system activity via vmstat.")
    subparsers.add_parser("gpu", help="Show GPU usage if nvidia-smi is available.")

    watch_parser = subparsers.add_parser("watch", help="Refresh TTS jobs, processes, CPU, and GPU in a loop.")
    watch_parser.add_argument("--interval", type=float, default=3.0)
    watch_parser.add_argument("--limit", type=int, default=10)

    subparsers.add_parser("jobs", help="Show queued and processing TTS jobs from local metadata.")

    args = parser.parse_args()

    if args.command == "processes":
        print_processes(args.limit)
        return 0
    if args.command == "cpu":
        print_cpu()
        return 0
    if args.command == "gpu":
        print_gpu()
        return 0
    if args.command == "jobs":
        print(summarize_jobs())
        return 0
    if args.command == "watch":
        try:
            watch(args.interval, args.limit)
        except KeyboardInterrupt:
            print()
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
