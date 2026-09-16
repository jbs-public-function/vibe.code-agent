"""GPU stats adapter (NFR-00.3, XR-5): the only module allowed to import `pynvml`.

NVML may be unavailable inside sandboxed shells, so an `nvidia-smi` CSV
fallback is mandatory, not optional (Risks, 00-foundation).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import time
from dataclasses import dataclass

_MIB = 1024 * 1024

# IMP-02.28: the admin dashboard refreshes GPU stats every few seconds and
# each refresh spawns ~2 `nvidia-smi` processes; cache the fallback for 1 s
# so bursts of callers (dashboard + `vibe status`) share one pair of reads.
_SMI_TTL_SECONDS = 1.0

# (monotonic timestamp, stats) — only ever written by the nvidia-smi path.
_smi_cache: tuple[float, GpuStats] | None = None


@dataclass(frozen=True)
class GpuProcess:
    """A single process using the GPU."""

    pid: int
    name: str
    used_mib: int


@dataclass(frozen=True)
class GpuStats:
    """Aggregate GPU state: memory, utilization, and compute processes."""

    used_mib: int
    total_mib: int
    utilization_pct: int
    processes: tuple[GpuProcess, ...]


def get_gpu_stats() -> GpuStats:
    """Return current GPU stats via NVML, falling back to `nvidia-smi` CSV output.

    Raises `RuntimeError` when NVML fails and `nvidia-smi` is unavailable
    or itself fails.
    """
    try:
        import pynvml  # deferred: a missing pynvml must not break module import

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            processes: list[GpuProcess] = []
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                name_bytes = pynvml.nvmlSystemGetProcessName(proc.pid) or b"<unknown>"
                processes.append(
                    GpuProcess(
                        pid=proc.pid,
                        name=name_bytes.decode("utf-8", "replace"),
                        used_mib=int(proc.usedGpuMemory) // _MIB,
                    )
                )
            processes.sort(key=lambda p: p.pid)
            return GpuStats(
                used_mib=int(memory.used) // _MIB,
                total_mib=int(memory.total) // _MIB,
                utilization_pct=int(utilization.gpu),
                processes=tuple(processes),
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
    except Exception:
        # Any NVML failure (ImportError, NVMLError, ...) falls through to nvidia-smi.
        pass

    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "no GPU stats available: NVML failed and nvidia-smi is not installed"
        )

    global _smi_cache
    now = time.monotonic()
    cached = _smi_cache
    if cached is not None and now - cached[0] < _SMI_TTL_SECONDS:
        return cached[1]
    stats = _smi_stats()
    _smi_cache = (now, stats)
    return stats


def _smi_stats() -> GpuStats:
    """One fresh pair of `nvidia-smi` reads, parsed into `GpuStats`."""
    gpu_stats = parse_gpu_csv(
        _run_smi(
            [
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    process_csv = _run_smi(
        [
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = parse_process_csv(process_csv) if process_csv.strip() else []
    return GpuStats(
        used_mib=gpu_stats.used_mib,
        total_mib=gpu_stats.total_mib,
        utilization_pct=gpu_stats.utilization_pct,
        processes=tuple(processes),
    )


def parse_gpu_csv(text: str) -> GpuStats:
    """Parse `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu` output.

    Pure and side-effect free (unit-test seam). Raises `ValueError` on
    malformed input.
    """
    rows = list(csv.reader(text.splitlines()))
    if not rows or len(rows[0]) < 3:
        raise ValueError(f"malformed nvidia-smi GPU output: {text!r}")

    def cell_int(cell: str) -> int:
        stripped = cell.strip()
        return int(stripped) if stripped else 0

    return GpuStats(
        used_mib=cell_int(rows[0][0]),
        total_mib=cell_int(rows[0][1]),
        utilization_pct=cell_int(rows[0][2]),
        processes=(),
    )


def parse_process_csv(text: str) -> list[GpuProcess]:
    """Parse `nvidia-smi --query-compute-apps=pid,process_name,used_memory` output.

    Pure and side-effect free (unit-test seam). Process names may contain
    commas and `nvidia-smi` quoting is unreliable, so the name is rebuilt by
    joining all fields between the pid and the trailing memory field.
    Unparseable rows are silently skipped.
    """
    processes: list[GpuProcess] = []
    for row in csv.reader(text.splitlines()):
        if len(row) < 3:
            continue
        try:
            pid = int(row[0].strip())
            used_mib = int(row[-1].strip())
        except ValueError:
            continue
        name = ",".join(field.strip() for field in row[1:-1])
        if not name:
            name = "<unknown>"
        processes.append(GpuProcess(pid=pid, name=name, used_mib=used_mib))
    return processes


def _run_smi(args: list[str]) -> str:
    """Run `nvidia-smi` with `args` and return stdout.

    Converts any failure (`CalledProcessError`, `TimeoutExpired`, `OSError`,
    or anything else) into a `RuntimeError` that includes stderr when present.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception as exc:
        stderr = getattr(exc, "stderr", None)
        detail = stderr.strip() if stderr else str(exc)
        raise RuntimeError(f"nvidia-smi failed: {detail}") from exc
    return result.stdout
