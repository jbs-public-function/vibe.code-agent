"""Shared helpers for the vibe laboratory scripts (FR-05.10, NFR-05.3, NFR-05.4).

The laboratory runs PyTorch experiments, but this module — and every script
that imports it — must stay importable in a torch-free environment: heavy
imports (`torch`, `pynvml`) happen inside functions so the unit tests run
without a GPU or PyTorch installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Exit codes (mirrors src/vibe/errors.py; the lab must not import src/vibe).
EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2

# Repository-relative layout: artifacts live under laboratory/ (FR-05.11).
LAB_DIR = Path(__file__).resolve().parent
MODELS_DIR = LAB_DIR / "models"
DATA_DIR = LAB_DIR / "data"
RUNS_DIR = LAB_DIR / "runs"

# GiB per billion parameters by precision (spec VRAM budget table).
BYTES_PER_PARAM = {"fp16": 2.0, "bf16": 2.0, "4bit": 0.55}


def setup_logging(level: int = logging.INFO) -> None:
    """Route lab logging to stderr with a compact timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@dataclass
class Run:
    """One experiment run: a directory with args, metrics, and a console log (FR-05.10)."""

    dir: Path
    script: str
    started: datetime

    def write_args(self, argv: list[str]) -> None:
        """Write the exact argv the script was invoked with to `args.json`."""
        (self.dir / "args.json").write_text(
            json.dumps({"script": self.script, "argv": argv}, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge `metrics` into `metrics.json` (repeated calls accumulate)."""
        path = self.dir / "metrics.json"
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update(metrics)
        path.write_text(
            json.dumps(existing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def log_line(self, message: str) -> None:
        """Append one timestamped line to `console.log`."""
        with (self.dir / "console.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}  {message}\n")


def start_run(script: str, argv: list[str], runs_dir: Path | None = None) -> Run:
    """Create a per-run directory and return its `Run` handle (FR-05.10).

    The directory is named `<YYYYmmdd-HHMMSS>-<script>` under `runs_dir`
    (default `RUNS_DIR`; the default is resolved at call time so tests can
    repoint `common.RUNS_DIR`). On collision a `-1`, `-2`, ... suffix is
    appended. `args.json` is written before returning.
    """
    if runs_dir is None:
        runs_dir = RUNS_DIR
    now = datetime.now(UTC)
    base = runs_dir / f"{now:%Y%m%d-%H%M%S}-{script}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{now:%Y%m%d-%H%M%S}-{script}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    run = Run(dir=run_dir, script=script, started=now)
    run.write_args(argv)
    return run


def log(run: Run, message: str) -> None:
    """Record a message in the run's console log and echo it to stderr."""
    run.log_line(message)
    print(message, file=sys.stderr)


def free_vram_gib() -> float | None:
    """Best-effort free GPU memory in GiB, or None when no GPU is visible.

    Tries `torch.cuda.mem_get_info()` first (only when CUDA is available),
    then falls back to NVML via `pynvml`. Never raises (NFR-05.3).
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return float(free_bytes) / 2**30
    except Exception:  # noqa: BLE001 — guards must never raise
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.free) / 2**30
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return None
    return None


def vram_guard(estimate_gib: float, *, force: bool) -> tuple[bool, str]:
    """Check an estimated VRAM need against the free memory (NFR-05.3).

    Refuses when the estimate exceeds free VRAM by more than 10 %; warns but
    allows when it merely exceeds free VRAM; `force` bypasses the refusal.
    Returns `(allowed, one-sentence message)`.
    """
    free = free_vram_gib()
    if free is None:
        return True, "VRAM unknown — guard skipped"
    if force:
        return True, (
            f"VRAM guard overridden by --force (estimate {estimate_gib:.1f} GiB, "
            f"{free:.1f} GiB free)"
        )
    if estimate_gib > free * 1.1:
        return False, (
            f"estimated {estimate_gib:.1f} GiB exceeds free VRAM ({free:.1f} GiB) "
            "by more than 10% — pass --force to override"
        )
    if estimate_gib > free:
        return True, (
            f"estimated {estimate_gib:.1f} GiB exceeds free VRAM ({free:.1f} GiB) "
            "— proceeding anyway"
        )
    return True, f"estimated {estimate_gib:.1f} GiB fits in free VRAM ({free:.1f} GiB)"


def _vibe_status_json() -> str | None:
    """Run `vibe status --json` and return its stdout, or None on failure.

    Raises FileNotFoundError when the `vibe` CLI itself is missing so
    `serving_guard` can distinguish "not installed" (guard skipped) from
    "controller down" (guard skipped with a different warning).
    """
    proc = subprocess.run(
        ["vibe", "status", "--json"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def serving_guard(allow_coexist: bool) -> tuple[bool, str]:
    """Refuse to run while a model is READY on the GPU, unless allowed (NFR-05.4).

    Parses `vibe status --json` (`{"state": ..., "active_alias": ...}`):
    state READY or a non-null active alias means the GPU is serving.
    """
    try:
        raw = _vibe_status_json()
    except FileNotFoundError:
        return True, "vibe CLI not found — guard skipped"
    except subprocess.TimeoutExpired:
        return True, "`vibe status` timed out — serving guard skipped"
    if raw is None or not raw.strip():
        return True, "controller not reachable — serving guard skipped"
    try:
        status = json.loads(raw)
    except json.JSONDecodeError:
        return True, "could not parse `vibe status` output — serving guard skipped"
    state = status.get("state")
    active_alias = status.get("active_alias")
    if state != "READY" and not active_alias:
        return True, "no model is READY on the GPU"
    if allow_coexist:
        return True, "a model is READY on the GPU — proceeding because --allow-coexist was passed"
    return False, "a model is READY on the GPU — run `vibe unload` first (or pass --allow-coexist)"


def parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file: one JSON object per line, blank lines skipped.

    Raises ValueError with the offending line number on bad JSON or when a
    line does not hold a JSON object.
    """
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        rows.append(obj)
    return rows


def estimate_vram_gib(params: float, bytes_per_param: float, seq_len: int = 2048) -> float:
    """Estimate inference VRAM in GiB (spec VRAM budget table, NFR-05.3).

    Assumes: weights = params (billions) × GiB per billion params; a fixed
    2.5 GiB for CUDA context, framework, and tokenizer; and 0.5 GiB of KV
    cache + activations at 2048 tokens, scaled linearly with `seq_len`.
    """
    return params * bytes_per_param + 2.5 + 0.5 * (seq_len / 2048)


def model_params_billions(config: dict | None) -> float | None:
    """Approximate decoder-only model size in billions of parameters.

    Best-effort heuristic from a transformers-style config.json: per layer
    attention (Q, K, V, O ≈ 4·h²) + MLP (3·h·intermediate), plus vocab·h
    embeddings. QKV bias, tied embeddings, and MoE gates are ignored, so the
    result is only approximate. Returns None when the config is absent or
    fields are missing, and never raises.
    """
    if config is None:
        return None
    try:
        layers = float(config["num_hidden_layers"])
        hidden = float(config["hidden_size"])
        intermediate = float(config["intermediate_size"])
        vocab = float(config["vocab_size"])
    except (KeyError, TypeError, ValueError):
        return None
    per_layer = 4.0 * hidden * hidden + 3.0 * hidden * intermediate
    return (layers * per_layer + vocab * hidden) / 1e9


def parse_seed(seed: int | None) -> int:
    """Resolve `--seed` (FR-05.10).

    None → pick a random seed and print it (stderr, keeping stdout clean).
    Explicit seed → seed Python's `random`, then numpy and torch when they
    are importable (lazy imports keep this torch-free).
    """
    if seed is None:
        seed = random.randrange(1 << 32)
        print(f"seed: {seed}", file=sys.stderr)
        return seed
    random.seed(seed)
    try:
        import numpy  # pyright: ignore[reportMissingImports]

        numpy.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # pyright: ignore[reportMissingImports]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass
    return seed


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by every lab script (FR-05.10, NFR-05.3, NFR-05.4)."""
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: random, printed to stderr)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="bypass the VRAM guard (NFR-05.3)",
    )
    parser.add_argument(
        "--allow-coexist",
        action="store_true",
        help="allow running while a model is READY on the GPU (NFR-05.4)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUNS_DIR,
        help="directory for per-run args/metrics/logs (default: laboratory/runs)",
    )


def print_peak_vram(run: Run) -> None:
    """Record `peak_vram_gib` in metrics and print it, when CUDA is available.

    Silent no-op in torch-free or CPU-only environments (FR-05.10).
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except Exception:  # noqa: BLE001
        return
    try:
        if not torch.cuda.is_available():
            return
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
    except Exception:  # noqa: BLE001
        return
    message = f"peak_vram_gib = {peak_gib:.2f}"
    run.log_line(message)
    print(message)
    run.write_metrics({"peak_vram_gib": round(peak_gib, 4)})
