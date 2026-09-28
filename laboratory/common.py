"""Shared helpers for the vibe laboratory scripts (FR-05.10, NFR-05.3, NFR-05.4).

The laboratory runs PyTorch experiments, but this module — and every script
that imports it — must stay importable in a torch-free environment: heavy
imports (`torch`, `pynvml`, `vibe.client`) happen inside functions so the
unit tests run without a GPU or PyTorch installed.
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

# Exit codes (mirrors src/vibe/errors.py; the lab must not import src/vibe
# at module level).
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

# IMP-05.11: `vibe status` can take a cold typer+grpc import; 5 s was too
# tight, 15 s is generous without hanging the script forever.
STATUS_TIMEOUT_SECONDS = 15.0

# IMP-05.8: model states that hold the GPU (NFR-05.4). READY serves; LOADING
# and UNLOADING have a container on the card and a job in flight.
BUSY_STATES = {"READY", "LOADING", "UNLOADING"}

# IMP-05.10: training VRAM model constants (see estimate_training_vram_gib).
# 12 B per LoRA parameter: bf16 param + grad + two Adam moments.
ADAPTER_OPT_BYTES_PER_PARAM = 12.0
# ~34 fp16 activations per (layer, token, hidden) without recomputation.
ACTIVATION_BYTES = 68.0
# Gradient checkpointing recomputes; ~40 % of activations are retained.
GRAD_CKPT_FACTOR = 0.4
# Fixed framework/CUDA-context/tokenizer overhead (GiB), as in the inference
# estimator.
FRAMEWORK_GIB = 2.5


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

    def read_metrics(self) -> dict[str, Any]:
        """Return the metrics recorded so far ({} when absent)."""
        path = self.dir / "metrics.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

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
    then falls back to NVML via `pynvml` (nvidia-ml-py, IMP-05.7). Never
    raises (NFR-05.3).
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


def _status_via_grpc(timeout: float = STATUS_TIMEOUT_SECONDS) -> str:
    """Fetch the controller `Status` via a direct gRPC `GetStatus` (IMP-05.11).

    IMP-05.2 puts the `vibe` package in the lab venv, so the CLI subprocess
    is only a fallback. Returns the same JSON shape as `vibe status --json`
    (snake_case keys, real ints — the IMP-03.15 contract). Raises
    ImportError when the `vibe` package is unavailable so `_vibe_status_json`
    can fall back to the CLI.
    """
    import asyncio

    from vibe.client import ControllerClient  # pyright: ignore[reportMissingImports]
    from vibe.gen.vibe.v1 import control_pb2  # pyright: ignore[reportMissingImports]

    async def _get() -> str:
        client = ControllerClient()
        try:
            status = await client.status()
        finally:
            await client.close()
        return json.dumps(
            {
                "active_alias": status.active_alias,
                "state": control_pb2.ModelState.Name(status.state),
                "container_id": status.container_id,
                "host_port": int(status.host_port),
                "started_at_unix": int(status.started_at_unix),
                "current_job": {
                    "id": status.current_job.id,
                    "kind": int(status.current_job.kind),
                    "alias": status.current_job.alias,
                },
                "last_error": status.last_error,
            },
            sort_keys=True,
        )

    return asyncio.run(asyncio.wait_for(_get(), timeout=timeout))


def _status_via_cli(timeout: float = STATUS_TIMEOUT_SECONDS) -> str | None:
    """Run `vibe status --json` and return its stdout, or None on failure.

    Raises FileNotFoundError when the `vibe` CLI itself is missing so
    `serving_guard` can report it clearly.
    """
    proc = subprocess.run(
        ["vibe", "status", "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _vibe_status_json(timeout: float = STATUS_TIMEOUT_SECONDS) -> str | None:
    """Return the controller Status as a JSON string, or None on failure.

    Prefers a direct gRPC `GetStatus` (IMP-05.11); falls back to the `vibe`
    CLI when the package is not importable. `FileNotFoundError` propagates
    when neither the package nor the CLI exists.
    """
    try:
        return _status_via_grpc(timeout)
    except ImportError:
        return _status_via_cli(timeout)
    except Exception:  # noqa: BLE001 — controller down/misconfigured
        return None


def status_snapshot() -> dict[str, Any] | None:
    """Fetch and parse the controller Status; None when unavailable.

    Accepts the snake_case keys of `vibe status --json` (IMP-03.15) and the
    camelCase keys of raw `MessageToDict` output during the transition.
    """
    try:
        raw = _vibe_status_json()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _status_key(status: dict[str, Any], snake: str, camel: str) -> Any:
    """Read `snake` or `camel` from a Status dict (IMP-05.8 transition)."""
    if snake in status:
        return status[snake]
    return status.get(camel)


def status_busy(status: dict[str, Any]) -> bool:
    """Whether the GPU is busy serving or transitioning (IMP-05.8).

    Busy = state ∈ {READY, LOADING, UNLOADING}, a non-empty active alias, or
    a current job in flight.
    """
    state = _status_key(status, "state", "state")
    alias = _status_key(status, "active_alias", "activeAlias") or ""
    job = _status_key(status, "current_job", "currentJob")
    job_active = isinstance(job, dict) and bool(job.get("id"))
    return state in BUSY_STATES or bool(alias) or job_active


def serving_guard(allow_coexist: bool) -> tuple[bool, str]:
    """Refuse to run while the GPU is serving, unless allowed (NFR-05.4).

    Parses the controller Status (snake_case `vibe status --json` keys after
    IMP-03.15; camelCase accepted during transition). Busy = state ∈
    {READY, LOADING, UNLOADING} or a current_job in flight.

    IMP-05.2 decision: when the controller/CLI is unreachable the guard
    **hard-fails** instead of silently skipping — a silent skip voids
    NFR-05.4 on the first real run (e.g. a missing `--network host` would
    otherwise go unnoticed). `--allow-coexist` remains the explicit override
    in every branch.
    """
    status = status_snapshot()
    if status is None:
        message = (
            "controller not reachable — serving guard cannot check the GPU; "
            "pass --allow-coexist to proceed without the guard"
        )
        if allow_coexist:
            return True, message
        return False, message
    if status_busy(status):
        if allow_coexist:
            return True, (
                "a model is busy on the GPU — proceeding because "
                "--allow-coexist was passed"
            )
        return False, (
            "a model is busy on the GPU (READY/LOADING/UNLOADING or a job in "
            "flight) — run `vibe unload` first (or pass --allow-coexist)"
        )
    return True, "no model is busy on the GPU"


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
    return params * bytes_per_param + FRAMEWORK_GIB + 0.5 * (seq_len / 2048)


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


def _sum_in_out(config: dict) -> float | None:
    """Σ(in+out) over the Linear modules a LoRA adapter targets (IMP-05.10).

    `target_modules="all-linear"`: per layer attention Q/K/V/O (4 × 2h) plus
    MLP gate/up (h + i) and down (i + h), plus the lm_head (h + vocab).
    """
    try:
        layers = float(config["num_hidden_layers"])
        hidden = float(config["hidden_size"])
        intermediate = float(config["intermediate_size"])
        vocab = float(config["vocab_size"])
    except (KeyError, TypeError, ValueError):
        return None
    per_layer = 10.0 * hidden + 2.0 * intermediate
    return layers * per_layer + (hidden + vocab)


def _activation_gib(config: dict, seq_len: int) -> float | None:
    """Activation memory in GiB: layers × seq × hidden × k, grad-ckpt factor."""
    try:
        layers = float(config["num_hidden_layers"])
        hidden = float(config["hidden_size"])
    except (KeyError, TypeError, ValueError):
        return None
    return (
        layers * seq_len * hidden * ACTIVATION_BYTES * GRAD_CKPT_FACTOR / 2**30
    )


def estimate_training_vram_gib(
    config: dict | None,
    qlora: bool,
    seq_len: int = 2048,
    r: int = 16,
) -> float | None:
    """Training VRAM estimate derived from the model config (IMP-05.10).

    Weights (2 GiB/B bf16, or 0.55 GiB/B for a 4-bit QLoRA base) +
    LoRA adapter/optimizer states (`r × Σ(in+out) × 12 B`: bf16 param, grad,
    two Adam moments) + activations (`layers × seq × hidden × 68 B` with a
    0.4 grad-checkpoint recompute factor) + 2.5 GiB framework overhead.

    Calibrated (bs 1, grad ckpt, seq 2048): QLoRA 7B ≈ 12.3 GiB (allowed on
    16 GB, just under the runbook's ≤12.5 peak) and QLoRA 14B ≈ 22.5 GiB
    (refused), matching the spec VRAM table. Returns None when the config is
    unavailable (the caller then skips the guard, as before).

    When the config has no per-layer dims the previous back-fitted constants
    (`1.5 + 0.36·params` QLoRA / `0.4·params` LoRA adapters) are the
    documented fallback.
    """
    if config is None:
        return None
    params = model_params_billions(config)
    if params is None:
        return None
    sigma = _sum_in_out(config)
    activations = _activation_gib(config, seq_len)
    if sigma is None or activations is None:
        # Documented fallback: the back-fitted constants from the spec table.
        if qlora:
            adapters = 1.5 + 0.36 * params
            base = params * BYTES_PER_PARAM["4bit"]
        else:
            adapters = 0.4 * params
            base = params * BYTES_PER_PARAM["bf16"]
        return base + adapters + FRAMEWORK_GIB + 0.5 * (seq_len / 2048)
    weights = params * (
        BYTES_PER_PARAM["4bit"] if qlora else BYTES_PER_PARAM["bf16"]
    )
    adapter_opt = r * sigma * ADAPTER_OPT_BYTES_PER_PARAM / 2**30
    return weights + adapter_opt + activations + FRAMEWORK_GIB


def parse_seed(seed: int | None) -> int:
    """Resolve `--seed` (FR-05.10).

    None → generate a random seed, print it to **stdout** (run log), and
    apply it exactly like an explicit seed. Both paths seed Python's
    `random`, then numpy and torch when they are importable (lazy imports
    keep this torch-free), so the printed seed reproduces the run
    (IMP-05.9).
    """
    if seed is None:
        seed = random.randrange(1 << 32)
        print(f"seed: {seed}")
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


def record_estimate_vs_peak(run: Run, estimate_gib: float) -> None:
    """Record the training estimate vs. the measured peak in metrics (IMP-05.10).

    `estimate_vs_peak` ≤ 1 means the estimator was conservative; the runbook
    records real values so the constants can be tuned.
    """
    peak = run.read_metrics().get("peak_vram_gib")
    if isinstance(peak, (int, float)) and peak > 0:
        run.write_metrics(
            {
                "estimate_vram_gib": round(estimate_gib, 3),
                "estimate_vs_peak": round(estimate_gib / float(peak), 4),
            }
        )


# Shared flag groups (IMP-05.17: finetune_lora.py and distill.py must expose
# the same training flags, FR-05.7).
DEFAULT_R = 16
DEFAULT_ALPHA = 32
DEFAULT_EPOCHS = 1
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LR = 2e-4
DEFAULT_SAVE_STEPS = 100
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_LR_SCHEDULER = "cosine"
DEFAULT_TARGET_MODULES = "all-linear"


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add the training flags shared by finetune_lora.py and distill.py."""
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="4-bit nf4 QLoRA (quantized base, bf16 adapters)",
    )
    parser.add_argument("--r", type=int, default=DEFAULT_R, help=f"LoRA rank (default: {DEFAULT_R})")
    parser.add_argument(
        "--alpha",
        type=int,
        default=DEFAULT_ALPHA,
        help=f"LoRA alpha (default: {DEFAULT_ALPHA})",
    )
    parser.add_argument(
        "--target-modules",
        default=DEFAULT_TARGET_MODULES,
        help=(
            "LoRA target modules: 'all-linear' or a comma-separated module "
            f"name list (default: {DEFAULT_TARGET_MODULES})"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"max sequence length (default: {DEFAULT_MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        help=f"gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM})",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help=f"save a checkpoint every N steps (default: {DEFAULT_SAVE_STEPS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help=f"linear warmup ratio (default: {DEFAULT_WARMUP_RATIO})",
    )
    parser.add_argument(
        "--lr-scheduler",
        default=DEFAULT_LR_SCHEDULER,
        help=f"learning rate scheduler (default: {DEFAULT_LR_SCHEDULER})",
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by every lab script (FR-05.10, NFR-05.3, NFR-05.4)."""
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: random, printed to stdout and applied)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="bypass the VRAM guard (NFR-05.3)",
    )
    parser.add_argument(
        "--allow-coexist",
        action="store_true",
        help="allow running while a model is busy on the GPU (NFR-05.4)",
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
