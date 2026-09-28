"""Distill a served teacher model into a small student via text SFT (FR-05.8, FR-05.10).

Two sequential phases (the spec's VRAM table implies they cannot overlap):

1. Generate teacher completions through the gateway
   (`http://localhost:8080/v1`, model = teacher alias; the alias must be
   READY, or swap-on-request is used), cached to
   `laboratory/data/<student>.teacher.jsonl` — resumable by prompt index.
2. SFT the student on the cache (reuses finetune_lora internals).

By default the script stops after phase 1 and prints instructions to run
`vibe unload` and re-run with `--train-only`. `--auto-swap` runs `vibe
unload` itself, waits for the GPU to become AVAILABLE, and trains in one go;
`--train-only` skips phase 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

import common

GATEWAY_URL = "http://localhost:8080"
DEFAULT_CONCURRENCY = 4
# IMP-05.18: swap-on-request can hold a request for VIBE_LOAD_TIMEOUT
# (default 600 s); 600 s is the safe default, overridable with --timeout.
DEFAULT_TIMEOUT = 600.0
# IMP-05.19: poll interval/timeout after `vibe unload` until AVAILABLE.
AVAIL_POLL_INTERVAL = 2.0
AVAIL_POLL_TIMEOUT = 300.0
VRAM_STABLE_GIB = 0.1
CLI_TIMEOUT = 300.0


def build_parser() -> argparse.ArgumentParser:
    """Build the distill.py argument parser (FR-05.8)."""
    parser = argparse.ArgumentParser(
        prog="distill.py",
        description="Generate teacher completions via the gateway, then SFT a student model (FR-05.8).",
    )
    parser.add_argument(
        "--teacher-alias",
        required=True,
        help="alias of the teacher model (served through the gateway)",
    )
    parser.add_argument("--student", required=True, help="student base model: HF repo id or local path")
    parser.add_argument("--prompts", required=True, help="JSONL of prompts (messages or prompt rows)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="student adapter dir (default: laboratory/models/<student>-distill)",
    )
    parser.add_argument("--n", type=int, default=None, help="only use the first N prompts")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="teacher sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"per-request HTTP timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="max_tokens for teacher completions (default: model default)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"concurrent teacher requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--auto-swap",
        action="store_true",
        help="run `vibe unload` after generation and train in one go",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="skip generation and train from the cached teacher JSONL",
    )
    common.add_training_args(parser)
    common.add_common_args(parser)
    return parser


def format_prompts(rows: list[dict]) -> list[dict]:
    """Normalize teacher-prompt rows into `{"prompt": str, "messages": [...]}`.

    Accepts `{"messages": [...]}` chat rows (the prompt text is the last user
    turn) or `{"prompt": "..."}` rows. Pure helper covered by tests.
    """
    normalized: list[dict] = []
    for i, row in enumerate(rows):
        if "messages" in row:
            messages = row["messages"]
            if not isinstance(messages, list) or not all(
                isinstance(msg, dict) and "role" in msg and "content" in msg
                for msg in messages
            ):
                raise ValueError(f"row {i}: messages must be a list of {{role, content}} dicts")
            user_turns = [msg["content"] for msg in messages if msg["role"] == "user"]
            if not user_turns:
                raise ValueError(f"row {i}: messages has no user turn")
            normalized.append({"prompt": user_turns[-1], "messages": messages})
        elif "prompt" in row:
            text = row["prompt"]
            if not isinstance(text, str):
                raise ValueError(f"row {i}: prompt must be a string")
            normalized.append(
                {"prompt": text, "messages": [{"role": "user", "content": text}]}
            )
        else:
            raise ValueError(f"row {i}: expected {{messages}} or {{prompt}}")
    return normalized


async def _generate_teacher(
    alias: str,
    prompts: list[dict],
    out_cache: Path,
    temperature: float,
    n: int | None,
    timeout: float = DEFAULT_TIMEOUT,
    max_tokens: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """Generate teacher completions over the gateway, appending to `out_cache`.

    Pure-async seam (tests drive it with a fake httpx AsyncClient).
    Resumable: prompts whose idx is already in `out_cache` are skipped, so a
    partially completed cache can be re-run. IMP-05.18: one shared
    `httpx.AsyncClient` per run with the `--timeout`; `--max-tokens` is sent
    in the payload; concurrency is capped by an asyncio.Semaphore and the
    task group is an `asyncio.TaskGroup`, so one failing row cancels its
    siblings (IMP-05.20). Each request is retried once before aborting with
    an error that mentions `vibe load <alias>`. Cache rows keep the full
    `messages` the teacher saw plus the assistant completion (IMP-05.21).
    Returns the rows written by this call.
    """
    selected = prompts[:n] if n is not None else prompts
    try:
        existing = (
            {row["idx"] for row in common.parse_jsonl(out_cache)}
            if out_cache.is_file()
            else set()
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"teacher cache {out_cache} is corrupt: {exc}") from exc
    semaphore = asyncio.Semaphore(concurrency)
    written = 0

    async def generate(i: int, prompt: dict, client: httpx.AsyncClient) -> None:
        nonlocal written
        if i in existing:
            return
        payload: dict[str, Any] = {
            "model": alias,
            "messages": prompt["messages"],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                async with semaphore:
                    resp = await client.post(
                        f"{GATEWAY_URL}/v1/chat/completions",
                        json=payload,
                    )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                out_cache.parent.mkdir(parents=True, exist_ok=True)
                with out_cache.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "idx": i,
                                "prompt": prompt["prompt"],
                                "messages": prompt["messages"],
                                "completion": content,
                            }
                        )
                        + "\n"
                    )
                written += 1
                return
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(
            f"teacher generation failed after a retry ({last_error}) — "
            f"make sure the teacher is loaded (`vibe load {alias}`) and "
            f"reachable at {GATEWAY_URL}/v1"
        )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with asyncio.TaskGroup() as group:
                for i, prompt in enumerate(selected):
                    group.create_task(generate(i, prompt, client))
        except* RuntimeError as group_err:
            # One failing row cancels its siblings (TaskGroup); surface the
            # first failure as a plain RuntimeError for the caller.
            raise group_err.exceptions[0]
    return written


def _run_cli(argv: list[str], timeout: float = CLI_TIMEOUT) -> int:
    """Run a vibe CLI command, inheriting stdio; returns its exit code (seam).

    IMP-05.20: no `capture_output` — output streams live to the run console —
    and a timeout bounds the call. A missing CLI surfaces as a clean
    `RuntimeError` instead of a raw FileNotFoundError traceback.
    """
    try:
        return subprocess.run(argv, check=False, timeout=timeout).returncode
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"command not found: {' '.join(argv)} — the `vibe` CLI is required "
            f"({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout:g}s: {' '.join(argv)}") from exc


def _poll_until_available(timeout_s: float = AVAIL_POLL_TIMEOUT) -> bool:
    """Poll the controller until AVAILABLE and free VRAM stabilises (IMP-05.19).

    `vibe unload` completes while the container is still being torn down
    (state may linger at UNLOADING with VRAM held), so phase 2 must wait for
    the terminal AVAILABLE state **and** two consecutive stable
    `free_vram_gib()` readings before training. Without a visible GPU the
    state check alone gates (free VRAM is unmeasurable there).
    """
    deadline = time.monotonic() + timeout_s
    last_free: float | None = None
    seen_available = False
    while time.monotonic() < deadline:
        status = common.status_snapshot()
        state = (status or {}).get("state")
        if state == "AVAILABLE":
            free = common.free_vram_gib()
            if seen_available and (
                free is None
                or (last_free is not None and abs(free - last_free) <= VRAM_STABLE_GIB)
            ):
                return True
            seen_available = True
            if free is not None:
                last_free = free
        else:
            seen_available = False
            last_free = None
        time.sleep(AVAIL_POLL_INTERVAL)
    return False


def main(argv: list[str] | None = None) -> int:
    """Run distillation. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.auto_swap and args.train_only:
        print("usage error: --auto-swap and --train-only are mutually exclusive", file=sys.stderr)
        return common.EXIT_USAGE

    # IMP-05.15: validate arguments and data before start_run so usage
    # errors do not litter runs/.
    try:
        prompt_rows = common.parse_jsonl(Path(args.prompts))
        prompts = format_prompts(prompt_rows)
    except (OSError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE

    from finetune_lora import _slug

    teacher_cache = common.DATA_DIR / f"{_slug(args.student)}.teacher.jsonl"
    teacher_rows: list[dict] = []
    if args.train_only:
        # IMP-05.20: a corrupt cache line reports the file:line, not a traceback.
        try:
            teacher_rows = common.parse_jsonl(teacher_cache)
        except (OSError, ValueError) as exc:
            print(f"usage error: teacher cache {exc}", file=sys.stderr)
            return common.EXIT_USAGE
        if not teacher_rows:
            print(
                f"usage error: --train-only but teacher cache {teacher_cache} is empty — "
                "run without --train-only first",
                file=sys.stderr,
            )
            return common.EXIT_USAGE

    run = common.start_run(
        "distill",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    if not args.train_only:
        # Phase 1 — the teacher must be READY; vLLM manages its own VRAM, so
        # neither guard applies here (NFR-05.4 covers the reverse direction).
        try:
            written = asyncio.run(
                _generate_teacher(
                    args.teacher_alias,
                    prompts,
                    teacher_cache,
                    args.temperature,
                    args.n,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    concurrency=args.concurrency,
                )
            )
        except Exception as exc:
            common.log(run, f"teacher generation failed: {exc}")
            return common.EXIT_RUNTIME
        common.log(run, f"teacher completions: {written}")
        run.write_metrics(
            {"teacher_completions": written, "teacher_cache": str(teacher_cache)}
        )
        if args.auto_swap:
            try:
                rc = _run_cli(["vibe", "unload"])
            except RuntimeError as exc:
                common.log(run, f"{exc} — training skipped")
                return common.EXIT_RUNTIME
            if rc != 0:
                common.log(run, f"`vibe unload` failed (exit {rc}) — training skipped")
                return common.EXIT_RUNTIME
            common.log(run, "`vibe unload` issued — waiting for the GPU to become AVAILABLE")
            if not _poll_until_available():
                common.log(
                    run,
                    f"timed out after {AVAIL_POLL_TIMEOUT:g}s waiting for AVAILABLE "
                    "after `vibe unload` — training skipped",
                )
                return common.EXIT_RUNTIME
            common.log(run, "GPU available after unload — proceeding to training")
        else:
            print(
                "teacher generation done. Unload the teacher (`vibe unload`) "
                "and re-run with --train-only",
                file=sys.stderr,
            )
            run.log_line("teacher generation done — run `vibe unload` and re-run with --train-only")
            return common.EXIT_OK

    # Phase 2 — SFT the student on the teacher cache (teacher must be
    # unloaded; the serving guard enforces that unless --allow-coexist).
    from finetune_lora import _sft_train
    from finetune_lora import _target_modules
    from finetune_lora import _training_estimate_gib
    from finetune_lora import format_samples

    if not teacher_rows:
        try:
            teacher_rows = common.parse_jsonl(teacher_cache)
        except (OSError, ValueError) as exc:
            print(f"usage error: teacher cache {exc}", file=sys.stderr)
            return common.EXIT_USAGE
        if not teacher_rows:
            print(
                f"usage error: teacher cache {teacher_cache} is empty — "
                "run without --train-only first",
                file=sys.stderr,
            )
            return common.EXIT_USAGE
    samples = format_samples(teacher_rows)
    common.log(run, f"samples: {len(samples)}")

    from inference_hf import _load_config

    config = _load_config(args.student)
    estimate = _training_estimate_gib(config, args.qlora, args.max_seq_len, args.r)
    if estimate is None:
        print("model params unknown — VRAM guard skipped", file=sys.stderr)
    else:
        run.write_metrics({"estimate_vram_gib": round(estimate, 3)})
        allowed, message = common.vram_guard(estimate, force=args.force)
        common.log(run, message)
        if not allowed:
            return common.EXIT_RUNTIME
    allowed, message = common.serving_guard(args.allow_coexist)
    common.log(run, message)
    if not allowed:
        return common.EXIT_RUNTIME

    out = args.out if args.out is not None else common.MODELS_DIR / f"{_slug(args.student)}-distill"
    try:
        adapter = _sft_train(
            base=args.student,
            samples=samples,
            out=out.resolve(),
            qlora=args.qlora,
            r=args.r,
            alpha=args.alpha,
            target_modules=_target_modules(args.target_modules),
            epochs=args.epochs,
            max_seq_len=args.max_seq_len,
            grad_accum=args.grad_accum,
            save_steps=args.save_steps,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler=args.lr_scheduler,
            seed=seed,
            run=run,
        )
    except Exception as exc:
        common.log(run, f"training failed: {exc}")
        return common.EXIT_RUNTIME
    common.log(run, f"adapter: {adapter}")
    run.write_metrics({"out": str(adapter), "samples": len(samples), "seed": seed})
    common.print_peak_vram(run)
    if estimate is not None:
        common.record_estimate_vs_peak(run, estimate)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
