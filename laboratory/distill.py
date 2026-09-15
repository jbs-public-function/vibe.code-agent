"""Distill a served teacher model into a small student via text SFT (FR-05.8, FR-05.10).

Two sequential phases (the spec's VRAM table implies they cannot overlap):

1. Generate teacher completions through the gateway
   (`http://localhost:8080/v1`, model = teacher alias; the alias must be
   READY, or swap-on-request is used), cached to
   `laboratory/data/<student>.teacher.jsonl` — resumable by prompt index.
2. SFT the student on the cache (reuses finetune_lora internals).

By default the script stops after phase 1 and prints instructions to run
`vibe unload` and re-run with `--train-only`. `--auto-swap` runs `vibe
unload` itself and trains in one go; `--train-only` skips phase 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

import common

GATEWAY_URL = "http://localhost:8080"
DEFAULT_CONCURRENCY = 4


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
        "--auto-swap",
        action="store_true",
        help="run `vibe unload` after generation and train in one go",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="skip generation and train from the cached teacher JSONL",
    )
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
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """Generate teacher completions over the gateway, appending to `out_cache`.

    Pure-async seam (tests drive it with a fake httpx AsyncClient).
    Resumable: prompts whose idx is already in `out_cache` are skipped, so a
    partially completed cache can be re-run. Requests POST
    `{"model": alias, "messages": [...], "temperature": ...}` to
    `{GATEWAY_URL}/v1/chat/completions` with concurrency capped by an
    asyncio.Semaphore; each request is retried once before aborting with an
    error that mentions `vibe load <alias>`. Returns the rows written by this
    call.
    """
    selected = prompts[:n] if n is not None else prompts
    existing = (
        {row["idx"] for row in common.parse_jsonl(out_cache)}
        if out_cache.is_file()
        else set()
    )
    semaphore = asyncio.Semaphore(concurrency)
    written = 0

    async def generate(i: int, prompt: dict) -> None:
        nonlocal written
        if i in existing:
            return
        payload = {
            "model": alias,
            "messages": prompt["messages"],
            "temperature": temperature,
        }
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                async with semaphore:
                    async with httpx.AsyncClient(timeout=60.0) as client:
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

    await asyncio.gather(*(generate(i, prompt) for i, prompt in enumerate(selected)))
    return written


def _run_cli(argv: list[str]) -> int:
    """Run a vibe CLI command, streaming its output; returns its exit code (seam)."""
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    """Run distillation. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.auto_swap and args.train_only:
        print("usage error: --auto-swap and --train-only are mutually exclusive", file=sys.stderr)
        return common.EXIT_USAGE
    run = common.start_run(
        "distill",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    from finetune_lora import _slug

    try:
        prompt_rows = common.parse_jsonl(Path(args.prompts))
        prompts = format_prompts(prompt_rows)
    except (OSError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE
    teacher_cache = common.DATA_DIR / f"{_slug(args.student)}.teacher.jsonl"

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
            rc = _run_cli(["vibe", "unload"])
            if rc != 0:
                common.log(run, f"`vibe unload` failed (exit {rc}) — training skipped")
                return common.EXIT_RUNTIME
            common.log(run, "teacher unloaded — proceeding to training")
        else:
            print(
                "teacher generation done. Unload the teacher (`vibe unload`) "
                "and re-run with --train-only",
                file=sys.stderr,
            )
            run.log_line("teacher generation done — run `vibe unload` and re-run with --train-only")
            return common.EXIT_OK
    elif not teacher_cache.is_file() or not common.parse_jsonl(teacher_cache):
        print(
            f"usage error: --train-only but teacher cache {teacher_cache} is empty — "
            "run without --train-only first",
            file=sys.stderr,
        )
        return common.EXIT_USAGE

    # Phase 2 — SFT the student on the teacher cache (teacher must be
    # unloaded; the serving guard enforces that unless --allow-coexist).
    from finetune_lora import _sft_train
    from finetune_lora import _training_estimate_gib
    from finetune_lora import format_samples

    teacher_rows = common.parse_jsonl(teacher_cache)
    samples = format_samples(teacher_rows)
    common.log(run, f"samples: {len(samples)}")

    from inference_hf import _load_config

    config = _load_config(args.student)
    params = common.model_params_billions(config)
    if params is None:
        print("model params unknown — VRAM guard skipped", file=sys.stderr)
    else:
        # Distill trains the student without QLoRA (finetune_lora defaults).
        allowed, message = common.vram_guard(
            _training_estimate_gib(params, qlora=False),
            force=args.force,
        )
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
            qlora=False,
            r=16,
            alpha=32,
            epochs=1,
            max_seq_len=2048,
            grad_accum=4,
            seed=seed,
            run=run,
        )
    except Exception as exc:
        common.log(run, f"training failed: {exc}")
        return common.EXIT_RUNTIME
    common.log(run, f"adapter: {adapter}")
    run.write_metrics({"out": str(adapter), "samples": len(samples), "seed": seed})
    common.print_peak_vram(run)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
