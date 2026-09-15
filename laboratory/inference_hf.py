"""Run one-shot or chat inference with transformers (FR-05.5, NFR-05.3, FR-05.10).

Streams tokens to stdout, prints tokens/s and peak VRAM, and refuses
configurations that do not fit the documented VRAM budget unless `--force`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import common

AUTO_4BIT_PARAMS = 3.0  # auto-select 4-bit above this many params (FR-05.5)


def build_parser() -> argparse.ArgumentParser:
    """Build the inference_hf.py argument parser (FR-05.5)."""
    parser = argparse.ArgumentParser(
        prog="inference_hf.py",
        description="Run a model with transformers, streaming tokens to stdout (FR-05.5).",
    )
    parser.add_argument("--model", required=True, help="HF repo id or local model directory")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default=None,
        help="load precision (default: auto)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="load with bitsandbytes 4-bit quantization",
    )
    parser.add_argument("--prompt", default=None, help="one-shot prompt (default: --chat)")
    parser.add_argument("--chat", action="store_true", help="interactive loop reading prompts from stdin")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature (default: 0.7)",
    )
    common.add_common_args(parser)
    return parser


def _load_config(model: str) -> dict | None:
    """Return the model's config.json as a dict, or None when unavailable (seam).

    Local path → reads `<model>/config.json`. Repo id → downloads config.json
    via huggingface_hub into the HF cache, then reads it. Best-effort: any
    failure returns None. Never imports torch.
    """
    local = Path(model)
    path: Path | None = None
    if local.is_dir() and (local / "config.json").is_file():
        path = local / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    repo_id=model,
                    filename="config.json",
                    token=os.environ.get("HF_TOKEN"),
                )
            )
        except Exception:  # noqa: BLE001 — config probing is best-effort
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _choose_precision(
    params: float | None,
    dtype: str | None,
    load_in_4bit: bool,
) -> tuple[str, dict[str, Any]]:
    """Decide the load precision and bitsandbytes kwargs (FR-05.5).

    Returns `(precision_label, load_kwargs)` for `_load_pipeline`.
    Auto-selects 4-bit for models > 3B unless `--dtype` or `--load-in-4bit`
    pins a choice.
    """
    if load_in_4bit:
        return "4bit", {"load_in_4bit": True}
    if dtype is not None:
        return dtype, {"dtype": dtype}
    if params is not None and params > AUTO_4BIT_PARAMS:
        print("auto-selecting 4-bit quantization for a >3B model", file=sys.stderr)
        return "4bit", {"load_in_4bit": True}
    return "bf16", {"dtype": "bf16"}


def _load_pipeline(model: str, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """Load `(model, tokenizer)` with transformers (seam; lazy heavy imports)."""
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
    from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]

    kwargs: dict[str, Any] = {"device_map": "auto"}
    if load_kwargs.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if load_kwargs.get("dtype") == "bf16" else torch.float16
        )
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    model_obj.eval()
    return model_obj, tokenizer


def _stream_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    **gen_kwargs: Any,
) -> Iterator[str]:
    """Generate from `prompt`, yielding one token string at a time (seam).

    The real implementation runs `model.generate` on a background thread with
    transformers' TextIteratorStreamer; tests monkeypatch this to yield
    canned tokens without torch.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from threading import Thread
    from transformers import TextIteratorStreamer  # pyright: ignore[reportMissingImports]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(
        target=model.generate,
        kwargs={**inputs, "streamer": streamer, **gen_kwargs},
    )
    thread.start()
    yield from streamer
    thread.join()


def _generate(
    run: common.Run,
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_kwargs: dict[str, Any],
    params: float | None,
    precision: str,
) -> None:
    """Stream one generation to stdout and record tokens/s in metrics (FR-05.5)."""
    start = time.perf_counter()
    n_tokens = 0
    for token in _stream_tokens(model, tokenizer, prompt, **gen_kwargs):
        print(token, end="", flush=True)
        n_tokens += 1
    print()
    elapsed = time.perf_counter() - start
    tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0.0
    common.log(run, f"tokens/s: {tokens_per_sec:.1f} ({n_tokens} tokens)")
    run.write_metrics(
        {
            "tokens": n_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "model_params_billions": params,
            "precision": precision,
        }
    )


def main(argv: list[str] | None = None) -> int:
    """Run inference. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prompt is not None and args.chat:
        print("usage error: --prompt and --chat are mutually exclusive", file=sys.stderr)
        return common.EXIT_USAGE
    if args.prompt is None and not args.chat:
        print("usage error: pass --prompt '...' or --chat", file=sys.stderr)
        return common.EXIT_USAGE
    run = common.start_run(
        "inference_hf",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    config = _load_config(args.model)
    params = common.model_params_billions(config)
    if params is None:
        print("model params unknown — VRAM estimate unavailable", file=sys.stderr)
    precision, load_kwargs = _choose_precision(params, args.dtype, args.load_in_4bit)
    run.log_line(f"precision: {precision}")

    if params is not None:
        estimate = common.estimate_vram_gib(params, common.BYTES_PER_PARAM[precision])
        allowed, message = common.vram_guard(estimate, force=args.force)
        common.log(run, message)
        if not allowed:
            return common.EXIT_RUNTIME
    allowed, message = common.serving_guard(args.allow_coexist)
    common.log(run, message)
    if not allowed:
        return common.EXIT_RUNTIME

    try:
        model, tokenizer = _load_pipeline(args.model, load_kwargs)
    except Exception as exc:
        common.log(run, f"model load failed: {exc}")
        return common.EXIT_RUNTIME

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature

    if args.prompt is not None:
        _generate(run, model, tokenizer, args.prompt, gen_kwargs, params, precision)
    else:
        print("chat mode: type prompts, Ctrl-D to exit", file=sys.stderr)
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                continue
            _generate(run, model, tokenizer, prompt, gen_kwargs, params, precision)
    run.write_metrics({"seed": seed})
    common.print_peak_vram(run)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
