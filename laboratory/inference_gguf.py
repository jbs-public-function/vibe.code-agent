"""Run GGUF inference via transformers or llama-cpp-python (FR-05.6, NFR-05.3, FR-05.10).

Same prompt/chat/metrics interface as inference_hf.py; the VRAM estimate
assumes Q4 GGUF weights (0.55 GiB per billion params).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import common

BACKENDS = ("transformers", "llamacpp")
GGUF_BYTES_PER_PARAM = common.BYTES_PER_PARAM["4bit"]  # Q4_K_M ≈ 4-bit

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)[bB]")


def build_parser() -> argparse.ArgumentParser:
    """Build the inference_gguf.py argument parser (FR-05.6)."""
    parser = argparse.ArgumentParser(
        prog="inference_gguf.py",
        description="Run a GGUF model via transformers or llama-cpp-python (FR-05.6).",
    )
    parser.add_argument(
        "--gguf",
        required=True,
        help="path to a .gguf file, or repo:filename on the Hub",
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default="transformers",
        help="inference backend (default: transformers)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="layers to offload for the llamacpp backend (default: all)",
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


def _resolve_gguf(gguf: str) -> Path:
    """Resolve a --gguf spec to a local file (seam for tests).

    Plain paths are used as-is; `repo:filename` downloads the file from the
    Hub via `hf_hub_download` (a cache hit when already present).
    """
    if ":" in gguf:
        repo_id, _, filename = gguf.rpartition(":")
        if not filename:
            raise ValueError(f"invalid --gguf spec {gguf!r}: expected repo:filename")
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    path = Path(gguf)
    if not path.is_file():
        raise ValueError(f"--gguf file not found: {gguf}")
    return path.resolve()


def _guess_params_from_name(name: str) -> float | None:
    """Best-effort parameter count (billions) from a model filename.

    e.g. 'qwen2.5-coder-7b-instruct-q4_k_m.gguf' → 7.0, None when
    unrecognized. Drives the VRAM guard when no config.json is available.
    """
    for part in name.replace(".gguf", "").replace("_", "-").split("-"):
        match = _PARAM_RE.fullmatch(part)
        if match:
            return float(match.group(1))
    return None


def _load_gguf_model(gguf_path: Path, backend: str, n_gpu_layers: int) -> tuple[Any, Any]:
    """Load a GGUF file with the selected backend (seam; lazy heavy imports).

    Returns `(model, tokenizer)`; the llamacpp backend combines both roles
    in one object, returned twice. The transformers backend reads the tokenizer
    from the same directory as the GGUF file (download.py the repo first).
    """
    if backend == "llamacpp":
        from llama_cpp import Llama  # pyright: ignore[reportMissingImports]

        model = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False,
        )
        return model, model
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    model_dir = gguf_path.parent
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), gguf_file=gguf_path.name)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        gguf_file=gguf_path.name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def _stream_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    backend: str,
    **gen_kwargs: Any,
) -> Iterator[str]:
    """Generate from `prompt`, yielding one token string at a time (seam).

    Real implementation: transformers' TextIteratorStreamer, or the llamacpp
    streaming completion API. Tests monkeypatch this to yield canned tokens.
    """
    if backend == "llamacpp":
        kwargs: dict[str, Any] = {"stream": True}
        if "max_new_tokens" in gen_kwargs:
            kwargs["max_tokens"] = gen_kwargs["max_new_tokens"]
        if "temperature" in gen_kwargs:
            kwargs["temperature"] = gen_kwargs["temperature"]
        for chunk in model(prompt, **kwargs):
            yield chunk["choices"][0]["text"]
        return
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


def main(argv: list[str] | None = None) -> int:
    """Run GGUF inference. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prompt is not None and args.chat:
        print("usage error: --prompt and --chat are mutually exclusive", file=sys.stderr)
        return common.EXIT_USAGE
    if args.prompt is None and not args.chat:
        print("usage error: pass --prompt '...' or --chat", file=sys.stderr)
        return common.EXIT_USAGE
    run = common.start_run(
        "inference_gguf",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    try:
        gguf_path = _resolve_gguf(args.gguf)
    except ValueError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE
    params = _guess_params_from_name(gguf_path.name)
    if params is None:
        print("model size unknown from filename — VRAM guard skipped", file=sys.stderr)
    else:
        estimate = common.estimate_vram_gib(params, GGUF_BYTES_PER_PARAM)
        allowed, message = common.vram_guard(estimate, force=args.force)
        common.log(run, message)
        if not allowed:
            return common.EXIT_RUNTIME
    allowed, message = common.serving_guard(args.allow_coexist)
    common.log(run, message)
    if not allowed:
        return common.EXIT_RUNTIME

    try:
        model, tokenizer = _load_gguf_model(gguf_path, args.backend, args.n_gpu_layers)
    except Exception as exc:
        common.log(run, f"model load failed: {exc}")
        return common.EXIT_RUNTIME

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature

    def generate(prompt: str) -> None:
        start = time.perf_counter()
        n_tokens = 0
        for token in _stream_tokens(model, tokenizer, prompt, args.backend, **gen_kwargs):
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
                "precision": "4bit",
                "backend": args.backend,
            }
        )

    if args.prompt is not None:
        generate(args.prompt)
    else:
        print("chat mode: type prompts, Ctrl-D to exit", file=sys.stderr)
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                continue
            generate(prompt)
    run.write_metrics({"seed": seed})
    common.print_peak_vram(run)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
