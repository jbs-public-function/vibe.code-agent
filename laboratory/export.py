"""Merge a LoRA adapter into its base model, optionally AWQ-quantize, and register (FR-05.9, FR-05.10).

The merged model is saved with safetensors + tokenizer; `tokenizer_config.json`
(chat template) and `generation_config.json` are copied explicitly because
vLLM needs the chat template or tool calling breaks in the gateway (spec
risk note). `--register` runs `vibe register <out> --alias <alias>`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import common

CALIBRATION_SAMPLES = 128  # AWQ calibration set size (FR-05.9)


def build_parser() -> argparse.ArgumentParser:
    """Build the export.py argument parser (FR-05.9)."""
    parser = argparse.ArgumentParser(
        prog="export.py",
        description="Merge a LoRA adapter into its base model and optionally register it (FR-05.9).",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="adapter directory produced by finetune_lora.py",
    )
    parser.add_argument("--base", required=True, help="base model: HF repo id or local path")
    parser.add_argument("--out", required=True, type=Path, help="output model directory")
    parser.add_argument(
        "--awq",
        action="store_true",
        help="AWQ-quantize the merged model (autoawq)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="JSONL used for AWQ calibration (first 128 samples)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="run `vibe register <out> --alias <alias>` after export",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="alias for `vibe register` (required with --register)",
    )
    common.add_common_args(parser)
    return parser


def _merge_lora(adapter: Path, base: str, out: Path) -> Path:
    """Merge the adapter into the base model and save to `out` (seam).

    Lazy imports: peft's AutoPeftModelForCausalLM `merge_and_unload`, then a
    safetensors + tokenizer save; the tokenizer config files are copied
    explicitly as the spec risk note requires.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from peft import AutoPeftModelForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    tokenizer.save_pretrained(str(out))
    for name in ("tokenizer_config.json", "generation_config.json"):
        src = adapter / name
        if src.is_file():
            shutil.copyfile(src, out / name)
    return out


def _awq_quantize(
    model_dir: Path,
    out: Path,
    calibration: Path | None,
    n: int = CALIBRATION_SAMPLES,
) -> Path:
    """AWQ-quantize a merged model with autoawq (seam; lazy heavy imports).

    AWQ loads fp16/bf16 weights (~14.5 GB for 7B): `device_map="auto"` with
    CPU offload is used — slow but feasible (spec risk note). The calibration
    set is the first `n` prompts from the calibration JSONL, if provided.
    """
    from awq import AutoAWQForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    calib_data: list[str] = []
    if calibration is not None:
        rows = common.parse_jsonl(calibration)
        for row in rows[:n]:
            text = row.get("prompt") or row.get("completion") or row.get("text") or ""
            if text:
                calib_data.append(str(text))
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4},
        calib_data=calib_data or None,
    )
    out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out), safetensors=True)
    tokenizer.save_pretrained(str(out))
    return out


def _run_cli(argv: list[str]) -> int:
    """Run a vibe CLI command, streaming its output; returns its exit code (seam)."""
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    """Run the export. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.register and not args.alias:
        print("usage error: --register requires --alias", file=sys.stderr)
        return common.EXIT_USAGE
    run = common.start_run(
        "export",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        print(
            f"usage error: --adapter dir has no adapter_config.json: {adapter}",
            file=sys.stderr,
        )
        return common.EXIT_USAGE

    from inference_hf import _load_config

    config = _load_config(args.base)
    params = common.model_params_billions(config)
    if params is None:
        print("model params unknown — VRAM guard skipped", file=sys.stderr)
    else:
        # Merging (and AWQ) load full-precision weights (spec VRAM table).
        allowed, message = common.vram_guard(
            common.estimate_vram_gib(params, common.BYTES_PER_PARAM["bf16"]),
            force=args.force,
        )
        common.log(run, message)
        if not allowed:
            return common.EXIT_RUNTIME
    allowed, message = common.serving_guard(args.allow_coexist)
    common.log(run, message)
    if not allowed:
        return common.EXIT_RUNTIME

    out = args.out.resolve()
    try:
        _merge_lora(adapter.resolve(), args.base, out)
        common.log(run, f"merged adapter into {out}")
        if args.awq:
            _awq_quantize(out, out, args.calibration)
            common.log(run, f"AWQ-quantized model at {out}")
    except Exception as exc:
        common.log(run, f"export failed: {exc}")
        return common.EXIT_RUNTIME

    run.write_metrics(
        {
            "adapter": str(adapter.resolve()),
            "base": args.base,
            "out": str(out),
            "awq": args.awq,
            "seed": seed,
        }
    )

    if args.register:
        rc = _run_cli(["vibe", "register", str(out), "--alias", args.alias])
        if rc != 0:
            common.log(run, f"`vibe register` failed (exit {rc})")
            return common.EXIT_RUNTIME
        common.log(run, f"registered {out} as alias {args.alias}")
    common.print_peak_vram(run)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
