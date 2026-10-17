"""Merge a LoRA adapter into its base model, optionally AWQ-quantize, and register (FR-05.9, FR-05.10).

The merged model is saved with safetensors + tokenizer; `tokenizer_config.json`
(chat template) and `generation_config.json` are copied explicitly because
vLLM needs the chat template or tool calling breaks in the gateway (spec
risk note). `--register` runs `vibe register <out> --alias <alias>`.
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--tool-parser",
        choices=["hermes", "llama3_json"],
        default=None,
        help="tool-call parser for `vibe register` (hermes or llama3_json)",
    )
    common.add_common_args(parser)
    return parser


def _base_file(base: str, name: str) -> Path | None:
    """Resolve a single file from the base model (local dir or hub), or None.

    IMP-05.26: `generation_config.json` must come from the resolved base
    snapshot — the adapter dir never has one.
    """
    local = Path(base)
    if local.is_dir() and (local / name).is_file():
        return local / name
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=base,
                filename=name,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    except Exception:  # noqa: BLE001 — best-effort copy
        return None


def _merge_lora(adapter: Path, base: str, out: Path) -> Path:
    """Merge the adapter into the base model and save to `out` (seam).

    IMP-05.22: the base is loaded explicitly (`AutoModelForCausalLM.from_pretrained(base)`
    + `PeftModel.from_pretrained(model, adapter)` + `merge_and_unload`); the
    old `AutoPeftModelForCausalLM` path resolves the base from
    `adapter_config.json` and silently ignores `--base`. IMP-05.23: the merge
    runs on CPU (`device_map="cpu"`, 32 GB RAM) so 7B exports do not need the
    GPU or `--force`. IMP-05.26: `generation_config.json` is copied from the
    resolved base snapshot.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from peft import PeftModel  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="cpu",
        dtype=torch.bfloat16,  # IMP-05.16: torch_dtype= removed in transformers ≥4.56
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    tokenizer.save_pretrained(str(out))
    gen_cfg = _base_file(base, "generation_config.json")
    if gen_cfg is not None:
        shutil.copyfile(gen_cfg, out / "generation_config.json")
    return out


def _calibration_texts(calibration: Path, n: int = CALIBRATION_SAMPLES) -> list[str]:
    """Flatten the first `n` calibration rows into prompt strings (IMP-05.25).

    The finetune data format is used: `messages` rows are flattened by
    joining every role's content with newlines, `prompt`/`completion`/`text`
    rows use the first present field. Raises ValueError when fewer than `n`
    usable samples exist — autoawq must not silently download `pileval`
    (FR-05.9).
    """
    rows = common.parse_jsonl(calibration)
    texts: list[str] = []
    for row in rows[:n]:
        if "messages" in row:
            messages = row["messages"]
            parts = [
                str(msg["content"])
                for msg in messages
                if isinstance(msg, dict) and isinstance(msg.get("content"), str)
            ]
            text = "\n".join(parts)
        else:
            text = row.get("prompt") or row.get("completion") or row.get("text") or ""
        if text:
            texts.append(str(text))
    if len(texts) < n:
        raise ValueError(
            f"calibration data {calibration} has {len(texts)} usable samples — "
            f"need at least {n} (FR-05.9)"
        )
    return texts


def _awq_quantize(
    model_dir: Path,
    out: Path,
    calibration: Path | None,
    n: int = CALIBRATION_SAMPLES,
) -> Path:
    """AWQ-quantize a merged model with autoawq (seam; lazy heavy imports).

    AWQ loads fp16/bf16 weights (~14.5 GB for 7B): `device_map="auto"` with
    CPU offload is used — slow but feasible (spec risk note). The calibration
    set is the first `n` rows from the calibration JSONL (messages rows are
    flattened, IMP-05.25); fewer than `n` usable samples is an error.
    """
    from awq import AutoAWQForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    calib_data = (
        _calibration_texts(calibration, n) if calibration is not None else None
    )
    if calib_data is None:
        raise ValueError(
            f"--awq requires --calibration <jsonl> with at least {n} samples "
            "(FR-05.9: never fall back to a remote calibration set)"
        )
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4},
        calib_data=calib_data,
    )
    out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out), safetensors=True)
    tokenizer.save_pretrained(str(out))
    return out


def _verify_export(out: Path) -> None:
    """Assert the exported dir has a chat template (IMP-05.26).

    vLLM applies `tokenizer_config.json`'s `chat_template` for chat and tool
    calling; a missing one silently breaks the gateway.
    """
    path = out / "tokenizer_config.json"
    if not path.is_file():
        raise RuntimeError(f"exported tokenizer_config.json missing at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"exported tokenizer_config.json unreadable: {exc}") from exc
    if not isinstance(data, dict) or not data.get("chat_template"):
        raise RuntimeError(
            f"exported tokenizer_config.json at {path} has no chat_template — "
            "chat/tool calling would break in vLLM"
        )


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

    # IMP-05.15: validate arguments before start_run so usage errors do not
    # litter runs/.
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        print(
            f"usage error: --adapter dir has no adapter_config.json: {adapter}",
            file=sys.stderr,
        )
        return common.EXIT_USAGE

    run = common.start_run(
        "export",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    from inference_hf import _load_config

    config = _load_config(args.base)
    params = common.model_params_billions(config)
    if args.awq:
        # IMP-05.23: merging runs on CPU, so only the AWQ step (which loads
        # full-precision weights on the GPU) needs the VRAM guard.
        if params is None:
            print("model params unknown — VRAM guard skipped", file=sys.stderr)
        else:
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
        if args.awq:
            # IMP-05.24: merge to a staging dir, quantize into `--out`, then
            # delete the bf16 shards so vLLM/`vibe register` only ever see
            # quantized weights.
            bf16_dir = out.parent / f"{out.name}.bf16"
            _merge_lora(adapter.resolve(), args.base, bf16_dir)
            common.log(run, f"merged adapter into {bf16_dir} (staging)")
            _awq_quantize(bf16_dir, out, args.calibration)
            shutil.rmtree(bf16_dir, ignore_errors=True)
            common.log(run, f"AWQ-quantized model at {out} (staging dir removed)")
        else:
            _merge_lora(adapter.resolve(), args.base, out)
            common.log(run, f"merged adapter into {out}")
        _verify_export(out)
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
        register_cmd = ["vibe", "register", str(out), "--alias", args.alias]
        if args.awq:
            register_cmd += ["--quantization", "awq"]
        if args.tool_parser:
            register_cmd += ["--tool-parser", args.tool_parser]
        try:
            rc = _run_cli(register_cmd)
        except FileNotFoundError as exc:
            common.log(run, f"`vibe register` failed: {exc}")
            return common.EXIT_RUNTIME
        if rc != 0:
            common.log(run, f"`vibe register` failed (exit {rc})")
            return common.EXIT_RUNTIME
        common.log(run, f"registered {out} as alias {args.alias}")
    common.print_peak_vram(run)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
