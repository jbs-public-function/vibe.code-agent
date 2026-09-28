"""QLoRA/LoRA fine-tuning with PEFT + TRL SFTTrainer (FR-05.7, NFR-05.3, FR-05.10).

JSONL training data accepts chat rows (`{"messages": [...]}`) or completion
rows (`{"prompt", "completion"}`). The adapter is saved under `--out`
(default `laboratory/models/<base>-lora`) and training resumes from an
existing checkpoint in that directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import common

_CHECKPOINT_RE = re.compile(r"^checkpoint-\d+$")

DEFAULT_R = common.DEFAULT_R
DEFAULT_ALPHA = common.DEFAULT_ALPHA
DEFAULT_EPOCHS = common.DEFAULT_EPOCHS
DEFAULT_MAX_SEQ_LEN = common.DEFAULT_MAX_SEQ_LEN
DEFAULT_GRAD_ACCUM = common.DEFAULT_GRAD_ACCUM


def build_parser() -> argparse.ArgumentParser:
    """Build the finetune_lora.py argument parser (FR-05.7)."""
    parser = argparse.ArgumentParser(
        prog="finetune_lora.py",
        description="Fine-tune a base model with PEFT + TRL SFTTrainer (FR-05.7).",
    )
    parser.add_argument("--base", required=True, help="base model: HF repo id or local path")
    parser.add_argument(
        "--data",
        required=True,
        help="JSONL training file (messages or prompt/completion rows)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="adapter output dir (default: laboratory/models/<base>-lora)",
    )
    common.add_training_args(parser)
    common.add_common_args(parser)
    return parser


def _slug(base: str) -> str:
    """Filesystem-safe slug of a repo id/path.

    e.g. 'Qwen/Qwen2.5-Coder-7B-Instruct' → 'qwen-qwen2.5-coder-7b-instruct'.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-.").lower() or "model"


def format_samples(rows: list[dict]) -> list[dict]:
    """Normalize JSONL rows into SFT samples (FR-05.7, pure helper).

    Chat rows (`{"messages": [{"role", "content"}, ...]}`) pass through; a
    `completion` alongside `messages` (the distill teacher-cache form,
    IMP-05.21) is appended as the assistant turn; completion rows
    (`{"prompt", "completion"}`) become a user/assistant message pair so
    SFTTrainer can apply the tokenizer's chat template. Raises ValueError on
    malformed rows.
    """
    samples: list[dict] = []
    for i, row in enumerate(rows):
        if "messages" in row:
            messages = row["messages"]
            if not messages or not all(
                isinstance(msg, dict) and "role" in msg and "content" in msg
                for msg in messages
            ):
                raise ValueError(f"row {i}: messages must be a list of {{role, content}} dicts")
            completion = row.get("completion")
            if completion is not None:
                if not isinstance(completion, str) or not completion:
                    raise ValueError(f"row {i}: completion must be a non-empty string")
                messages = [dict(msg) for msg in messages] + [
                    {"role": "assistant", "content": completion}
                ]
            samples.append({"messages": messages})
        elif "prompt" in row and "completion" in row:
            samples.append(
                {
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["completion"]},
                    ]
                }
            )
        else:
            raise ValueError(f"row {i}: expected {{messages}} or {{prompt, completion}}")
    return samples


def get_last_checkpoint(out: Path) -> Path | None:
    """Highest-numbered `checkpoint-<n>` dir in `out`, or None when absent.

    IMP-05.12: must return None (never raise) when `out` does not exist or
    contains no checkpoint (pre-created dir, completed run, distill re-run).
    """
    if not out.is_dir():
        return None
    checkpoints = [
        path for path in out.iterdir()
        if path.is_dir() and _CHECKPOINT_RE.fullmatch(path.name)
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1]))


def _legacy_training_estimate_gib(params: float, qlora: bool, seq_len: int = 2048) -> float:
    """Back-fitted training estimate from the spec VRAM table (documented fallback).

    QLoRA: 4-bit base weights (0.55 GiB/B) + bf16 adapter weights, gradients,
    and optimizer states (1.5 GiB fixed + 0.36 GiB per B params) + 2.5 GiB
    framework overhead + an activation/KV term. Plain LoRA: bf16 weights +
    0.4 GiB per B params for adapters/optimizer states. Used only when
    `estimate_training_vram_gib` cannot read per-layer dims from config.json
    (IMP-05.10).
    """
    if qlora:
        base = params * common.BYTES_PER_PARAM["4bit"]
        adapters = 1.5 + 0.36 * params
    else:
        base = params * common.BYTES_PER_PARAM["bf16"]
        adapters = 0.4 * params
    return base + adapters + common.FRAMEWORK_GIB + 0.5 * (seq_len / 2048)


def _training_estimate_gib(
    config: dict | None,
    qlora: bool,
    seq_len: int = DEFAULT_MAX_SEQ_LEN,
    r: int = DEFAULT_R,
) -> float | None:
    """VRAM estimate for LoRA training (NFR-05.3, IMP-05.10).

    Config-derived: weights + `r × Σ(in+out) × 12 B` adapter/optimizer +
    activations with a grad-checkpoint factor. Falls back to the legacy
    back-fitted constants when config.json is unavailable; None when the
    model size is unknown (caller skips the guard).
    """
    estimate = common.estimate_training_vram_gib(config, qlora, seq_len, r)
    if estimate is not None:
        return estimate
    params = common.model_params_billions(config)
    if params is None:
        return None
    return _legacy_training_estimate_gib(params, qlora, seq_len)


def _target_modules(spec: str) -> list[str] | str:
    """`--target-modules` value → LoraConfig value (IMP-05.14)."""
    if spec == "all-linear":
        return "all-linear"
    return [part.strip() for part in spec.split(",") if part.strip()]


def _sft_train(
    *,
    base: str,
    samples: list[dict],
    out: Path,
    qlora: bool,
    r: int,
    alpha: int,
    target_modules: list[str] | str,
    epochs: int,
    max_seq_len: int,
    grad_accum: int,
    save_steps: int,
    lr: float,
    warmup_ratio: float,
    lr_scheduler: str,
    seed: int,
    run: common.Run,
) -> Path:
    """Fine-tune with TRL SFTTrainer + PEFT LoRA; returns the adapter dir.

    Seam: tests monkeypatch this to avoid torch. The real implementation is
    ported to the TRL ≥1.x API (IMP-05.1): `SFTConfig` (with `max_length=`)
    + `SFTTrainer(..., processing_class=tokenizer, train_dataset=Dataset)`;
    `tokenizer=`, `max_seq_length=`, `TrainingArguments`, and a plain `list`
    dataset all fail on the locked trl 1.13. Gradient checkpointing is on for
    both LoRA and QLoRA via `SFTConfig` (IMP-05.13); checkpoints are saved
    every `save_steps` steps and training resumes from the last checkpoint
    (IMP-05.12).
    """
    import torch  # pyright: ignore[reportMissingImports]
    from datasets import Dataset  # pyright: ignore[reportMissingImports]
    from peft import LoraConfig  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
    from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
    from trl import SFTConfig  # pyright: ignore[reportMissingImports]
    from trl import SFTTrainer  # pyright: ignore[reportMissingImports]

    model_kwargs: dict[str, Any] = {}
    if qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16  # IMP-05.16: torch_dtype= removed in transformers ≥4.56
    model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", **model_kwargs)
    # IMP-05.13: SFTConfig enables gradient checkpointing for both LoRA and
    # QLoRA; use_cache must be off while it is enabled.
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(base)
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        seed=seed,
        report_to=[],
        max_length=max_seq_len,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler,
    )
    if qlora:
        # IMP-05.14: 8-bit paged Adam keeps optimizer states off the GPU.
        cfg.optim = "paged_adamw_8bit"
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(samples),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(out))
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    run.log_line(f"adapter saved to {out}")
    return out


def _is_cuda_oom(exc: BaseException) -> bool:
    """Whether `exc` is a torch CUDA out-of-memory error (lazy torch import)."""
    try:
        import torch  # pyright: ignore[reportMissingImports]

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:  # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> int:
    """Run fine-tuning. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # IMP-05.15: validate arguments and data before start_run so usage
    # errors do not litter runs/.
    data_path = Path(args.data)
    try:
        rows = common.parse_jsonl(data_path)
        samples = format_samples(rows)
    except (OSError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE

    run = common.start_run(
        "finetune_lora",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")
    common.log(run, f"samples: {len(samples)}")

    from inference_hf import _load_config

    config = _load_config(args.base)
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

    out = args.out if args.out is not None else common.MODELS_DIR / f"{_slug(args.base)}-lora"
    out = out.resolve()
    common.log(run, f"training: base={args.base} out={out} qlora={args.qlora}")
    try:
        adapter = _sft_train(
            base=args.base,
            samples=samples,
            out=out,
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
        # IMP-05.15: surface a CUDA OOM with actionable hints.
        if _is_cuda_oom(exc):
            common.log(
                run,
                "CUDA out of memory during training — retry with "
                "--max-seq-len 1024, --qlora, or a larger --grad-accum "
                "(or --force if you know the card is free)",
            )
        common.log(run, f"training failed: {exc}")
        return common.EXIT_RUNTIME
    common.log(run, f"adapter: {adapter}")
    run.write_metrics(
        {
            "samples": len(samples),
            "out": str(adapter),
            "qlora": args.qlora,
            "epochs": args.epochs,
            "model_params_billions": common.model_params_billions(config),
            "seed": seed,
        }
    )
    common.print_peak_vram(run)
    if estimate is not None:
        common.record_estimate_vs_peak(run, estimate)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
