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

DEFAULT_R = 16
DEFAULT_ALPHA = 32
DEFAULT_EPOCHS = 1
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_GRAD_ACCUM = 4


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
    common.add_common_args(parser)
    return parser


def _slug(base: str) -> str:
    """Filesystem-safe slug of a repo id/path.

    e.g. 'Qwen/Qwen2.5-Coder-7B-Instruct' → 'qwen-qwen2.5-coder-7b-instruct'.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-.").lower() or "model"


def format_samples(rows: list[dict]) -> list[dict]:
    """Normalize JSONL rows into SFT samples (FR-05.7, pure helper).

    Chat rows (`{"messages": [{"role", "content"}, ...]}`) pass through;
    completion rows (`{"prompt", "completion"}`) become a user/assistant
    message pair so SFTTrainer can apply the tokenizer's chat template.
    Raises ValueError on malformed rows.
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


def _training_estimate_gib(params: float, qlora: bool, seq_len: int = 2048) -> float:
    """VRAM estimate for LoRA training (NFR-05.3, spec VRAM budget table).

    QLoRA: 4-bit base weights (0.55 GiB/B) + bf16 adapter weights, gradients,
    and optimizer states (1.5 GiB fixed + 0.36 GiB per B params) + 2.5 GiB
    framework overhead + an activation/KV term. Calibrated so 7B lands at
    ~10.9 GiB (allowed on 16 GB) and 14B at ~17.2 GiB (> free × 1.1 →
    refused, matching the table). Plain LoRA: bf16 weights + 0.4 GiB per B
    params for adapters/optimizer states (fits ≤ 3B per the table).
    """
    if qlora:
        base = params * common.BYTES_PER_PARAM["4bit"]
        adapters = 1.5 + 0.36 * params
    else:
        base = params * common.BYTES_PER_PARAM["bf16"]
        adapters = 0.4 * params
    return base + adapters + 2.5 + 0.5 * (seq_len / 2048)


def _sft_train(
    *,
    base: str,
    samples: list[dict],
    out: Path,
    qlora: bool,
    r: int,
    alpha: int,
    epochs: int,
    max_seq_len: int,
    grad_accum: int,
    seed: int,
    run: common.Run,
) -> Path:
    """Fine-tune with TRL SFTTrainer + PEFT LoRA; returns the adapter dir.

    Seam: tests monkeypatch this to avoid torch. The real implementation
    loads the base with bitsandbytes nf4 + double quant + bf16 compute dtype
    and enables gradient checkpointing when `qlora`, hands the conversational
    samples to SFTTrainer, and resumes from an existing checkpoint in `out`.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from peft import LoraConfig  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
    from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
    from transformers import TrainingArguments  # pyright: ignore[reportMissingImports]
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
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base)
    if qlora:
        model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=samples,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=max_seq_len,
        args=training_args,
    )
    trainer.train(resume_from_checkpoint=out.is_dir())
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    run.log_line(f"adapter saved to {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    """Run fine-tuning. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run = common.start_run(
        "finetune_lora",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    data_path = Path(args.data)
    try:
        rows = common.parse_jsonl(data_path)
    except (OSError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE
    try:
        samples = format_samples(rows)
    except ValueError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return common.EXIT_USAGE
    common.log(run, f"samples: {len(samples)}")

    from inference_hf import _load_config

    config = _load_config(args.base)
    params = common.model_params_billions(config)
    if params is None:
        print("model params unknown — VRAM guard skipped", file=sys.stderr)
    else:
        estimate = _training_estimate_gib(params, args.qlora, args.max_seq_len)
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
            epochs=args.epochs,
            max_seq_len=args.max_seq_len,
            grad_accum=args.grad_accum,
            seed=seed,
            run=run,
        )
    except Exception as exc:
        common.log(run, f"training failed: {exc}")
        return common.EXIT_RUNTIME
    common.log(run, f"adapter: {adapter}")
    run.write_metrics(
        {
            "samples": len(samples),
            "out": str(adapter),
            "qlora": args.qlora,
            "epochs": args.epochs,
            "model_params_billions": params,
            "seed": seed,
        }
    )
    common.print_peak_vram(run)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
