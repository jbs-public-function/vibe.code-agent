"""Compatibility checks against the locked TRL/transformers versions (IMP-05.1, IMP-05.16).

These are the "signature tests" from doc 05: they import the heavy packages
and assert the exact API the lab scripts call. Skipped in the torch-free root
environment (`pytest.importorskip("trl")`) and run in the lab venv/container
(`uv sync --project laboratory`), where they would have caught IMP-05.1
before any GPU time.

The `lab` marker is declared via `pytestmark` only (the root pyproject is out
of scope for this doc); the undeclared-marker warning is filtered below.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

warnings.filterwarnings(
    "ignore",
    message=r".*unknown pytest\.mark\.lab.*",
    category=pytest.PytestUnknownMarkWarning,
)

trl = pytest.importorskip("trl")  # pyright: ignore[reportMissingImports]

pytestmark = pytest.mark.lab


def test_sfttrainer_accepts_processing_class() -> None:
    """IMP-05.1: TRL ≥1.x renamed `tokenizer=` to `processing_class=`."""
    from trl import SFTTrainer  # pyright: ignore[reportMissingImports]

    sig = inspect.signature(SFTTrainer.__init__)
    assert "processing_class" in sig.parameters
    assert "tokenizer" not in sig.parameters
    assert "max_seq_length" not in sig.parameters


def test_sftconfig_accepts_max_length() -> None:
    """IMP-05.1: SFTConfig (not TrainingArguments) with `max_length=`."""
    from trl import SFTConfig  # pyright: ignore[reportMissingImports]

    cfg = SFTConfig(
        output_dir="/tmp/vibe-lab-compat",
        max_length=2048,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
    )
    assert cfg.max_length == 2048
    assert cfg.gradient_checkpointing is True
    assert cfg.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert cfg.save_strategy.value == "steps"


def test_automodelforcausallm_accepts_dtype() -> None:
    """IMP-05.16: transformers ≥4.56 replaced `torch_dtype=` with `dtype=`."""
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]

    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    assert "dtype" in sig.parameters


def test_locked_versions_satisfy_pins() -> None:
    """IMP-05.7: the lock resolves within the pinned major bounds."""
    import packaging.version
    import peft  # pyright: ignore[reportMissingImports]
    import transformers  # pyright: ignore[reportMissingImports]

    assert packaging.version.parse(transformers.__version__) >= packaging.version.parse("5.0")
    assert packaging.version.parse(transformers.__version__) < packaging.version.parse("6.0")
    assert packaging.version.parse(trl.__version__) >= packaging.version.parse("1.13")
    assert packaging.version.parse(trl.__version__) < packaging.version.parse("2.0")
    assert packaging.version.parse(peft.__version__) >= packaging.version.parse("0.21")
    assert packaging.version.parse(peft.__version__) < packaging.version.parse("1.0")
