"""End-to-end tests for the laboratory scripts (FR-05.4..FR-05.9, NFR-05.3, NFR-05.4).

Every heavy seam is monkeypatched — no test requires torch or a GPU. The lab
modules are imported from `laboratory/` via sys.path, mirroring how the
scripts run with `laboratory/` as cwd inside the container.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laboratory"))

import common  # noqa: E402  # pyright: ignore[reportMissingImports]
import distill  # noqa: E402  # pyright: ignore[reportMissingImports]
import download  # noqa: E402  # pyright: ignore[reportMissingImports]
import export  # noqa: E402  # pyright: ignore[reportMissingImports]
import finetune_lora  # noqa: E402  # pyright: ignore[reportMissingImports]
import inference_gguf  # noqa: E402  # pyright: ignore[reportMissingImports]
import inference_hf  # noqa: E402  # pyright: ignore[reportMissingImports]

QWEN_7B_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "vocab_size": 152064,
}  # ≈ 7.69B params
QWEN_1_5B_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "vocab_size": 151936,
}  # ≈ 1.65B params
QWEN_14B_CONFIG = {
    "num_hidden_layers": 48,
    "hidden_size": 4608,
    "intermediate_size": 14336,
    "vocab_size": 152064,
}  # ≈ 14.29B params


# -- helpers ----------------------------------------------------------------


def _single_run_dir(tmp_path: Path, script: str) -> Path:
    dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.endswith(f"-{script}")]
    assert len(dirs) == 1, f"expected one run dir for {script}, got {dirs}"
    return dirs[0]


def _metrics(tmp_path: Path, script: str) -> dict:
    path = _single_run_dir(tmp_path, script) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _guards_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Free VRAM 15.2 GiB and no model busy: both guards pass.

    IMP-05.2: an unreachable controller hard-fails the serving guard, so the
    status seam must report AVAILABLE rather than None.
    """
    monkeypatch.setattr(common, "free_vram_gib", lambda: 15.2)
    monkeypatch.setattr(
        common, "_vibe_status_json", lambda: '{"state": "AVAILABLE", "active_alias": ""}'
    )


def _prompt(i: int) -> dict:
    return {"prompt": f"prompt-{i}", "messages": [{"role": "user", "content": f"prompt-{i}"}]}


# -- torch-free import rule -------------------------------------------------


def test_no_torch_imported_by_scripts() -> None:
    """Importing every script must not pull in any heavy dependency."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "peft" not in sys.modules
    assert "trl" not in sys.modules
    assert "llama_cpp" not in sys.modules
    assert "awq" not in sys.modules


# -- download.py (FR-05.4) --------------------------------------------------


def test_download_build_parser() -> None:
    args = download.build_parser().parse_args(
        [
            "org/repo",
            "--include",
            "*.safetensors",
            "--include",
            "config.json",
            "--revision",
            "main",
            "--force",
            "--allow-coexist",
            "--run-dir",
            "/tmp/x",
        ]
    )
    assert args.repo == "org/repo"
    assert args.include == ["*.safetensors", "config.json"]
    assert args.revision == "main"
    assert args.force is True
    assert args.allow_coexist is True
    assert args.run_dir == Path("/tmp/x")


def test_download_main_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(download, "_snapshot_download", lambda repo, patterns, rev: f"/cache/{repo}")
    rc = download.main(["org/repo", "--run-dir", str(tmp_path)])
    assert rc == 0
    run_dir = _single_run_dir(tmp_path, "download")
    assert (run_dir / "args.json").is_file()
    assert (run_dir / "console.log").is_file()
    assert _metrics(tmp_path, "download")["snapshot"] == "/cache/org/repo"
    # IMP-05.9 prints the generated seed to stdout; the machine-readable
    # snapshot path stays the final stdout line.
    assert capsys.readouterr().out.splitlines()[-1] == "/cache/org/repo"


def test_download_ignores_serving_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IMP-05.11: download is a network-only op and skips the serving guard,
    even when a model is READY."""
    monkeypatch.setattr(common, "free_vram_gib", lambda: 15.2)
    monkeypatch.setattr(
        common,
        "_vibe_status_json",
        lambda: '{"state": "READY", "active_alias": "x"}',
    )
    monkeypatch.setattr(download, "_snapshot_download", lambda *args: "/cache/org/repo")
    rc = download.main(["org/repo", "--run-dir", str(tmp_path / "r1")])
    assert rc == 0
    assert _metrics(tmp_path / "r1", "download")["snapshot"] == "/cache/org/repo"


def test_download_argparse_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        download.main(["org/repo", "--bogus"])
    assert isinstance(excinfo.value, SystemExit)
    assert excinfo.value.code == 2


def test_snapshot_download_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    calls: list[dict] = []

    def fake_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "/tmp/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.delenv("VIBE_DATA_DIR", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert download._snapshot_download("org/repo", ["*.bin"], "rev1") == "/tmp/snap"
    assert calls[-1] == {
        "repo_id": "org/repo",
        "allow_patterns": ["*.bin"],
        "revision": "rev1",
        "token": None,
    }
    monkeypatch.setenv("VIBE_DATA_DIR", "/data/vibe")
    download._snapshot_download("org/repo", None, None)
    # IMP-05.5: VIBE_DATA_DIR is HF_HOME; snapshots land under <dir>/hub.
    assert calls[-1]["cache_dir"] == "/data/vibe/hub"


# -- inference_hf.py (FR-05.5) ----------------------------------------------


def test_inference_hf_config_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(QWEN_7B_CONFIG), encoding="utf-8")
    assert inference_hf._load_config(str(model_dir)) == QWEN_7B_CONFIG


def test_inference_hf_config_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import huggingface_hub

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(QWEN_7B_CONFIG), encoding="utf-8")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: str(config_file),
    )
    assert inference_hf._load_config("org/repo") == QWEN_7B_CONFIG


def test_inference_hf_config_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def boom(**kwargs: Any) -> str:
        raise RuntimeError("no network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    assert inference_hf._load_config("org/nonexistent") is None
    assert inference_hf._load_config("/no/such/dir") is None


def test_choose_precision() -> None:
    assert inference_hf._choose_precision(1.65, None, False) == ("bf16", {"dtype": "bf16"})
    assert inference_hf._choose_precision(7.69, None, False) == ("4bit", {"load_in_4bit": True})
    assert inference_hf._choose_precision(7.69, "fp16", False) == ("fp16", {"dtype": "fp16"})
    assert inference_hf._choose_precision(1.65, None, True) == ("4bit", {"load_in_4bit": True})
    assert inference_hf._choose_precision(None, None, False) == ("bf16", {"dtype": "bf16"})


def test_inference_hf_main_auto_4bit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_7B_CONFIG))
    loaded: list[tuple[str, dict]] = []

    def fake_load(model: str, kwargs: dict) -> tuple[SimpleNamespace, SimpleNamespace]:
        loaded.append((model, kwargs))
        return SimpleNamespace(), SimpleNamespace()

    def fake_stream(model: Any, tokenizer: Any, prompt: str, **gen_kwargs: Any) -> Any:
        yield "hel"
        yield "lo"

    monkeypatch.setattr(inference_hf, "_load_pipeline", fake_load)
    monkeypatch.setattr(inference_hf, "_stream_tokens", fake_stream)
    rc = inference_hf.main(
        ["--model", "org/7b", "--prompt", "hi", "--run-dir", str(tmp_path)]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert "auto-selecting 4-bit quantization for a >3B model" in captured.err
    assert loaded[0] == ("org/7b", {"load_in_4bit": True})
    metrics = _metrics(tmp_path, "inference_hf")
    assert metrics["precision"] == "4bit"
    assert metrics["tokens"] == 2
    assert metrics["tokens_per_sec"] > 0
    assert metrics["model_params_billions"] == pytest.approx(7.686848512)


def test_inference_hf_refuses_fp16_7b_then_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_7B_CONFIG))
    loaded: list[str] = []

    def fake_load(model: str, kwargs: dict) -> tuple[SimpleNamespace, SimpleNamespace]:
        loaded.append(model)
        return SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(inference_hf, "_load_pipeline", fake_load)
    monkeypatch.setattr(inference_hf, "_stream_tokens", lambda *a, **k: iter(()))

    # 7B fp16 → 7.69*2.0 + 2.5 + 0.5 ≈ 18.4 GiB vs 15.2 free → refused.
    rc = inference_hf.main(
        ["--model", "org/7b", "--dtype", "fp16", "--prompt", "hi", "--run-dir", str(tmp_path / "r1")]
    )
    assert rc == 1
    assert loaded == []
    assert "exceeds free VRAM" in capsys.readouterr().err
    assert (_single_run_dir(tmp_path / "r1", "inference_hf") / "console.log").is_file()

    rc = inference_hf.main(
        ["--model", "org/7b", "--dtype", "fp16", "--prompt", "hi", "--force", "--run-dir", str(tmp_path / "r2")]
    )
    assert rc == 0
    assert loaded == ["org/7b"]


def test_inference_hf_usage_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        inference_hf.main(["--prompt", "hi", "--run-dir", str(tmp_path)])
    assert isinstance(excinfo.value, SystemExit)
    assert excinfo.value.code == 2  # missing --model
    assert inference_hf.main(["--model", "x", "--run-dir", str(tmp_path)]) == 2
    assert (
        inference_hf.main(["--model", "x", "--prompt", "hi", "--chat", "--run-dir", str(tmp_path)])
        == 2
    )
    # IMP-05.28: --load-in-4bit + --dtype is contradictory and rejected.
    assert (
        inference_hf.main(
            ["--model", "x", "--prompt", "hi", "--load-in-4bit", "--dtype", "fp16",
             "--run-dir", str(tmp_path)]
        )
        == 2
    )


class _FakeTokenizer:
    """Tokenizer stand-in with a chat template and a callable encode path."""

    def __init__(self, input_len: int = 3) -> None:
        self.chat_template = "{{ messages }}"
        self._input_len = input_len
        self.template_calls: list[list[dict]] = []

    def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> str:
        self.template_calls.append(messages)
        return "<template>" + messages[-1]["content"]

    def __call__(self, prompt: str, **kwargs: Any) -> dict:
        return {"input_ids": [[0] * self._input_len]}


def test_apply_chat_template() -> None:
    """IMP-05.28: chat templates are applied unless --raw or absent."""
    tokenizer = _FakeTokenizer()
    assert inference_hf._apply_chat_template(tokenizer, "hi", raw=False) == "<template>hi"
    assert inference_hf._apply_chat_template(tokenizer, "hi", raw=True) == "hi"
    plain = SimpleNamespace()
    assert inference_hf._apply_chat_template(plain, "hi", raw=False) == "hi"


def test_count_output_tokens_from_ids() -> None:
    """IMP-05.28: tokens/s uses len(output_ids) minus the prompt, not chunks."""
    tokenizer = _FakeTokenizer(input_len=2)
    output_ids = [0, 0, 101, 102, 103]  # 5 ids, 3 generated
    assert inference_hf._count_output_tokens(tokenizer, "hi", output_ids, chunk_count=9) == 3
    # No ids available (monkeypatched seam) → chunk count is the fallback.
    assert inference_hf._count_output_tokens(tokenizer, "hi", None, chunk_count=9) == 9


# -- inference_gguf.py (FR-05.6) --------------------------------------------


def test_gguf_resolve_local(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    assert inference_gguf._resolve_gguf(str(gguf)) == gguf.resolve()


def test_gguf_resolve_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import huggingface_hub

    target = tmp_path / "cache" / "model.gguf"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(target))
    assert inference_gguf._resolve_gguf("org/repo:model.gguf") == target


def test_gguf_resolve_invalid() -> None:
    with pytest.raises(ValueError):
        inference_gguf._resolve_gguf("/no/such/file.gguf")
    with pytest.raises(ValueError):
        inference_gguf._resolve_gguf("org/repo:")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("qwen2.5-coder-7b-instruct-q4_k_m.gguf", 7.0),
        ("Qwen2.5-7B-Instruct-Q4_K_M.gguf", 7.0),
        ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 8.0),
        ("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1.5),
        ("model.gguf", None),
    ],
)
def test_gguf_guess_params(name: str, expected: float | None) -> None:
    assert inference_gguf._guess_params_from_name(name) == expected


def test_gguf_bytes_per_param_by_backend() -> None:
    """IMP-05.27: transformers dequantizes GGUF to fp16 on load."""
    assert inference_gguf._bytes_per_param("llamacpp") == pytest.approx(0.55)
    assert inference_gguf._bytes_per_param("transformers") == pytest.approx(2.0)


def test_gguf_main_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _guards_ok(monkeypatch)
    gguf_file = tmp_path / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    gguf_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(inference_gguf, "_resolve_gguf", lambda spec: gguf_file)
    loaded: list[tuple] = []

    def fake_load(path: Path, backend: str, n_gpu_layers: int, n_ctx: int = 2048) -> tuple[SimpleNamespace, SimpleNamespace]:
        loaded.append((backend, n_gpu_layers, n_ctx))
        return SimpleNamespace(), SimpleNamespace()

    def fake_stream(model: Any, tokenizer: Any, prompt: str, backend: str, **kwargs: Any) -> Any:
        yield "a"
        yield "b"

    monkeypatch.setattr(inference_gguf, "_load_gguf_model", fake_load)
    monkeypatch.setattr(inference_gguf, "_stream_tokens", fake_stream)
    rc = inference_gguf.main(
        ["--gguf", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "--prompt", "hi",
         "--n-ctx", "4096", "--run-dir", str(tmp_path)]
    )
    assert rc == 0
    assert "ab" in capsys.readouterr().out
    # IMP-05.27: the default backend is llamacpp; --n-ctx is forwarded.
    assert loaded == [("llamacpp", -1, 4096)]
    metrics = _metrics(tmp_path, "inference_gguf")
    assert metrics["precision"] == "4bit"
    assert metrics["backend"] == "llamacpp"
    assert metrics["model_params_billions"] == 7.0


def test_gguf_transformers_backend_refused_7b(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """IMP-05.27: the transformers backend estimates fp16 (7B → ~17 GiB)."""
    _guards_ok(monkeypatch)
    gguf_file = tmp_path / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    gguf_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(inference_gguf, "_resolve_gguf", lambda spec: gguf_file)
    loaded: list[tuple] = []

    def fake_load(path: Path, backend: str, n_gpu_layers: int, n_ctx: int = 2048) -> tuple[SimpleNamespace, SimpleNamespace]:
        loaded.append((backend, n_ctx))
        return SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(inference_gguf, "_load_gguf_model", fake_load)
    monkeypatch.setattr(inference_gguf, "_stream_tokens", lambda *a, **k: iter(()))
    rc = inference_gguf.main(
        ["--gguf", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "--backend", "transformers",
         "--prompt", "hi", "--run-dir", str(tmp_path / "r1")]
    )
    assert rc == 1
    assert loaded == []
    assert "exceeds free VRAM" in capsys.readouterr().err
    rc = inference_gguf.main(
        ["--gguf", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "--backend", "transformers",
         "--prompt", "hi", "--force", "--run-dir", str(tmp_path / "r2")]
    )
    assert rc == 0


def test_gguf_llamacpp_missing_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IMP-05.27: a missing llama_cpp import points at the llamacpp extra."""
    gguf_file = tmp_path / "m.gguf"
    gguf_file.write_text("x", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(RuntimeError, match="llamacpp.*extra"):
        inference_gguf._load_gguf_model(gguf_file, "llamacpp", -1)


def test_gguf_resolution_error_exit_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(spec: str) -> Path:
        raise ValueError("bad spec")

    monkeypatch.setattr(inference_gguf, "_resolve_gguf", boom)
    rc = inference_gguf.main(["--gguf", "x:y", "--prompt", "hi", "--run-dir", str(tmp_path)])
    assert rc == 2


def test_gguf_argparse_bad_backend() -> None:
    with pytest.raises(SystemExit) as excinfo:
        inference_gguf.main(["--gguf", "f.gguf", "--backend", "nope", "--prompt", "hi"])
    assert isinstance(excinfo.value, SystemExit)
    assert excinfo.value.code == 2


# -- finetune_lora.py (FR-05.7) ---------------------------------------------


def test_format_samples_chat() -> None:
    rows = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    assert finetune_lora.format_samples(rows) == rows


def test_format_samples_completion() -> None:
    assert finetune_lora.format_samples([{"prompt": "hi", "completion": "hello"}]) == [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]


def test_format_samples_errors() -> None:
    with pytest.raises(ValueError, match="row 0"):
        finetune_lora.format_samples([{"messages": []}])
    with pytest.raises(ValueError, match="row 0"):
        finetune_lora.format_samples([{"messages": [{"role": "user"}]}])
    with pytest.raises(ValueError, match="row 1"):
        finetune_lora.format_samples([{"prompt": "ok", "completion": "ok"}, {"nope": 1}])


def test_format_samples_messages_plus_completion() -> None:
    """IMP-05.21: distill teacher rows (messages + completion) keep the chat
    form with the assistant completion appended."""
    rows = [
        {
            "idx": 0,
            "prompt": "again",
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
            ],
            "completion": "hello",
        }
    ]
    assert finetune_lora.format_samples(rows) == [
        {
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]


def test_get_last_checkpoint() -> None:
    tmp = Path("/nonexistent")
    assert finetune_lora.get_last_checkpoint(tmp) is None


def test_get_last_checkpoint_empty_dir(tmp_path: Path) -> None:
    """IMP-05.12: an existing dir without checkpoint-* → None, never ValueError."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert finetune_lora.get_last_checkpoint(tmp_path) is None


def test_get_last_checkpoint_highest(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-4").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-garbage").mkdir()
    assert finetune_lora.get_last_checkpoint(tmp_path) == tmp_path / "checkpoint-10"


def test_training_estimate_pinned() -> None:
    """Numbers pinned to the IMP-05.10 config-derived model + the spec table.

    Adapter/optimizer = r × Σ(in+out) × 12 B; activations = layers × seq ×
    hidden × 68 B × 0.4 (grad-ckpt factor); + 2.5 GiB framework.
    """
    # 7B QLoRA: 7.687*0.55 + 16*2,220,032*12/2^30 + 28*2048*3584*68*0.4/2^30 + 2.5 ≈ 12.33.
    assert finetune_lora._training_estimate_gib(QWEN_7B_CONFIG, True) == pytest.approx(
        12.33, abs=0.02
    )
    # 14B QLoRA ≈ 22.5 → refused (> 15.2*1.1).
    assert finetune_lora._training_estimate_gib(QWEN_14B_CONFIG, True) == pytest.approx(
        22.5, abs=0.05
    )
    # 1.5B LoRA ≈ 8.23 → allowed (spec table: 8–12 GiB).
    assert finetune_lora._training_estimate_gib(QWEN_1_5B_CONFIG, False) == pytest.approx(
        8.23, abs=0.03
    )
    # 7B plain LoRA ≈ 23.5 → refused.
    assert finetune_lora._training_estimate_gib(QWEN_7B_CONFIG, False) > 20
    assert finetune_lora._training_estimate_gib(None, True) is None


def test_legacy_training_estimate_fallback() -> None:
    """IMP-05.10: the back-fitted constants remain the documented fallback."""
    # 7B QLoRA: 7*0.55 + 1.5 + 0.36*7 + 2.5 + 0.5 = 10.87 (spec table).
    assert finetune_lora._legacy_training_estimate_gib(7.0, True) == pytest.approx(10.87)
    # 1.5B LoRA: 1.5*2.0 + 0.4*1.5 + 3.0 = 6.6.
    assert finetune_lora._legacy_training_estimate_gib(1.5, False) == pytest.approx(6.6)


def test_finetune_main_qlora_7b(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_7B_CONFIG))
    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt": "hi", "completion": "hello"}\n\n', encoding="utf-8")
    train_calls: list[dict] = []

    def fake_train(**kwargs: Any) -> Path:
        train_calls.append(kwargs)
        out: Path = kwargs["out"]
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(finetune_lora, "_sft_train", fake_train)
    out = tmp_path / "adapter"
    rc = finetune_lora.main(
        ["--base", "org/7b", "--data", str(data), "--qlora", "--out", str(out), "--run-dir", str(tmp_path)]
    )
    assert rc == 0
    assert train_calls[0]["qlora"] is True
    assert train_calls[0]["out"] == out.resolve()
    assert train_calls[0]["samples"] == [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    ]
    # IMP-05.14/05.17: training flags are forwarded to _sft_train.
    assert train_calls[0]["r"] == 16
    assert train_calls[0]["alpha"] == 32
    assert train_calls[0]["target_modules"] == "all-linear"
    assert train_calls[0]["lr"] == 2e-4
    assert train_calls[0]["save_steps"] == 100
    metrics = _metrics(tmp_path, "finetune_lora")
    assert metrics["samples"] == 1
    assert metrics["qlora"] is True
    assert metrics["estimate_vram_gib"] == pytest.approx(12.33, abs=0.02)


def test_finetune_qlora_14b_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_14B_CONFIG))
    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt": "hi", "completion": "hello"}\n', encoding="utf-8")
    train_calls: list[dict] = []
    monkeypatch.setattr(
        finetune_lora,
        "_sft_train",
        lambda **kw: train_calls.append(kw) or kw["out"],
    )
    rc = finetune_lora.main(
        ["--base", "org/14b", "--data", str(data), "--qlora", "--out", str(tmp_path / "adapter"), "--run-dir", str(tmp_path)]
    )
    assert rc == 1
    assert train_calls == []
    assert "exceeds free VRAM" in capsys.readouterr().err


def test_finetune_default_out_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt": "hi", "completion": "hello"}\n', encoding="utf-8")
    train_calls: list[dict] = []
    monkeypatch.setattr(
        finetune_lora,
        "_sft_train",
        lambda **kw: train_calls.append(kw) or kw["out"],
    )
    rc = finetune_lora.main(
        ["--base", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "--data", str(data), "--run-dir", str(tmp_path)]
    )
    assert rc == 0
    assert train_calls[0]["out"] == common.MODELS_DIR / "qwen-qwen2.5-coder-1.5b-instruct-lora"


def test_finetune_bad_data_usage_error(tmp_path: Path) -> None:
    """IMP-05.15: a usage error exits before start_run — no runs/ litter."""
    data = tmp_path / "bad.jsonl"
    data.write_text("{not json}\n", encoding="utf-8")
    rc = finetune_lora.main(["--base", "org/7b", "--data", str(data), "--run-dir", str(tmp_path)])
    assert rc == 2
    assert not any(p.is_dir() for p in tmp_path.iterdir())


def test_finetune_argparse_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        finetune_lora.main(["--base", "org/7b"])
    assert isinstance(excinfo.value, SystemExit)
    assert excinfo.value.code == 2  # missing --data


# -- distill.py (FR-05.8) ---------------------------------------------------


class FakeResponse:
    """httpx.Response stand-in with a canned chat-completion body."""

    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class FakeAsyncClient:
    """httpx.AsyncClient stand-in recording calls, failures, and concurrency."""

    calls: list[int] = []
    concurrency = 0
    max_concurrency = 0
    fails_left: dict[int, int] = {}
    error_type: type[httpx.HTTPError] = httpx.ConnectError

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        payload = kwargs["json"]
        idx = int(payload["messages"][0]["content"].split("-")[1])
        FakeAsyncClient.calls.append(idx)
        FakeAsyncClient.concurrency += 1
        FakeAsyncClient.max_concurrency = max(FakeAsyncClient.max_concurrency, FakeAsyncClient.concurrency)
        try:
            await asyncio.sleep(0)
            if FakeAsyncClient.fails_left.get(idx, 0) > 0:
                FakeAsyncClient.fails_left[idx] -= 1
                raise FakeAsyncClient.error_type(f"boom {idx}")
            return FakeResponse(f"completion-{idx}")
        finally:
            FakeAsyncClient.concurrency -= 1


@pytest.fixture
def fake_async_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeAsyncClient]:
    FakeAsyncClient.calls = []
    FakeAsyncClient.concurrency = 0
    FakeAsyncClient.max_concurrency = 0
    FakeAsyncClient.fails_left = {}
    FakeAsyncClient.error_type = httpx.ConnectError
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return FakeAsyncClient


def test_format_prompts() -> None:
    assert distill.format_prompts([{"prompt": "hi"}]) == [
        {"prompt": "hi", "messages": [{"role": "user", "content": "hi"}]}
    ]
    rows = [
        {
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ]
        }
    ]
    assert distill.format_prompts(rows)[0]["prompt"] == "again"
    with pytest.raises(ValueError, match="row 0"):
        distill.format_prompts([{"messages": [{"role": "assistant", "content": "x"}]}])
    with pytest.raises(ValueError, match="row 1"):
        distill.format_prompts([{"prompt": "ok"}, {"nope": 1}])


async def test_generate_teacher_concurrency(fake_async_client: type[FakeAsyncClient], tmp_path: Path) -> None:
    out_cache = tmp_path / "teacher.jsonl"
    written = await distill._generate_teacher("teacher", [_prompt(i) for i in range(6)], out_cache, 0.7, None)
    assert written == 6
    assert FakeAsyncClient.max_concurrency == 4  # asyncio.Semaphore(4)
    rows = common.parse_jsonl(out_cache)
    assert [r["idx"] for r in rows] == list(range(6))
    # IMP-05.21: rows keep the full messages the teacher saw + the completion.
    assert rows[0] == {
        "idx": 0,
        "prompt": "prompt-0",
        "messages": [{"role": "user", "content": "prompt-0"}],
        "completion": "completion-0",
    }


async def test_generate_teacher_max_tokens_in_payload(monkeypatch: pytest.MonkeyPatch, fake_async_client: type[FakeAsyncClient], tmp_path: Path) -> None:
    """IMP-05.18: --max-tokens is sent in the request payload."""
    payloads: list[dict] = []
    original_post = FakeAsyncClient.post

    async def recording_post(self: FakeAsyncClient, url: str, **kwargs: Any) -> FakeResponse:
        payloads.append(kwargs["json"])
        return await original_post(self, url, **kwargs)

    # monkeypatch.setattr on the class restores `post` after the test.
    monkeypatch.setattr(FakeAsyncClient, "post", recording_post)
    out_cache = tmp_path / "teacher.jsonl"
    await distill._generate_teacher(
        "t", [_prompt(0)], out_cache, 0.7, None, timeout=123.0, max_tokens=512, concurrency=1
    )
    assert payloads[0]["max_tokens"] == 512


async def test_generate_teacher_resumable(fake_async_client: type[FakeAsyncClient], tmp_path: Path) -> None:
    out_cache = tmp_path / "teacher.jsonl"
    out_cache.write_text(
        json.dumps({"idx": 0, "prompt": "prompt-0", "completion": "old-0"}) + "\n"
        + json.dumps({"idx": 1, "prompt": "prompt-1", "completion": "old-1"}) + "\n",
        encoding="utf-8",
    )
    written = await distill._generate_teacher("t", [_prompt(i) for i in range(4)], out_cache, 0.7, None)
    assert written == 2
    assert FakeAsyncClient.calls == [2, 3]
    rows = common.parse_jsonl(out_cache)
    assert [r["idx"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["completion"] == "old-0"


async def test_generate_teacher_n_limit(fake_async_client: type[FakeAsyncClient], tmp_path: Path) -> None:
    written = await distill._generate_teacher("t", [_prompt(i) for i in range(5)], tmp_path / "c.jsonl", 0.7, 2)
    assert written == 2
    assert FakeAsyncClient.calls == [0, 1]


async def test_generate_teacher_retry_once(fake_async_client: type[FakeAsyncClient], tmp_path: Path) -> None:
    FakeAsyncClient.fails_left = {1: 1}
    written = await distill._generate_teacher("t", [_prompt(i) for i in range(3)], tmp_path / "c.jsonl", 0.7, None)
    assert written == 3
    # Concurrent scheduling makes the exact order nondeterministic; idx 1 is
    # attempted twice (fail → retry) while 0 and 2 succeed once.
    assert sorted(FakeAsyncClient.calls) == [0, 1, 1, 2]
    assert FakeAsyncClient.calls.count(1) == 2


async def test_generate_teacher_timeout_retry(fake_async_client: type[FakeAsyncClient], tmp_path: Path) -> None:
    """IMP-05.18/05.20: a teacher timeout (httpx.ReadTimeout) is retried once."""
    FakeAsyncClient.error_type = httpx.ReadTimeout
    FakeAsyncClient.fails_left = {0: 1}
    written = await distill._generate_teacher("t", [_prompt(i) for i in range(2)], tmp_path / "c.jsonl", 0.7, None)
    assert written == 2
    assert FakeAsyncClient.calls.count(0) == 2


async def test_generate_teacher_abort_after_retry(fake_async_client: type[FakeAsyncClient], tmp_path: Path) -> None:
    FakeAsyncClient.fails_left = {0: 3}
    with pytest.raises(RuntimeError, match="vibe load"):
        await distill._generate_teacher("t", [_prompt(i) for i in range(2)], tmp_path / "c.jsonl", 0.7, None)


def test_distill_phase1_only_instruction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")

    async def fake_generate(*args: Any, **kwargs: Any) -> int:
        return 3

    monkeypatch.setattr(distill, "_generate_teacher", fake_generate)
    run_cli_calls: list[list[str]] = []
    monkeypatch.setattr(distill, "_run_cli", lambda argv: run_cli_calls.append(argv) or 0)
    train_calls: list[dict] = []
    monkeypatch.setattr(finetune_lora, "_sft_train", lambda **kw: train_calls.append(kw) or kw["out"])
    rc = distill.main(
        ["--teacher-alias", "t", "--student", "org/student", "--prompts", str(prompts), "--run-dir", str(tmp_path)]
    )
    assert rc == 0
    assert run_cli_calls == []
    assert train_calls == []
    assert "teacher generation done" in capsys.readouterr().err
    assert _metrics(tmp_path, "distill")["teacher_completions"] == 3


def test_distill_train_only_corrupt_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IMP-05.20: a corrupt teacher-cache line is a clean line-number error."""
    monkeypatch.setattr(common, "free_vram_gib", lambda: 15.2)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    cache = tmp_path / "data" / "org-student.teacher.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text('{"idx": 0, "prompt": "hi", "completion": "ok"}\n{not json}\n', encoding="utf-8")
    rc = distill.main(
        ["--teacher-alias", "t", "--student", "org/student", "--prompts", str(prompts),
         "--train-only", "--run-dir", str(tmp_path)]
    )
    assert rc == 2


def test_distill_run_cli_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMP-05.20: a missing `vibe` CLI is a clean RuntimeError, not a traceback."""

    def boom(argv: list[str], **kwargs: object) -> int:
        raise FileNotFoundError("vibe")

    monkeypatch.setattr(distill.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="vibe.*CLI"):
        distill._run_cli(["vibe", "unload"])


def test_distill_run_cli_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(argv: list[str], **kwargs: object) -> int:
        raise distill.subprocess.TimeoutExpired(argv, 300.0)

    monkeypatch.setattr(distill.subprocess, "run", hang)
    with pytest.raises(RuntimeError, match="timed out"):
        distill._run_cli(["vibe", "unload"])


def test_distill_flag_passthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IMP-05.17: distill forwards the shared training flags to _sft_train."""
    _guards_ok(monkeypatch)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    cache = data_dir / "org-student.teacher.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"idx": 0, "prompt": "hi", "completion": "bye"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    train_calls: list[dict] = []
    monkeypatch.setattr(finetune_lora, "_sft_train", lambda **kw: train_calls.append(kw) or kw["out"])
    rc = distill.main(
        [
            "--teacher-alias", "t", "--student", "org/student", "--prompts", str(prompts),
            "--train-only", "--qlora", "--r", "8", "--alpha", "16", "--epochs", "2",
            "--max-seq-len", "1024", "--grad-accum", "2", "--lr", "1e-4",
            "--save-steps", "50", "--target-modules", "q_proj,v_proj",
            "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    assert train_calls[0]["qlora"] is True
    assert train_calls[0]["r"] == 8
    assert train_calls[0]["alpha"] == 16
    assert train_calls[0]["epochs"] == 2
    assert train_calls[0]["max_seq_len"] == 1024
    assert train_calls[0]["grad_accum"] == 2
    assert train_calls[0]["lr"] == 1e-4
    assert train_calls[0]["save_steps"] == 50
    assert train_calls[0]["target_modules"] == ["q_proj", "v_proj"]


def test_distill_train_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    cache = data_dir / "org-student.teacher.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"idx": 0, "prompt": "hi", "completion": "bye"}) + "\n", encoding="utf-8")

    gen_calls: list[int] = []

    async def fake_generate(*args: Any, **kwargs: Any) -> int:
        gen_calls.append(1)
        return 0

    monkeypatch.setattr(distill, "_generate_teacher", fake_generate)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    train_calls: list[dict] = []

    def fake_train(**kwargs: Any) -> Path:
        train_calls.append(kwargs)
        out: Path = kwargs["out"]
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(finetune_lora, "_sft_train", fake_train)
    out = tmp_path / "out"
    rc = distill.main(
        [
            "--teacher-alias", "t", "--student", "org/student", "--prompts", str(prompts),
            "--train-only", "--out", str(out), "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    assert gen_calls == []
    assert train_calls[0]["base"] == "org/student"
    assert train_calls[0]["samples"] == [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]}
    ]


def test_distill_train_only_empty_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "data")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    rc = distill.main(
        ["--teacher-alias", "t", "--student", "org/student", "--prompts", str(prompts), "--train-only", "--run-dir", str(tmp_path)]
    )
    assert rc == 2


def test_distill_auto_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")

    async def fake_generate(*args: Any, **kwargs: Any) -> int:
        out_cache: Path = args[2]
        out_cache.parent.mkdir(parents=True, exist_ok=True)
        out_cache.write_text(json.dumps({"idx": 0, "prompt": "hi", "completion": "bye"}) + "\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(distill, "_generate_teacher", fake_generate)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    run_cli_calls: list[list[str]] = []
    monkeypatch.setattr(distill, "_run_cli", lambda argv: run_cli_calls.append(argv) or 0)
    # IMP-05.19: after `vibe unload` the script polls until AVAILABLE.
    poll_calls: list[float] = []

    def fake_poll(timeout_s: float = distill.AVAIL_POLL_TIMEOUT) -> bool:
        poll_calls.append(timeout_s)
        return True

    monkeypatch.setattr(distill, "_poll_until_available", fake_poll)
    train_calls: list[dict] = []

    def fake_train(**kwargs: Any) -> Path:
        train_calls.append(kwargs)
        out: Path = kwargs["out"]
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(finetune_lora, "_sft_train", fake_train)
    rc = distill.main(
        [
            "--teacher-alias", "t", "--student", "org/student", "--prompts", str(prompts),
            "--auto-swap", "--out", str(tmp_path / "out"), "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    assert run_cli_calls == [["vibe", "unload"]]
    assert poll_calls == [distill.AVAIL_POLL_TIMEOUT]
    assert len(train_calls) == 1


def test_poll_until_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMP-05.19: AVAILABLE must persist over two polls with stable VRAM."""
    states = ["UNLOADING", "AVAILABLE", "AVAILABLE"]
    frees = [10.0, 9.6, 9.55]

    monkeypatch.setattr(
        common, "status_snapshot", lambda: {"state": states.pop(0) if states else "AVAILABLE"}
    )
    monkeypatch.setattr(common, "free_vram_gib", lambda: frees.pop(0) if frees else 9.55)
    monkeypatch.setattr(distill.time, "sleep", lambda _s: None)
    monkeypatch.setattr(distill.time, "monotonic", lambda: 0.0)
    # The third poll (AVAILABLE, 9.55 vs 9.6 within 0.1 GiB) → stable.
    assert distill._poll_until_available() is True


def test_poll_until_available_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "status_snapshot", lambda: {"state": "UNLOADING"})
    monkeypatch.setattr(common, "free_vram_gib", lambda: None)
    monkeypatch.setattr(distill.time, "sleep", lambda _s: None)
    clock = {"n": 0}

    def fake_monotonic() -> float:
        clock["n"] += 1
        return 0.0 if clock["n"] < 3 else 1000.0

    monkeypatch.setattr(distill.time, "monotonic", fake_monotonic)
    assert distill._poll_until_available() is False


def test_distill_serving_guard_before_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "free_vram_gib", lambda: 15.2)
    monkeypatch.setattr(common, "_vibe_status_json", lambda: '{"state": "READY", "active_alias": "t"}')
    data_dir = tmp_path / "data"
    monkeypatch.setattr(common, "DATA_DIR", data_dir)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    cache = data_dir / "org-student.teacher.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"idx": 0, "prompt": "hi", "completion": "bye"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    train_calls: list[dict] = []
    monkeypatch.setattr(finetune_lora, "_sft_train", lambda **kw: train_calls.append(kw) or kw["out"])
    rc = distill.main(
        ["--teacher-alias", "t", "--student", "org/student", "--prompts", str(prompts), "--train-only", "--run-dir", str(tmp_path)]
    )
    assert rc == 1
    assert train_calls == []


# -- export.py (FR-05.9) ----------------------------------------------------


def _make_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    return adapter


def _write_exported_model(out: Path) -> None:
    """A minimal exported dir passing `_verify_export` (IMP-05.26)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )


def test_export_merge_and_register(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    adapter = _make_adapter(tmp_path)
    merge_calls: list[tuple[Path, str, Path]] = []

    def fake_merge(adapter_path: Path, base: str, out: Path) -> Path:
        merge_calls.append((adapter_path, base, out))
        _write_exported_model(out)
        return out

    monkeypatch.setattr(export, "_merge_lora", fake_merge)
    run_cli_calls: list[list[str]] = []
    monkeypatch.setattr(export, "_run_cli", lambda argv: run_cli_calls.append(argv) or 0)
    out = tmp_path / "out-model"
    rc = export.main(
        [
            "--adapter", str(adapter), "--base", "org/base", "--out", str(out),
            "--register", "--alias", "my-alias", "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    assert merge_calls == [(adapter.resolve(), "org/base", out.resolve())]
    assert run_cli_calls == [["vibe", "register", str(out.resolve()), "--alias", "my-alias"]]
    assert _metrics(tmp_path, "export")["awq"] is False


def test_export_awq_quantize_seam(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    adapter = _make_adapter(tmp_path)

    def fake_merge(adapter_path: Path, base: str, out: Path) -> Path:
        _write_exported_model(out)
        return out

    monkeypatch.setattr(export, "_merge_lora", fake_merge)
    awq_calls: list[tuple] = []

    def fake_awq(model_dir: Path, out: Path, calibration: Path | None, n: int = 128) -> Path:
        awq_calls.append((model_dir, out, calibration, n))
        _write_exported_model(out)
        return out

    monkeypatch.setattr(export, "_awq_quantize", fake_awq)
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"prompt": "p1"}\n{"prompt": "p2"}\n', encoding="utf-8")
    out = tmp_path / "out-model"
    rc = export.main(
        [
            "--adapter", str(adapter), "--base", "org/base", "--out", str(out),
            "--awq", "--calibration", str(calib), "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    # IMP-05.24: merge to a staging <out>.bf16 dir, quantize into --out,
    # and leave no bf16 shards behind.
    bf16_dir = out.parent / f"{out.name}.bf16"
    assert awq_calls == [(bf16_dir.resolve(), out.resolve(), calib, 128)]
    assert not bf16_dir.exists()
    assert _metrics(tmp_path, "export")["awq"] is True


def test_export_register_awq_and_tool_parser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IMP-05.26: --register passes --quantization awq and --tool-parser."""
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    adapter = _make_adapter(tmp_path)
    monkeypatch.setattr(
        export, "_merge_lora", lambda a, b, out: _write_exported_model(out) or out
    )
    monkeypatch.setattr(
        export, "_awq_quantize", lambda model_dir, out, calib, n=128: _write_exported_model(out) or out
    )
    run_cli_calls: list[list[str]] = []
    monkeypatch.setattr(export, "_run_cli", lambda argv: run_cli_calls.append(argv) or 0)
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"prompt": "p"}\n', encoding="utf-8")
    out = tmp_path / "out-model"
    rc = export.main(
        [
            "--adapter", str(adapter), "--base", "org/base", "--out", str(out),
            "--awq", "--calibration", str(calib), "--register", "--alias", "a",
            "--tool-parser", "hermes", "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    assert run_cli_calls == [
        [
            "vibe", "register", str(out.resolve()), "--alias", "a",
            "--quantization", "awq", "--tool-parser", "hermes",
        ]
    ]


def test_export_verify_chat_template_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IMP-05.26: an export without a chat template is refused."""
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    adapter = _make_adapter(tmp_path)

    def fake_merge(adapter_path: Path, base: str, out: Path) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        (out / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return out

    monkeypatch.setattr(export, "_merge_lora", fake_merge)
    rc = export.main(
        ["--adapter", str(adapter), "--base", "org/base", "--out", str(tmp_path / "out"),
         "--run-dir", str(tmp_path)]
    )
    assert rc == 1


def test_calibration_texts_messages_rows(tmp_path: Path) -> None:
    """IMP-05.25: calibration flattens the finetune `messages` format."""
    calib = tmp_path / "calib.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        for _ in range(3)
    ]
    calib.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    texts = export._calibration_texts(calib, n=3)
    assert texts == ["be nice\nhi\nhello"] * 3


def test_calibration_texts_insufficient(tmp_path: Path) -> None:
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"prompt": "p"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="usable samples"):
        export._calibration_texts(calib, n=128)


def test_export_base_file_local_and_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IMP-05.26: generation_config.json resolves from the base snapshot."""
    local = tmp_path / "base"
    local.mkdir()
    (local / "generation_config.json").write_text("{}", encoding="utf-8")
    assert export._base_file(str(local), "generation_config.json") == local / "generation_config.json"
    assert export._base_file(str(local), "missing.json") is None

    import huggingface_hub

    remote = tmp_path / "remote-gen.json"
    remote.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(remote))
    assert export._base_file("org/base", "generation_config.json") == remote


def test_export_register_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    adapter = _make_adapter(tmp_path)
    monkeypatch.setattr(
        export, "_merge_lora", lambda a, b, out: _write_exported_model(out) or out
    )
    monkeypatch.setattr(export, "_run_cli", lambda argv: 1)
    rc = export.main(
        [
            "--adapter", str(adapter), "--base", "org/base", "--out", str(tmp_path / "out"),
            "--register", "--alias", "x", "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 1


def test_export_usage_errors(tmp_path: Path) -> None:
    """IMP-05.15: usage errors exit before start_run — no runs/ litter."""
    adapter = _make_adapter(tmp_path)
    rc = export.main(
        ["--adapter", str(adapter), "--base", "b", "--out", str(tmp_path / "o"), "--register", "--run-dir", str(tmp_path)]
    )
    assert rc == 2  # --register without --alias
    rc = export.main(
        ["--adapter", str(tmp_path / "nope"), "--base", "b", "--out", str(tmp_path / "o"), "--run-dir", str(tmp_path)]
    )
    assert rc == 2  # no adapter_config.json
    assert not any(
        p.is_dir() and p.name.endswith("-export") for p in tmp_path.iterdir()
    )
    with pytest.raises(SystemExit) as excinfo:
        export.main(["--base", "b"])
    assert isinstance(excinfo.value, SystemExit)
    assert excinfo.value.code == 2  # missing --adapter
