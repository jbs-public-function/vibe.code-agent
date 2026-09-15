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
    """Free VRAM 15.2 GiB and no model READY: both guards pass."""
    monkeypatch.setattr(common, "free_vram_gib", lambda: 15.2)
    monkeypatch.setattr(common, "_vibe_status_json", lambda: None)


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
    assert capsys.readouterr().out.strip() == "/cache/org/repo"


def test_download_refuses_when_model_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "free_vram_gib", lambda: 15.2)
    monkeypatch.setattr(
        common,
        "_vibe_status_json",
        lambda: '{"state": "READY", "active_alias": "x"}',
    )
    monkeypatch.setattr(download, "_snapshot_download", lambda *args: "/never")
    rc = download.main(["org/repo", "--run-dir", str(tmp_path / "r1")])
    assert rc == 1
    rc = download.main(
        ["org/repo", "--allow-coexist", "--run-dir", str(tmp_path / "r2")]
    )
    assert rc == 0


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
    assert calls[-1]["cache_dir"] == "/data/vibe"


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
        ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 8.0),
        ("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1.5),
        ("model.gguf", None),
    ],
)
def test_gguf_guess_params(name: str, expected: float | None) -> None:
    assert inference_gguf._guess_params_from_name(name) == expected


def test_gguf_main_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _guards_ok(monkeypatch)
    gguf_file = tmp_path / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    gguf_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(inference_gguf, "_resolve_gguf", lambda spec: gguf_file)
    loaded: list[tuple[str, int]] = []

    def fake_load(path: Path, backend: str, n_gpu_layers: int) -> tuple[SimpleNamespace, SimpleNamespace]:
        loaded.append((backend, n_gpu_layers))
        return SimpleNamespace(), SimpleNamespace()

    def fake_stream(model: Any, tokenizer: Any, prompt: str, backend: str, **kwargs: Any) -> Any:
        yield "a"
        yield "b"

    monkeypatch.setattr(inference_gguf, "_load_gguf_model", fake_load)
    monkeypatch.setattr(inference_gguf, "_stream_tokens", fake_stream)
    rc = inference_gguf.main(
        ["--gguf", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "--prompt", "hi", "--run-dir", str(tmp_path)]
    )
    assert rc == 0
    assert "ab" in capsys.readouterr().out
    assert loaded == [("transformers", -1)]
    metrics = _metrics(tmp_path, "inference_gguf")
    assert metrics["precision"] == "4bit"
    assert metrics["backend"] == "transformers"
    assert metrics["model_params_billions"] == 7.0


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


def test_training_estimate_pinned() -> None:
    """Numbers pinned to the spec VRAM table (NFR-05.3)."""
    # 7B QLoRA: 7*0.55 + 1.5 + 0.36*7 + 2.5 + 0.5 = 10.87 → allowed.
    assert finetune_lora._training_estimate_gib(7.0, qlora=True) == pytest.approx(10.87)
    # 14B QLoRA: 14*0.55 + 1.5 + 0.36*14 + 3.0 = 17.24 > 15.2*1.1 → refused.
    assert finetune_lora._training_estimate_gib(14.0, qlora=True) == pytest.approx(17.24)
    # 1.5B LoRA: 1.5*2.0 + 0.4*1.5 + 3.0 = 6.6 → allowed.
    assert finetune_lora._training_estimate_gib(1.5, qlora=False) == pytest.approx(6.6)


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
    metrics = _metrics(tmp_path, "finetune_lora")
    assert metrics["samples"] == 1
    assert metrics["qlora"] is True


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
    data = tmp_path / "bad.jsonl"
    data.write_text("{not json}\n", encoding="utf-8")
    rc = finetune_lora.main(["--base", "org/7b", "--data", str(data), "--run-dir", str(tmp_path)])
    assert rc == 2


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
                raise httpx.ConnectError(f"boom {idx}")
            return FakeResponse(f"completion-{idx}")
        finally:
            FakeAsyncClient.concurrency -= 1


@pytest.fixture
def fake_async_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeAsyncClient]:
    FakeAsyncClient.calls = []
    FakeAsyncClient.concurrency = 0
    FakeAsyncClient.max_concurrency = 0
    FakeAsyncClient.fails_left = {}
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
    assert rows[0] == {"idx": 0, "prompt": "prompt-0", "completion": "completion-0"}


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

    async def fake_generate(alias: str, prompts_list: list[dict], out_cache: Path, temperature: float, n: int | None, concurrency: int = 4) -> int:
        out_cache.parent.mkdir(parents=True, exist_ok=True)
        out_cache.write_text(json.dumps({"idx": 0, "prompt": "hi", "completion": "bye"}) + "\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(distill, "_generate_teacher", fake_generate)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    run_cli_calls: list[list[str]] = []
    monkeypatch.setattr(distill, "_run_cli", lambda argv: run_cli_calls.append(argv) or 0)
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
    assert len(train_calls) == 1


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


def test_export_merge_and_register(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    adapter = _make_adapter(tmp_path)
    merge_calls: list[tuple[Path, str, Path]] = []

    def fake_merge(adapter_path: Path, base: str, out: Path) -> Path:
        merge_calls.append((adapter_path, base, out))
        out.mkdir(parents=True, exist_ok=True)
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
    monkeypatch.setattr(export, "_merge_lora", lambda a, b, out: out.mkdir(parents=True, exist_ok=True) or out)
    awq_calls: list[tuple] = []

    def fake_awq(model_dir: Path, out: Path, calibration: Path | None, n: int = 128) -> Path:
        awq_calls.append((model_dir, out, calibration, n))
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
    assert awq_calls == [(out.resolve(), out.resolve(), calib, 128)]
    assert _metrics(tmp_path, "export")["awq"] is True


def test_export_register_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _guards_ok(monkeypatch)
    monkeypatch.setattr(inference_hf, "_load_config", lambda model: dict(QWEN_1_5B_CONFIG))
    adapter = _make_adapter(tmp_path)
    monkeypatch.setattr(export, "_merge_lora", lambda a, b, out: out.mkdir(parents=True, exist_ok=True) or out)
    monkeypatch.setattr(export, "_run_cli", lambda argv: 1)
    rc = export.main(
        [
            "--adapter", str(adapter), "--base", "org/base", "--out", str(tmp_path / "out"),
            "--register", "--alias", "x", "--run-dir", str(tmp_path),
        ]
    )
    assert rc == 1


def test_export_usage_errors(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    rc = export.main(
        ["--adapter", str(adapter), "--base", "b", "--out", str(tmp_path / "o"), "--register", "--run-dir", str(tmp_path)]
    )
    assert rc == 2  # --register without --alias
    rc = export.main(
        ["--adapter", str(tmp_path / "nope"), "--base", "b", "--out", str(tmp_path / "o"), "--run-dir", str(tmp_path)]
    )
    assert rc == 2  # no adapter_config.json
    with pytest.raises(SystemExit) as excinfo:
        export.main(["--base", "b"])
    assert isinstance(excinfo.value, SystemExit)
    assert excinfo.value.code == 2  # missing --adapter
