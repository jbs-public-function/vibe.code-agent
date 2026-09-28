"""Tests for laboratory/common.py — run dirs, VRAM/serving guards, and helpers.

Everything runs in a torch-free environment: `common` never imports torch at
module level (asserted below), and GPU state is faked via monkeypatching.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laboratory"))

import common  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_module_level_no_torch() -> None:
    """Importing common must not pull in torch/transformers (testability rule)."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "peft" not in sys.modules


# -- start_run / Run --------------------------------------------------------


def test_start_run_creates_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "RUNS_DIR", tmp_path)
    run = common.start_run("download", ["org/repo"])
    assert run.dir.parent == tmp_path
    assert run.dir.name.endswith("-download")
    assert run.script == "download"
    assert run.started.tzinfo is not None
    args = json.loads((run.dir / "args.json").read_text(encoding="utf-8"))
    assert args == {"script": "download", "argv": ["org/repo"]}


def test_start_run_runs_dir_override(tmp_path: Path) -> None:
    run = common.start_run("download", [], runs_dir=tmp_path)
    assert run.dir.parent == tmp_path


def test_start_run_suffix_on_collision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_datetime = datetime

    class FakeDatetime:
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return real_datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(common, "datetime", FakeDatetime)
    first = common.start_run("download", [], runs_dir=tmp_path)
    second = common.start_run("download", [], runs_dir=tmp_path)
    third = common.start_run("download", [], runs_dir=tmp_path)
    assert first.dir.name == "20260915-120000-download"
    assert second.dir.name == "20260915-120000-download-1"
    assert third.dir.name == "20260915-120000-download-2"
    assert all(run.dir.is_dir() for run in (first, second, third))


def test_run_write_metrics_merges(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run = common.Run(dir=run_dir, script="x", started=datetime.now(UTC))
    run.write_metrics({"a": 1})
    run.write_metrics({"b": 2})
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {"a": 1, "b": 2}


def test_run_log_line_appends(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run = common.Run(dir=run_dir, script="x", started=datetime.now(UTC))
    run.log_line("hello")
    run.log_line("world")
    lines = (run_dir / "console.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("  hello")
    assert lines[1].endswith("  world")


# -- vram_guard (NFR-05.3) --------------------------------------------------


@pytest.mark.parametrize(
    ("estimate", "force", "allowed", "fragment"),
    [
        # 7B fp16: 7*2.0 + 2.5 + 0.5 = 17 GiB vs 15.2 free → refused.
        (17.0, False, False, "exceeds free VRAM"),
        # 7B QLoRA: 7*0.55 + 1.5 + 0.36*7 + 2.5 + 0.5 ≈ 10.87 → allowed.
        (10.87, False, True, "fits in free VRAM"),
        # Over free but within 10 % → allowed with a warning.
        (16.0, False, True, "proceeding anyway"),
        # Refusal bypassed by --force.
        (17.0, True, True, "overridden by --force"),
    ],
)
def test_vram_guard(
    monkeypatch: pytest.MonkeyPatch,
    estimate: float,
    force: bool,
    allowed: bool,
    fragment: str,
) -> None:
    monkeypatch.setattr(common, "free_vram_gib", lambda: 15.2)
    actual, message = common.vram_guard(estimate, force=force)
    assert actual is allowed
    assert fragment in message


def test_vram_guard_unknown_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "free_vram_gib", lambda: None)
    allowed, message = common.vram_guard(17.0, force=False)
    assert allowed is True
    assert message == "VRAM unknown — guard skipped"


def test_vram_guard_never_raises_when_imports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """free_vram_gib returns None (not an exception) when torch/pynvml are absent."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert common.free_vram_gib() is None
    assert common.vram_guard(17.0, force=False)[0] is True


# -- serving_guard (NFR-05.4, IMP-05.2/05.8) ---------------------------------


def test_serving_guard_refuses_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        common,
        "_vibe_status_json",
        lambda: '{"state": "READY", "active_alias": "qwen2.5-coder-7b"}',
    )
    allowed, message = common.serving_guard(allow_coexist=False)
    assert allowed is False
    assert "run `vibe unload` first" in message


def test_serving_guard_ready_with_allow_coexist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "_vibe_status_json", lambda: '{"state": "READY"}')
    allowed, message = common.serving_guard(allow_coexist=True)
    assert allowed is True
    assert "--allow-coexist" in message


def test_serving_guard_loading_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOADING is busy even without an alias (IMP-05.8)."""
    monkeypatch.setattr(
        common,
        "_vibe_status_json",
        lambda: '{"state": "LOADING", "active_alias": "qwen2.5-coder-7b"}',
    )
    allowed, _message = common.serving_guard(allow_coexist=False)
    assert allowed is False


def test_serving_guard_unloading_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "_vibe_status_json", lambda: '{"state": "UNLOADING"}')
    allowed, _message = common.serving_guard(allow_coexist=False)
    assert allowed is False


def test_serving_guard_current_job_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A current_job in flight is busy even in AVAILABLE state (IMP-05.8)."""
    monkeypatch.setattr(
        common,
        "_vibe_status_json",
        lambda: '{"state": "AVAILABLE", "current_job": {"id": "job-1", "kind": 1, "alias": "x"}}',
    )
    allowed, _message = common.serving_guard(allow_coexist=False)
    assert allowed is False


def test_serving_guard_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        common,
        "_vibe_status_json",
        lambda: '{"state": "AVAILABLE", "active_alias": ""}',
    )
    allowed, message = common.serving_guard(allow_coexist=False)
    assert allowed is True
    assert message == "no model is busy on the GPU"


def test_serving_guard_no_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMP-05.2: an unreachable controller hard-fails, never silently skips."""
    monkeypatch.setattr(common, "_vibe_status_json", lambda: None)
    allowed, message = common.serving_guard(allow_coexist=False)
    assert allowed is False
    assert "serving guard cannot check the GPU" in message
    allowed, message = common.serving_guard(allow_coexist=True)
    assert allowed is True
    assert "--allow-coexist" in message


def test_serving_guard_cli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> str:
        raise FileNotFoundError("vibe")

    monkeypatch.setattr(common, "_vibe_status_json", missing)
    allowed, message = common.serving_guard(allow_coexist=False)
    assert allowed is False
    assert "serving guard cannot check the GPU" in message


def test_serving_guard_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "_vibe_status_json", lambda: "not json")
    allowed, message = common.serving_guard(allow_coexist=False)
    assert allowed is False
    assert "serving guard cannot check the GPU" in message


def test_serving_guard_genuine_messagedict(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMP-05.8: the guard parses genuine `MessageToDict(Status(...))` output.

    CamelCase keys with `always_print_fields_with_no_presence=True`
    (protobuf 7.x) is what a plain MessageToDict emits; READY must be busy.
    """
    from google.protobuf.json_format import MessageToDict
    from vibe.gen.vibe.v1 import control_pb2

    status = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        current_job=control_pb2.Job(id="", kind=control_pb2.LOAD, alias=""),
    )
    raw = json.dumps(MessageToDict(status, always_print_fields_with_no_presence=True))
    assert "activeAlias" in raw and "active_alias" not in raw
    monkeypatch.setattr(common, "_vibe_status_json", lambda: raw)
    allowed, _message = common.serving_guard(allow_coexist=False)
    assert allowed is False


def test_serving_guard_snake_case_cli_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual `vibe status --json` contract: snake_case, real ints (IMP-03.15)."""
    from google.protobuf.json_format import MessageToDict
    from vibe.gen.vibe.v1 import control_pb2

    status = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.LOADING,
        started_at_unix=1234567890,
        current_job=control_pb2.Job(id="job-9", kind=control_pb2.LOAD, alias="x"),
    )
    raw = json.dumps(
        MessageToDict(
            status,
            always_print_fields_with_no_presence=True,
            preserving_proto_field_name=True,
        )
    )
    assert "active_alias" in raw and "activeAlias" not in raw
    monkeypatch.setattr(common, "_vibe_status_json", lambda: raw)
    allowed, _message = common.serving_guard(allow_coexist=False)
    assert allowed is False


def _grpc_unavailable(timeout: float) -> str:
    """Stand-in for `_status_via_grpc` in a torch-free env without the package."""
    raise ImportError("vibe")


def test_vibe_status_json_runs_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"state": "AVAILABLE"}')

    monkeypatch.setattr(common, "_status_via_grpc", _grpc_unavailable)
    monkeypatch.setattr(common.subprocess, "run", fake_run)
    assert common._vibe_status_json() == '{"state": "AVAILABLE"}'
    argv, kwargs = calls[0]
    assert argv == ["vibe", "status", "--json"]
    assert kwargs["timeout"] == 15  # IMP-05.11


def test_vibe_status_json_prefers_grpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMP-05.11: a direct gRPC GetStatus is preferred over the CLI."""
    calls: list[tuple] = []

    def fake_grpc(timeout: float) -> str:
        calls.append(("grpc", timeout))
        return '{"state": "AVAILABLE"}'

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(("cli", kwargs))
        return SimpleNamespace(returncode=0, stdout='{"state": "READY"}')

    monkeypatch.setattr(common, "_status_via_grpc", fake_grpc)
    monkeypatch.setattr(common.subprocess, "run", fake_run)
    assert common._vibe_status_json() == '{"state": "AVAILABLE"}'
    assert calls == [("grpc", 15.0)]


def test_vibe_status_json_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "_status_via_grpc", _grpc_unavailable)
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert common._vibe_status_json() is None


def test_vibe_status_json_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(argv: list[str], **kwargs: object) -> SimpleNamespace:
        raise FileNotFoundError("vibe")

    monkeypatch.setattr(common, "_status_via_grpc", _grpc_unavailable)
    monkeypatch.setattr(common.subprocess, "run", boom)
    with pytest.raises(FileNotFoundError):
        common._vibe_status_json()


def test_serving_guard_end_to_end_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """serving_guard parses the real subprocess seam's stdout."""
    monkeypatch.setattr(common, "_status_via_grpc", _grpc_unavailable)
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(
            returncode=0, stdout='{"state": "READY", "active_alias": "x"}'
        ),
    )
    allowed, _message = common.serving_guard(allow_coexist=False)
    assert allowed is False


# -- parse_jsonl ------------------------------------------------------------


def test_parse_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"prompt": "a", "completion": "b"}\n\n  \n{"prompt": "c", "completion": "d"}\n',
        encoding="utf-8",
    )
    assert common.parse_jsonl(path) == [
        {"prompt": "a", "completion": "b"},
        {"prompt": "c", "completion": "d"},
    ]


def test_parse_jsonl_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"prompt": "ok"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"data\.jsonl:2: invalid JSON"):
        common.parse_jsonl(path)


def test_parse_jsonl_non_object(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        common.parse_jsonl(path)


# -- VRAM estimation helpers -------------------------------------------------


def test_estimate_vram_gib() -> None:
    # 7B fp16: 7*2.0 + 2.5 + 0.5*(2048/2048) = 17 GiB.
    assert common.estimate_vram_gib(7.0, common.BYTES_PER_PARAM["fp16"]) == pytest.approx(17.0)
    # Half the context → half the KV/activation term.
    assert common.estimate_vram_gib(7.0, 2.0, seq_len=1024) == pytest.approx(16.75)
    # 7B 4-bit: 7*0.55 + 3.0 = 6.85 GiB.
    assert common.estimate_vram_gib(7.0, common.BYTES_PER_PARAM["4bit"]) == pytest.approx(6.85)


QWEN_7B_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "vocab_size": 152064,
}
QWEN_1_5B_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "vocab_size": 151936,
}


def test_model_params_billions_qwen_7b() -> None:
    # 28 * (4*3584² + 3*3584*18944) + 152064*3584 = 7,686,848,512.
    assert common.model_params_billions(QWEN_7B_CONFIG) == pytest.approx(7.686848512)


def test_model_params_billions_qwen_1_5b() -> None:
    # 28 * (4*1536² + 3*1536*8960) + 151936*1536 = 1,653,669,888.
    assert common.model_params_billions(QWEN_1_5B_CONFIG) == pytest.approx(1.653669888)


def test_model_params_billions_missing_keys() -> None:
    assert common.model_params_billions({}) is None
    assert common.model_params_billions({"num_hidden_layers": 28}) is None
    assert common.model_params_billions({"num_hidden_layers": None, "hidden_size": 2}) is None


# -- parse_seed -------------------------------------------------------------


def test_parse_seed_none_prints_choice_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """IMP-05.9: the generated seed is printed to stdout, not stderr."""
    seed = common.parse_seed(None)
    assert isinstance(seed, int)
    captured = capsys.readouterr()
    assert f"seed: {seed}" in captured.out
    assert captured.err == ""


def test_parse_seed_none_applies_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMP-05.9: parse_seed(None) actually seeds random/numpy/torch.

    numpy and torch are not root deps, so they are faked in sys.modules
    (the same pattern the print_peak_vram test uses for torch).
    """
    numpy_seeds: list[int] = []
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.random = types.SimpleNamespace(  # type: ignore[attr-defined]
        seed=lambda s: numpy_seeds.append(s)
    )
    torch_seeds: list[int] = []
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: False  # type: ignore[attr-defined]
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    fake_torch.manual_seed = lambda s: torch_seeds.append(s)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    seed = common.parse_seed(None)
    assert numpy_seeds == [seed]
    assert torch_seeds == [seed]
    # Python's random module was seeded too: the next draw must equal a
    # fresh Random(seed) draw (deterministic, IMP-05.9/FR-05.10).
    import random as stdlib_random

    assert common.random.random() == stdlib_random.Random(seed).random()


def test_parse_seed_explicit_without_torch() -> None:
    """Explicit seed works in a torch-free env (numpy/torch branches skipped)."""
    assert common.parse_seed(42) == 42
    first = common.random.random()
    common.parse_seed(42)
    assert common.random.random() == first


# -- add_common_args --------------------------------------------------------


def test_add_common_args() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    common.add_common_args(parser)
    args = parser.parse_args(["--seed", "7", "--force", "--allow-coexist", "--run-dir", "/tmp/runs"])
    assert args.seed == 7
    assert args.force is True
    assert args.allow_coexist is True
    assert args.run_dir == Path("/tmp/runs")
    defaults = parser.parse_args([])
    assert defaults.seed is None
    assert defaults.force is False
    assert defaults.allow_coexist is False
    assert defaults.run_dir == common.RUNS_DIR


# -- print_peak_vram --------------------------------------------------------


def test_print_peak_vram_noop_without_torch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run = common.Run(dir=run_dir, script="x", started=datetime.now(UTC))
    common.print_peak_vram(run)
    assert not (run_dir / "metrics.json").exists()
    assert capsys.readouterr().out == ""


def test_print_peak_vram_with_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: True  # type: ignore[attr-defined]
    fake_cuda.max_memory_allocated = lambda: 3 * 2**30  # type: ignore[attr-defined]
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run = common.Run(dir=run_dir, script="x", started=datetime.now(UTC))
    common.print_peak_vram(run)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["peak_vram_gib"] == pytest.approx(3.0)
    assert "peak_vram_gib = 3.00" in capsys.readouterr().out
