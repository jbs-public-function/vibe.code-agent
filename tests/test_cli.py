"""CLI tests for the Phase 4 commands (FR-04.3–04.5, FR-04.8).

Every command talks to the controller only through `ControllerClient`; the
tests monkeypatch `vibe.cli.ControllerClient` with a recording fake and invoke
the Typer app with `CliRunner` (the sync-wrapper pattern keeps the commands
`CliRunner`-safe). `CliRunner` captures stdout/stderr separately (click ≥ 8.2),
so the deterministic non-tty progress lines land on stdout and errors on
stderr.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import pytest
from typer.testing import CliRunner

import vibe.cli as cli
from vibe.cli import app
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2


class FakeClient:
    """Recording in-memory stand-in for the Phase 4 client contract."""

    def __init__(self) -> None:
        self.models: list[control_pb2.ModelInfo] = []
        self.models_error: Exception | None = None
        self.download_infos: list[control_pb2.DownloadInfo] = []
        self.download_events: list[control_pb2.DownloadProgress] = []
        self.download_error: Exception | None = None
        self.download_calls: list[tuple[str, list[str], str | None]] = []
        self.cache_value: control_pb2.ListCacheResponse = control_pb2.ListCacheResponse()
        self.cache_error: Exception | None = None
        self.remove_calls: list[str] = []
        self.remove_result: tuple[bool, str] = (True, "")
        self.propose_response: control_pb2.ProposeRegistrationResponse = (
            control_pb2.ProposeRegistrationResponse()
        )
        self.propose_error: Exception | None = None
        self.propose_calls: list[control_pb2.RegistrationSpec] = []
        self.write_response: control_pb2.WriteRegistrationResponse = (
            control_pb2.WriteRegistrationResponse(ok=True)
        )
        self.write_error: Exception | None = None
        self.write_calls: list[control_pb2.RegistrationSpec] = []
        self.reload_errors: list[str] = []
        self.reload_calls: int = 0
        self.calls: list[str] = []

    async def list_models(self) -> list[control_pb2.ModelInfo]:
        self.calls.append("list_models")
        if self.models_error is not None:
            raise self.models_error
        return list(self.models)

    async def list_downloads(self) -> list[control_pb2.DownloadInfo]:
        self.calls.append("list_downloads")
        return list(self.download_infos)

    async def download(
        self,
        hf_repo: str,
        include: Sequence[str] | None = None,
        revision: str | None = None,
    ) -> AsyncIterator[control_pb2.DownloadProgress]:
        self.calls.append("download")
        self.download_calls.append((hf_repo, list(include or []), revision))
        if self.download_error is not None:
            raise self.download_error
        for event in self.download_events:
            yield event

    async def cancel_download(self, download_id: str) -> bool:
        self.calls.append("cancel_download")
        return True

    async def list_cache(self) -> control_pb2.ListCacheResponse:
        self.calls.append("list_cache")
        if self.cache_error is not None:
            raise self.cache_error
        return self.cache_value

    async def remove_cached(self, repo_id: str) -> tuple[bool, str]:
        self.calls.append("remove_cached")
        self.remove_calls.append(repo_id)
        return self.remove_result

    async def reload_registry(self) -> list[str]:
        self.calls.append("reload_registry")
        self.reload_calls += 1
        return list(self.reload_errors)

    async def propose_registration(
        self, spec: control_pb2.RegistrationSpec
    ) -> control_pb2.ProposeRegistrationResponse:
        self.calls.append("propose_registration")
        self.propose_calls.append(spec)
        if self.propose_error is not None:
            raise self.propose_error
        return self.propose_response

    async def write_registration(
        self, spec: control_pb2.RegistrationSpec
    ) -> control_pb2.WriteRegistrationResponse:
        self.calls.append("write_registration")
        self.write_calls.append(spec)
        if self.write_error is not None:
            raise self.write_error
        return self.write_response


@pytest.fixture
def fake_client() -> FakeClient:
    """A fresh recording fake for one command invocation."""
    return FakeClient()


@pytest.fixture
def runner(fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """`CliRunner` wired to the fake via the `vibe.cli.ControllerClient` symbol."""
    monkeypatch.setenv("VLLM_IMAGE_TAG", "v0.11.2")  # Settings() validation
    monkeypatch.setattr(cli, "ControllerClient", lambda settings: fake_client)
    return CliRunner()


def make_progress(
    download_id: str,
    *,
    bytes_done: int,
    bytes_total: int = 1000,
    progress: float,
    current_file: str = "",
    terminal: bool = False,
    error: str = "",
) -> control_pb2.DownloadProgress:
    return control_pb2.DownloadProgress(
        id=download_id,
        progress=progress,
        bytes_done=bytes_done,
        bytes_total=bytes_total,
        current_file=current_file,
        terminal=terminal,
        error=error,
    )


# -- download / downloads (FR-04.3) ------------------------------------------


def test_download_streams_progress_and_exits_zero(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.download_events = [
        make_progress("dl-1", bytes_done=250, progress=0.25, current_file="model-00001.safetensors"),
        make_progress("dl-1", bytes_done=500, progress=0.5, current_file="model-00002.safetensors"),
        make_progress("dl-1", bytes_done=1000, progress=1.0, terminal=True),
    ]
    result = runner.invoke(
        app,
        ["download", "org/repo", "--include", "*.safetensors", "--revision", "main"],
    )
    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "model-00001.safetensors  250/1000 bytes (25%)",
        "model-00002.safetensors  500/1000 bytes (50%)",
        "1000/1000 bytes (100%)",
        "downloaded org/repo",
    ]
    assert fake_client.download_calls == [("org/repo", ["*.safetensors"], "main")]


def test_download_terminal_error_exits_one(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.download_events = [
        make_progress("dl-2", bytes_done=100, progress=0.1),
        make_progress(
            "dl-2",
            bytes_done=100,
            progress=0.1,
            terminal=True,
            error="401: gated repo — check HF_TOKEN",
        ),
    ]
    result = runner.invoke(app, ["download", "org/gated"])
    assert result.exit_code == 1
    assert "download failed: 401: gated repo — check HF_TOKEN" in result.stderr
    assert "downloaded" not in result.stdout


def test_download_controller_down_exits_one(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.download_error = ControllerUnavailable("down")
    result = runner.invoke(app, ["download", "org/repo"])
    assert result.exit_code == 1
    assert "controller unreachable — start it with `vibe serve`" in result.stderr


def test_downloads_table(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.download_infos = [
        control_pb2.DownloadInfo(
            id="dl-1", hf_repo="org/a", progress=0.25, bytes_done=250, bytes_total=1000
        ),
        control_pb2.DownloadInfo(
            id="dl-2", hf_repo="org/b", progress=1.0, bytes_done=10, bytes_total=10,
            done=True, ok=True,
        ),
        control_pb2.DownloadInfo(
            id="dl-3", hf_repo="org/c", progress=0.5, bytes_done=5, bytes_total=10,
            done=True, ok=False, error="boom",
        ),
    ]
    result = runner.invoke(app, ["downloads"])
    assert result.exit_code == 0
    assert "ID" in result.stdout and "REPO" in result.stdout and "STATE" in result.stdout
    assert "dl-1" in result.stdout and "25%" in result.stdout and "DOWNLOADING" in result.stdout
    assert "dl-2" in result.stdout and "DONE" in result.stdout
    assert "dl-3" in result.stdout and "ERROR: boom" in result.stdout


def test_downloads_empty(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["downloads"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "no downloads"


# -- cache (FR-04.4) ----------------------------------------------------------


def test_cache_list_table_and_json(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.cache_value = control_pb2.ListCacheResponse(
        entries=[
            control_pb2.CacheEntry(
                repo_id="org/a",
                size_bytes=3 * 1024**3,
                last_modified="2026-09-15 10:00",
                aliases=["qwen"],
            ),
            control_pb2.CacheEntry(
                repo_id="org/b", size_bytes=512 * 1024**2, last_modified="2026-09-14 09:00"
            ),
        ],
        disk_total_bytes=1024 * 1024**3,
        disk_free_bytes=100 * 1024**3,
    )
    table = runner.invoke(app, ["cache", "list"])
    assert table.exit_code == 0
    assert "REPO" in table.stdout and "SIZE" in table.stdout and "ALIASES" in table.stdout
    assert "org/a" in table.stdout and "3.0 GiB" in table.stdout and "qwen" in table.stdout
    assert "org/b" in table.stdout and "512.0 MiB" in table.stdout
    as_json = runner.invoke(app, ["cache", "list", "--json"])
    assert as_json.exit_code == 0
    data = json.loads(as_json.stdout)
    assert data[0]["repoId"] == "org/a"
    assert data[0]["sizeBytes"] == str(3 * 1024**3)
    assert data[1]["aliases"] == []


def test_cache_du(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.cache_value = control_pb2.ListCacheResponse(
        disk_total_bytes=500 * 1024**3, disk_free_bytes=120 * 1024**3
    )
    result = runner.invoke(app, ["cache", "du"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "total: 500.0 GiB / free: 120.0 GiB"


def test_cache_rm_confirmed(
    runner: CliRunner, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.typer, "confirm", lambda prompt: True)
    result = runner.invoke(app, ["cache", "rm", "org/a"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "removed org/a"
    assert fake_client.remove_calls == ["org/a"]


def test_cache_rm_declined(
    runner: CliRunner, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.typer, "confirm", lambda prompt: False)
    result = runner.invoke(app, ["cache", "rm", "org/a"])
    assert result.exit_code == 0
    assert result.stdout.strip() == ""
    assert fake_client.remove_calls == []


def test_cache_rm_not_removed_exits_one(
    runner: CliRunner, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.typer, "confirm", lambda prompt: True)
    fake_client.remove_result = (False, "permission denied")
    result = runner.invoke(app, ["cache", "rm", "org/a"])
    assert result.exit_code == 1
    assert "not removed: permission denied" in result.stderr


# -- register (FR-04.5) -------------------------------------------------------


def test_register_happy_path_records_propose_write_reload(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.propose_response = control_pb2.ProposeRegistrationResponse(
        model=control_pb2.ModelInfo(
            alias="qwen2.5-coder-1.5b",
            display_name="Qwen2.5 Coder 1.5B",
            hf_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            quantization="none",
            max_model_len=32768,
            gpu_memory_utilization=0.6,
            host_port=8011,
            tags=["code"],
            use_case="code gen",
            state=control_pb2.AVAILABLE,
            cached_locally=True,
        ),
        warnings=["vram estimate: 9.4 GiB"],
    )
    result = runner.invoke(
        app,
        [
            "register",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "--alias", "qwen2.5-coder-1.5b",
            "--quantization", "none",
            "--gpu-mem", "0.6",
            "--port", "8011",
            "--tool-parser", "hermes",
            "--use-case", "code gen",
            "--tags", "code",
            "--tags", "tools",
        ],
    )
    assert result.exit_code == 0
    assert "warning: vram estimate: 9.4 GiB" in result.stderr
    assert "registered qwen2.5-coder-1.5b — http://127.0.0.1:8011/v1" in result.stdout
    assert "models.yaml updated" in result.stdout
    assert fake_client.calls == [
        "propose_registration",
        "write_registration",
        "reload_registry",
    ]
    spec = fake_client.propose_calls[0]
    assert spec.hf_repo == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert spec.alias == "qwen2.5-coder-1.5b"
    assert spec.quantization == "none"
    assert spec.gpu_memory_utilization == pytest.approx(0.6)
    assert spec.host_port == 8011
    assert spec.tool_parser == "hermes"
    assert list(spec.tags) == ["code", "tools"]
    assert fake_client.write_calls[0].alias == "qwen2.5-coder-1.5b"


def test_register_propose_errors_exit_two(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.propose_response = control_pb2.ProposeRegistrationResponse(
        errors=["alias 'bad!' must match ^[a-z0-9][a-z0-9.-]*$", "host_port 8020 already in use"]
    )
    result = runner.invoke(app, ["register", "org/repo", "--alias", "bad!"])
    assert result.exit_code == 2
    assert "host_port 8020 already in use" in result.stderr
    assert "alias 'bad!'" in result.stderr
    assert fake_client.calls == ["propose_registration"]


def test_register_write_errors_exit_two(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.propose_response = control_pb2.ProposeRegistrationResponse(
        model=control_pb2.ModelInfo(
            alias="ok-alias",
            display_name="ok",
            hf_repo="org/repo",
            quantization="none",
            max_model_len=4096,
            gpu_memory_utilization=0.6,
            host_port=8011,
        )
    )
    fake_client.write_response = control_pb2.WriteRegistrationResponse(
        ok=False, errors=["models.yaml is not writable"]
    )
    result = runner.invoke(app, ["register", "org/repo", "--alias", "ok-alias"])
    assert result.exit_code == 2
    assert "models.yaml is not writable" in result.stderr
    assert fake_client.calls == ["propose_registration", "write_registration"]


def test_register_reload_errors_exit_two(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.propose_response = control_pb2.ProposeRegistrationResponse(
        model=control_pb2.ModelInfo(
            alias="ok-alias",
            display_name="ok",
            hf_repo="org/repo",
            quantization="none",
            max_model_len=4096,
            gpu_memory_utilization=0.6,
            host_port=8011,
        )
    )
    fake_client.reload_errors = ["models.yaml is not valid YAML"]
    result = runner.invoke(app, ["register", "org/repo", "--alias", "ok-alias"])
    assert result.exit_code == 2
    assert "models.yaml is not valid YAML" in result.stderr
    assert fake_client.reload_calls == 1


# -- zed-settings (FR-04.8) ---------------------------------------------------


def test_zed_settings_prints_json(
    runner: CliRunner, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client.models = [
        control_pb2.ModelInfo(
            alias="qwen2.5-coder-7b",
            display_name="Qwen 7B",
            hf_repo="org/qwen",
            max_model_len=32768,
            tags=["code"],
        )
    ]
    monkeypatch.setattr(
        cli,
        "zed_settings_json",
        lambda models: json.dumps(
            {"profiles": {"vibe-local": {"models": [m.alias for m in models]}}},
            indent=2,
        ),
    )
    result = runner.invoke(app, ["zed-settings"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "vibe-local" in data["profiles"]
    assert data["profiles"]["vibe-local"]["models"] == ["qwen2.5-coder-7b"]
    assert fake_client.calls == ["list_models"]


def test_zed_settings_controller_down_exits_one(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.models_error = ControllerUnavailable("down")
    result = runner.invoke(app, ["zed-settings"])
    assert result.exit_code == 1
    assert "controller unreachable — start it with `vibe serve`" in result.stderr
