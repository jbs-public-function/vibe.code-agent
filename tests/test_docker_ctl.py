"""Tests for the docker adapter's bind mounts (FR-00.9, M5.6) and
`wait_ready` polling (FR-00.10).

`volumes_for` is pure and unit-tested directly; `DockerCtl.start` and
`DockerCtl.wait_ready` are exercised against in-memory fake docker clients
(mirroring conftest's `FakeDocker`, but faking the client one level lower
so `containers.run` kwargs and readiness polling are visible). No real
Docker daemon is touched, and this module does not import `docker`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from vibe.config import Settings
from vibe.docker_ctl import (
    ContainerExited,
    DockerCtl,
    DockerError,
    ReadinessTimeout,
    volumes_for,
)
from vibe.docker_ctl import httpx as docker_ctl_httpx
from vibe.registry import ModelSpec


def make_spec(hf_repo: str) -> ModelSpec:
    """A minimal spec mirroring the registry's qwen entry."""
    return ModelSpec(
        alias="qwen2.5-coder-7b",
        display_name="Qwen2.5 Coder 7B Instruct (AWQ)",
        hf_repo=hf_repo,
        quantization="awq",
        max_model_len=32768,
        gpu_memory_utilization=0.70,
        container_name="vibe-qwen2.5-coder-7b",
        host_port=8030,
        extra_args=[],
    )


# -- pure volumes_for tests ------------------------------------------------


def test_volumes_for_remote_repo_id_is_hf_cache_only() -> None:
    """A repo id (`org/name`) gets exactly the HF-cache mount (M5.6)."""
    hf_home = Path("/host/hf-cache")
    volumes = volumes_for(make_spec("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"), hf_home)
    assert volumes == {
        str(hf_home): {"bind": "/root/.cache/huggingface", "mode": "rw"}
    }


def test_volumes_for_existing_local_path_adds_ro_bind(tmp_path: Path) -> None:
    """An absolute, existing local path adds a read-only bind at the resolved path."""
    model_dir = tmp_path / "laboratory" / "models" / "qwen-export"
    model_dir.mkdir(parents=True)
    # Symlink on purpose: the mount must use the resolved real path.
    link = tmp_path / "link"
    link.symlink_to(model_dir)

    volumes = volumes_for(make_spec(str(link)), Path("/host/hf-cache"))

    resolved = str(model_dir.resolve())
    assert volumes == {
        str(Path("/host/hf-cache")): {
            "bind": "/root/.cache/huggingface",
            "mode": "rw",
        },
        resolved: {"bind": resolved, "mode": "ro"},
    }
    assert volumes[resolved]["mode"] == "ro"
    assert volumes[resolved]["bind"] == resolved


def test_volumes_for_missing_local_path_raises(tmp_path: Path) -> None:
    """A missing absolute path raises DockerError instead of letting docker
    auto-create an empty directory (M5.6)."""
    spec = make_spec(str(tmp_path / "missing-model"))
    with pytest.raises(DockerError, match="local model path not found"):
        volumes_for(spec, Path("/host/hf-cache"))


def test_volumes_for_tilde_and_relative_paths_are_repo_ids() -> None:
    """Absolute-only is the contract: `~` and relative paths are repo ids (M5.6)."""
    hf_home = Path("/host/hf-cache")
    for hf_repo in ("~/models/qwen-export", "models/qwen-export", "org/name"):
        volumes = volumes_for(make_spec(hf_repo), hf_home)
        assert volumes == {
            str(hf_home): {"bind": "/root/.cache/huggingface", "mode": "rw"}
        }


# -- start() against a fake docker client ----------------------------------


class _FakeImages:
    """Minimal docker `images` API: the pinned image is always present."""

    def get(self, image: str) -> SimpleNamespace:
        return SimpleNamespace(id="sha256:" + "0" * 64)


class _FakeContainers:
    """Records `containers.run` kwargs; never talks to a daemon."""

    def __init__(self) -> None:
        self.run_kwargs: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> SimpleNamespace:
        self.run_kwargs.append(kwargs)
        return SimpleNamespace(id="c" * 12, name=kwargs.get("name"))


def test_start_mounts_local_model_path_read_only(tmp_path: Path) -> None:
    """`start()` forwards the local model volume into `containers.run` (M5.6)."""
    model_dir = tmp_path / "models" / "qwen-export"
    model_dir.mkdir(parents=True)
    hf_home = tmp_path / "hf"
    containers = _FakeContainers()
    client: Any = SimpleNamespace(images=_FakeImages(), containers=containers)
    ctl = DockerCtl(
        Settings(vllm_image_tag="v0.11.2", vibe_data_dir=str(hf_home)),
        client=client,
    )

    ctl.start(make_spec(str(model_dir)))

    assert len(containers.run_kwargs) == 1
    kwargs = containers.run_kwargs[0]
    resolved = str(model_dir.resolve())
    assert kwargs["volumes"][resolved] == {"bind": resolved, "mode": "ro"}
    assert kwargs["volumes"][str(hf_home)] == {
        "bind": "/root/.cache/huggingface",
        "mode": "rw",
    }
    # `--model <hf_repo>` flows through verbatim, and the local path is also
    # bind-mounted read-only at its resolved location (M5.6).
    assert kwargs["command"][:2] == ["--model", str(model_dir)]


def test_start_missing_local_path_raises_before_container_call(
    tmp_path: Path,
) -> None:
    """A missing local path fails `start()` before any container is created."""
    containers = _FakeContainers()
    client: Any = SimpleNamespace(images=_FakeImages(), containers=containers)
    ctl = DockerCtl(
        Settings(vllm_image_tag="v0.11.2", vibe_data_dir=str(tmp_path / "hf")),
        client=client,
    )

    with pytest.raises(DockerError, match="local model path not found"):
        ctl.start(make_spec(str(tmp_path / "does-not-exist")))

    assert containers.run_kwargs == []


# -- wait_ready against fakes (FR-00.10, IMP-02.9) ---------------------------


class _FakeWaitContainer:
    """Scriptable container: `attrs` status plus timestamped `logs` tail."""

    def __init__(self, status: str, log_text: str = "") -> None:
        self._status = status
        self._log_text = log_text

    @property
    def attrs(self) -> dict[str, Any]:
        return {"State": {"Status": self._status}}

    def logs(self, tail=None, timestamps=False) -> bytes:
        return self._log_text.encode()


class _FakeWaitContainers:
    """`containers.get` returning the scripted container."""

    def __init__(self, container: _FakeWaitContainer) -> None:
        self._container = container
        self.get_calls: list[str] = []

    def get(self, name: str) -> _FakeWaitContainer:
        self.get_calls.append(name)
        return self._container


class _FakeHttpResponse:
    def __init__(self, status_code: int, aliases: list[str]) -> None:
        self.status_code = status_code
        self._aliases = aliases

    def json(self) -> dict[str, Any]:
        return {"data": [{"id": alias} for alias in self._aliases]}


class _FakeAsyncClient:
    """Scriptable `httpx.AsyncClient` answering `/health` and `/v1/models`."""

    health_status = 200
    models_status = 200
    served_aliases: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        pass

    async def get(self, path: str) -> _FakeHttpResponse:
        if path == "/health":
            return _FakeHttpResponse(self.health_status, [])
        if path == "/v1/models":
            return _FakeHttpResponse(self.models_status, self.served_aliases)
        return _FakeHttpResponse(404, [])


def _wait_ctl(client: Any) -> DockerCtl:
    return DockerCtl(
        Settings(vllm_image_tag="v0.11.2", vibe_data_dir="/tmp/vibe-wait"),
        client=client,
    )


async def test_wait_ready_succeeds_when_alias_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/health` 200 and the alias in `/v1/models` → ready (FR-00.10)."""
    containers = _FakeWaitContainers(_FakeWaitContainer("running"))
    _FakeAsyncClient.health_status = 200
    _FakeAsyncClient.models_status = 200
    _FakeAsyncClient.served_aliases = ["qwen2.5-coder-7b"]
    monkeypatch.setattr(docker_ctl_httpx, "AsyncClient", _FakeAsyncClient)

    await _wait_ctl(
        SimpleNamespace(containers=containers)
    ).wait_ready(make_spec("org/repo"), timeout_s=5)

    assert containers.get_calls == ["vibe-qwen2.5-coder-7b"]


async def test_wait_ready_times_out_without_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healthy `/health` + `/v1/models` without the alias → ReadinessTimeout."""
    containers = _FakeWaitContainers(_FakeWaitContainer("running"))
    _FakeAsyncClient.health_status = 200
    _FakeAsyncClient.models_status = 200
    _FakeAsyncClient.served_aliases = ["some-other-model"]
    monkeypatch.setattr(docker_ctl_httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ReadinessTimeout) as excinfo:
        await _wait_ctl(
            SimpleNamespace(containers=containers)
        ).wait_ready(make_spec("org/repo"), timeout_s=0.05)
    assert "not ready within 0.05s" in str(excinfo.value)


async def test_wait_ready_times_out_when_health_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    containers = _FakeWaitContainers(_FakeWaitContainer("running"))
    _FakeAsyncClient.health_status = 404
    _FakeAsyncClient.models_status = 200
    _FakeAsyncClient.served_aliases = ["qwen2.5-coder-7b"]
    monkeypatch.setattr(docker_ctl_httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ReadinessTimeout):
        await _wait_ctl(
            SimpleNamespace(containers=containers)
        ).wait_ready(make_spec("org/repo"), timeout_s=0.05)


async def test_wait_ready_container_exited_with_gated_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead container raises ContainerExited with logs and the gated hint."""
    log_text = "line-a\nHTTP 401 Unauthorized\nline-c"
    containers = _FakeWaitContainers(_FakeWaitContainer("exited", log_text))
    _FakeAsyncClient.health_status = 404
    monkeypatch.setattr(docker_ctl_httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ContainerExited) as excinfo:
        await _wait_ctl(
            SimpleNamespace(containers=containers)
        ).wait_ready(make_spec("org/repo"), timeout_s=5)
    assert "exited before becoming ready" in str(excinfo.value)
    assert "gated" in str(excinfo.value)  # 401 hint (FR-00.11)
    assert "401" in excinfo.value.last_logs


async def test_wait_ready_last_logs_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`last_logs` is capped at _LAST_LOGS_CHARS (FR-00.11)."""
    log_text = "x" * 9000
    containers = _FakeWaitContainers(_FakeWaitContainer("exited", log_text))
    _FakeAsyncClient.health_status = 404
    monkeypatch.setattr(docker_ctl_httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ContainerExited) as excinfo:
        await _wait_ctl(
            SimpleNamespace(containers=containers)
        ).wait_ready(make_spec("org/repo"), timeout_s=5)
    assert len(excinfo.value.last_logs) == 8000


# -- daemon-gone degrade paths (IMP-02.10) ------------------------------------


class _DeadDaemonContainers:
    """`containers.list` raises the SDK's transport error (daemon restart)."""

    def list(self, **kwargs: Any):
        raise requests.ConnectionError("connection refused")


def test_daemon_gone_maps_to_docker_error() -> None:
    """A mid-session daemon restart surfaces as DockerError, not UNKNOWN."""
    client: Any = SimpleNamespace(containers=_DeadDaemonContainers())
    ctl = DockerCtl(
        Settings(vllm_image_tag="v0.11.2"),
        client=client,
    )
    with pytest.raises(DockerError) as excinfo:
        ctl.find_managed()
    assert "ConnectionError" in str(excinfo.value)
