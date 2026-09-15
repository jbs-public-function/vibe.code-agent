"""Tests for the docker adapter's bind mounts (FR-00.9, M5.6).

`volumes_for` is pure and unit-tested directly; `DockerCtl.start` is
exercised against an in-memory fake docker client (mirroring conftest's
`FakeDocker`, but faking the client one level lower so `containers.run`
kwargs are visible). No real Docker daemon is touched, and this module does
not import `docker`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vibe.config import Settings
from vibe.docker_ctl import DockerCtl, DockerError, volumes_for
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
