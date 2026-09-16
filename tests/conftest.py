"""Shared fixtures for the vibe test suite.

No tests import `docker` or `pynvml` directly; all docker interaction goes
through the in-memory `FakeDocker` below.
"""

from __future__ import annotations

import types

import pytest

from vibe.config import Settings
from vibe.registry import Registry


@pytest.fixture
def valid_registry_dict() -> dict:
    """Three registry entries mirroring the repo's `models.yaml`."""
    return {
        "models": [
            {
                "alias": "qwen2.5-coder-7b",
                "display_name": "Qwen2.5 Coder 7B Instruct (AWQ)",
                "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                "quantization": "awq",
                "max_model_len": 32768,
                "gpu_memory_utilization": 0.70,
                "container_name": "vibe-qwen2.5-coder-7b",
                "host_port": 8030,
                "extra_args": [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "hermes",
                ],
                "use_case": "Code generation, repository parsing, Zed agent default",
                "tags": ["code", "tools"],
            },
            {
                "alias": "llama-3.1-8b",
                "display_name": "Llama 3.1 8B Instruct (AWQ INT4)",
                "hf_repo": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                "quantization": "awq",
                "max_model_len": 16384,
                "gpu_memory_utilization": 0.75,
                "container_name": "vibe-llama-3.1-8b",
                "host_port": 8020,
                "extra_args": [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "llama3_json",
                ],
                "use_case": "Writing, brainstorming, general chat",
                "tags": ["general", "tools", "gated"],
            },
            {
                "alias": "deepseek-r1-14b",
                "display_name": "DeepSeek R1 Distill Qwen 14B (AWQ)",
                "hf_repo": "casperhansen/DeepSeek-R1-Distill-Qwen-14B-AWQ",
                "quantization": "awq",
                "max_model_len": 7984,
                "gpu_memory_utilization": 0.85,
                "container_name": "vibe-deepseek-r1-14b",
                "host_port": 8010,
                "extra_args": [],
                "use_case": "Deep reasoning, logic problems",
                "tags": ["reasoning"],
            },
        ]
    }


@pytest.fixture
def settings() -> Settings:
    """Settings with a pinned image tag.

    Init kwargs take priority over environment variables and `.env`
    (pydantic-settings), so passing `vllm_image_tag` here is robust even if a
    developer's `.env` grows a `VLLM_IMAGE_TAG` later.
    """
    return Settings(vllm_image_tag="v0.11.2")


@pytest.fixture
def registry(valid_registry_dict: dict) -> Registry:
    return Registry.from_dict(valid_registry_dict)


class FakeDocker:
    """In-memory stand-in for `vibe.docker_ctl.DockerCtl`.

    Every method except `wait_ready` is plain-synchronous, matching how
    `JobRunner` dispatches them through `asyncio.to_thread`. Tests tweak
    attributes (`managed`, `purge_names`, `info_by_name`) to script behavior.
    """

    def __init__(
        self,
        managed: list | None = None,
        purge_names: list[str] | None = None,
    ) -> None:
        self.managed: list = list(managed or [])
        self.purge_names: list[str] = list(purge_names or [])
        self.started: list = []  # (spec, container) pairs from start()
        self.stopped: list = []  # containers passed to stop()
        self.wait_ready_calls: list = []  # (spec, timeout_s) pairs
        # When set, `wait_ready` raises it (scriptable ContainerExited /
        # ReadinessTimeout for JobRunner error-path tests).
        self.wait_ready_error: Exception | None = None
        self.info_by_name: dict[str, dict] = {}
        self.log_lines: list[str] = []

    def find_managed(self) -> list:
        return list(self.managed)

    def purge_all(self) -> list[str]:
        return list(self.purge_names)

    def start(self, spec):
        container = types.SimpleNamespace(id="c" * 12, name=spec.container_name)
        self.started.append((spec, container))
        return container

    def stop(self, container) -> None:
        self.stopped.append(container)

    async def wait_ready(self, spec, timeout_s: int) -> None:
        self.wait_ready_calls.append((spec, timeout_s))
        if self.wait_ready_error is not None:
            raise self.wait_ready_error

    def container_info(self, container) -> dict:
        default = {
            "name": container.name,
            "id": getattr(container, "id", "c" * 12),
            "status": "running",
            "started_at": None,
            "host_port": None,
            "image": "vllm/vllm-openai:v0.11.2",
            "alias": None,
        }
        return dict(self.info_by_name.get(container.name, default))

    def logs(self, container, tail: int | None = None, follow: bool = False):
        return iter(self.log_lines)


@pytest.fixture
def fake_docker() -> FakeDocker:
    """A fresh, unconfigured FakeDocker."""
    return FakeDocker()
