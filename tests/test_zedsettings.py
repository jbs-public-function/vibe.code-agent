"""Zed settings JSON generator tests (M4.5, FR-04.8).

The expected values mirror the Phase-2 README snippet ("Gateway & Zed
integration"): qwen/llama advertise 4096 output tokens and tools, deepseek
advertises 2048 without tools, and the agent default is `qwen2.5-coder-7b`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from vibe.gen.vibe.v1 import control_pb2
from vibe.registry import Registry
from vibe.zedsettings import build_zed_settings, zed_settings_json


def _vibe_local(zed: dict):
    return zed["language_models"]["openai_compatible"]["vibe-local"]


def test_build_matches_readme_values(valid_registry_dict: dict) -> None:
    """The repo's three models render with the README's numbers (FR-04.8)."""
    registry = Registry.from_dict(valid_registry_dict)
    zed = build_zed_settings(registry.all())

    vibe_local = _vibe_local(zed)
    assert vibe_local["api_url"] == "http://localhost:8080/v1"
    models = vibe_local["available_models"]
    # Registry order preserved, never re-sorted.
    assert [model["name"] for model in models] == [
        "qwen2.5-coder-7b",
        "llama-3.1-8b",
        "deepseek-r1-14b",
    ]

    qwen, llama, deepseek = models
    assert qwen["max_tokens"] == 32768
    assert qwen["max_output_tokens"] == 4096
    assert qwen["max_completion_tokens"] == 4096
    assert llama["max_tokens"] == 16384
    assert llama["max_output_tokens"] == 4096
    assert llama["max_completion_tokens"] == 4096
    assert deepseek["max_tokens"] == 7984
    assert deepseek["max_output_tokens"] == 2048
    assert deepseek["max_completion_tokens"] == 2048

    assert qwen["capabilities"] == {
        "tools": True,
        "images": False,
        "parallel_tool_calls": False,
        "prompt_cache_key": False,
    }
    assert llama["capabilities"]["tools"] is True
    assert deepseek["capabilities"]["tools"] is False

    assert zed["agent"]["default_model"] == {
        "provider": "vibe-local",
        "model": "qwen2.5-coder-7b",
    }


def test_output_tokens_round_up_to_512(valid_registry_dict: dict) -> None:
    """max_output_tokens = min(4096, ceil((mml // 4) / 512) * 512)."""
    entry = {
        "alias": "tiny",
        "display_name": "Tiny",
        "hf_repo": "org/tiny",
        "max_model_len": 1024,  # 1024 // 4 = 256 → 512
        "gpu_memory_utilization": 0.5,
        "container_name": "vibe-tiny",
        "host_port": 8001,
    }
    zed = build_zed_settings([Registry.from_dict({"models": [entry]}).get("tiny")])
    tiny = _vibe_local(zed)["available_models"][0]
    assert tiny["max_output_tokens"] == 512
    assert tiny["max_completion_tokens"] == 512


def test_duck_typed_entries_and_default_fallback() -> None:
    """Any object with alias/display_name/max_model_len/tags works."""
    entries: list[Any] = [
        SimpleNamespace(alias="a", display_name="A", max_model_len=2048, tags=[]),
        SimpleNamespace(
            alias="b", display_name="B", max_model_len=8192, tags=["tools"]
        ),
    ]
    zed = build_zed_settings(entries)
    models = _vibe_local(zed)["available_models"]
    assert [model["name"] for model in models] == ["a", "b"]
    # First alias tagged tools wins as the agent default.
    assert zed["agent"]["default_model"] == {"provider": "vibe-local", "model": "b"}

    # Without any tools-tagged model, the first alias is the default.
    zed = build_zed_settings(entries[:1])
    assert zed["agent"]["default_model"]["model"] == "a"


def test_proto_model_info_entries() -> None:
    """The proto `ModelInfo` satisfies the same duck-typed shape."""
    model = control_pb2.ModelInfo(
        alias="proto-model",
        display_name="Proto Model",
        max_model_len=4096,
        tags=["tools"],
    )
    zed = build_zed_settings([model])
    rendered = _vibe_local(zed)["available_models"][0]
    assert rendered["name"] == "proto-model"
    assert rendered["max_tokens"] == 4096
    assert rendered["max_output_tokens"] == 1024
    assert rendered["capabilities"]["tools"] is True
    assert zed["agent"]["default_model"]["model"] == "proto-model"


def test_zed_settings_json_round_trips(valid_registry_dict: dict) -> None:
    registry = Registry.from_dict(valid_registry_dict)
    text = zed_settings_json(registry.all())
    assert json.loads(text) == build_zed_settings(registry.all())
    assert "\n  " in text  # indent=2
