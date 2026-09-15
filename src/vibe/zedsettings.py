"""Zed `settings.json` generator from the registry (M4.5, FR-04.8).

Renders the exact Phase-2 shape documented in the README ("Gateway & Zed
integration"): one `openai_compatible` provider `vibe-local` with an
`available_models` entry per registry alias, plus a default agent model
(the first alias tagged `tools`, else the first alias). Generating it from
`models.yaml` keeps the two from drifting by hand.

Entries are duck-typed (`alias`, `display_name`, `max_model_len`, `tags`)
so both `registry.ModelSpec` and the proto `ModelInfo` work.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Protocol

ZED_API_URL = "http://localhost:8080/v1"
PROVIDER_NAME = "vibe-local"

# Cap the advertised completion budget (Phase-2 README shape).
MAX_OUTPUT_TOKENS = 4096


class ZedEntry(Protocol):
    """Minimal shape shared by `ModelSpec` and proto `ModelInfo`.

    Read-only properties keep the protocol covariant, so both
    `ModelSpec` (pydantic) and `control_pb2.ModelInfo` (protobuf)
    satisfy it despite their different mutable tag containers.
    """

    @property
    def alias(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def max_model_len(self) -> int: ...

    @property
    def tags(self) -> Sequence[str]: ...


def _round_up_to_512(value: int) -> int:
    """Round `value` up to the next multiple of 512."""
    return math.ceil(value / 512) * 512


def _zed_model(entry: ZedEntry) -> dict[str, Any]:
    max_tokens = int(entry.max_model_len)
    output = min(
        MAX_OUTPUT_TOKENS, _round_up_to_512(max(max_tokens // 4, 1))
    )
    return {
        "name": entry.alias,
        "display_name": entry.display_name,
        "max_tokens": max_tokens,
        "max_output_tokens": output,
        "max_completion_tokens": output,
        "capabilities": {
            "tools": "tools" in entry.tags,
            "images": False,
            "parallel_tool_calls": False,
            "prompt_cache_key": False,
        },
    }


def build_zed_settings(entries: Sequence[ZedEntry]) -> dict[str, Any]:
    """Registry -> Zed `openai_compatible` settings block (FR-04.8).

    `available_models` keeps registry order (stable — never re-sorted).
    """
    models = [_zed_model(entry) for entry in entries]
    default_alias = ""
    if models:
        tools_alias = next(
            (model["name"] for model in models if model["capabilities"]["tools"]),
            None,
        )
        default_alias = tools_alias if tools_alias is not None else models[0]["name"]
    return {
        "language_models": {
            "openai_compatible": {
                PROVIDER_NAME: {
                    "api_url": ZED_API_URL,
                    "available_models": models,
                }
            }
        },
        "agent": {
            "default_model": {
                "provider": PROVIDER_NAME,
                "model": default_alias,
            }
        },
    }


def zed_settings_json(entries: Sequence[ZedEntry]) -> str:
    """The Zed settings block as pretty-printed JSON."""
    return json.dumps(build_zed_settings(entries), indent=2)
