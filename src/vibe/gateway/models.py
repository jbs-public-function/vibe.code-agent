"""OpenAI-style `/v1/models` payload aggregation (FR-03.2).

The gateway reports every registry alias — not just the active model — as an
OpenAI `model` object, annotated with vibe extension fields (`vibe_state`,
`vibe_display_name`, `vibe_max_model_len`, `vibe_active`, `vibe_cached`) so
clients like Zed can show a richer listing and pick the right `max_tokens`.
"""

from __future__ import annotations

from collections.abc import Sequence

from vibe.gen.vibe.v1 import control_pb2


def state_name(state: int) -> str:
    """Human-readable `ModelState` name; unset (0) reads as `AVAILABLE`.

    proto3 enum fields default to `MODEL_STATE_UNSPECIFIED` (0) when absent,
    and a model with no container is effectively `AVAILABLE`.
    """
    if state == control_pb2.MODEL_STATE_UNSPECIFIED:
        return "AVAILABLE"
    return control_pb2.ModelState.Name(state)


def build_models_payload(
    infos: Sequence[control_pb2.ModelInfo], status: control_pb2.Status | None
) -> dict[str, object]:
    """Build the OpenAI `list models` body for every registered model.

    `status` carries the active-model snapshot; when it is `None` (controller
    state unknown) every entry is reported inactive in state `AVAILABLE`.
    """
    data: list[dict[str, object]] = []
    for info in infos:
        data.append(
            {
                "id": info.alias,
                "object": "model",
                "created": 0,
                "owned_by": "vibe",
                "vibe_state": (
                    "AVAILABLE" if status is None else state_name(info.state)
                ),
                "vibe_display_name": info.display_name,
                "vibe_max_model_len": info.max_model_len,
                "vibe_active": (
                    status is not None and status.active_alias == info.alias
                ),
                "vibe_cached": info.cached_locally,
            }
        )
    return {"object": "list", "data": data}
