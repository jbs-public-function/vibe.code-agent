"""OpenAI-style `/v1/models` payload aggregation (FR-03.2).

The gateway reports every registry alias — not just the active model — as an
OpenAI `model` object, annotated with vibe extension fields (`vibe_state`,
`vibe_display_name`, `vibe_max_model_len`, `vibe_active`, `vibe_cached`,
`vibe_jev`) so clients like Zed can show a richer listing and pick the right
`max_tokens`.
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
    JEV-03.6: the active entry carries `vibe_jev` with the attached schema's
    provenance when Jev is active; every other entry — and the active one
    when Jev is off — carries `vibe_jev: null`.
    """
    active_jev = status.jev if status is not None and status.HasField("jev") else None
    data: list[dict[str, object]] = []
    for info in infos:
        jev: dict[str, object] | None = None
        if (
            active_jev is not None
            and status is not None
            and status.active_alias == info.alias
        ):
            jev = {
                "name": active_jev.schema_name,
                "sha256": active_jev.sha256,
                "source": active_jev.source,
                "stale": active_jev.stale,
            }
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
                "vibe_jev": jev,
            }
        )
    return {"object": "list", "data": data}
