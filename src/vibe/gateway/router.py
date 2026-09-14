"""Pure routing decision table for the gateway POST routes
(FR-03.3/03.5/03.6/03.7/03.8).

`decide` maps a request's `model` field, the controller's active-model
snapshot, the swap-on-request setting and any in-flight swap to one of:

- `Proxy(upstream)` — forward the request to the active vLLM container
- `Retry(status, after, error)` — answer the client with an OpenAI-style
  error body and an optional `Retry-After` header
- `Swap(alias)` — hand off to the swap coordinator, which loads `alias`
  before the request is proxied

The module is deliberately free of network/IO so the whole table is
unit-testable.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import state_name


@dataclass(frozen=True)
class Proxy:
    """Forward the request to this upstream base URL."""

    upstream: str  # e.g. "http://127.0.0.1:8030"


@dataclass(frozen=True)
class Retry:
    """Answer the client directly with an OpenAI-style error body."""

    status: int
    after: int  # Retry-After seconds; 0 means no header
    error: dict[str, str | None]  # {"message": str, "type": str, "code": str | None}


@dataclass(frozen=True)
class Swap:
    """Load the model first, then proxy (swap-on-request, FR-03.6)."""

    alias: str


RouteDecision = Proxy | Retry | Swap


def decide(
    model: str | None,
    known_aliases: Collection[str],
    status: control_pb2.Status,
    swap_on_request: bool,
    in_flight_alias: str | None,
) -> RouteDecision:
    """Pick the route for a POST request naming `model` (decision table).

    - missing/empty/non-string `model` → 400
    - unknown alias → 404 with the valid aliases
    - active alias, READY with a port → `Proxy`
    - active alias, any other state (or port 0) → 503, Retry-After 10
    - known-but-inactive alias:
        - swap off → 503, Retry-After 30 (`model_not_loaded`)
        - swap on, no job or job for this alias → `Swap` (start or attach)
        - swap on, job for a different alias → 503, Retry-After 30
    """
    if not isinstance(model, str) or not model:
        return Retry(
            400,
            0,
            {
                "message": "missing 'model' field",
                "type": "invalid_request_error",
                "code": None,
            },
        )
    if model not in known_aliases:
        valid = ", ".join(known_aliases)
        return Retry(
            404,
            0,
            {
                "message": f"model '{model}' not found; valid aliases: {valid}",
                "type": "model_not_found",
                "code": "model_not_found",
            },
        )
    if model == status.active_alias:
        if status.state == control_pb2.READY and status.host_port > 0:
            return Proxy(f"http://127.0.0.1:{status.host_port}")
        return Retry(
            503,
            10,
            {
                "message": (
                    f"model '{model}' is {state_name(status.state)}; retry shortly"
                ),
                "type": "model_not_ready",
                "code": "model_not_ready",
            },
        )
    # Known but not active.
    if not swap_on_request:
        if status.active_alias:
            message = (
                f"model {model} is not loaded (active: {status.active_alias}). "
                f"Run: vibe load {model}"
            )
        else:
            message = f"no model loaded. Run: vibe load {model}"
        return Retry(
            503,
            30,
            {
                "message": message,
                "type": "model_not_loaded",
                "code": "model_not_loaded",
            },
        )
    if in_flight_alias is None or in_flight_alias == model:
        return Swap(model)
    return Retry(
        503,
        30,
        {
            "message": f"swap to '{in_flight_alias}' is in progress",
            "type": "swap_in_progress",
            "code": "swap_in_progress",
        },
    )
