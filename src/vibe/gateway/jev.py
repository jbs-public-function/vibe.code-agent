"""Pure Jev request rewrite + compatibility helpers (JEV-03.1).

No IO: this module imports nothing beyond the pure `vibe.jev` package, so the
whole rewrite table unit-tests without HTTP. `apply_jev` deep-copies a
chat-completions body and forces the OpenAI `response_format.json_schema`
form of the active wire schema; `check_compatible` returns the typed
rejection for requests that cannot coexist with a forced schema (`tools`,
`tool_choice`, `/v1/completions`).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from vibe.jev import schema_name_for_openai, wire_schema

# Client-supplied keys that conflict with the forced response format (doc 03):
# vLLM rejects a request carrying more than one structured-output constraint,
# so every one of these is removed — Jev is authoritative.
_REMOVED_KEYS = ("response_format", "structured_outputs")
_GUIDED_PREFIX = "guided_"


@dataclass(frozen=True)
class JevRewrite:
    """A rewritten request body plus the client keys that were removed."""

    data: dict[str, Any]
    overrode: list[str]


@dataclass(frozen=True)
class ErrorSpec:
    """A request that cannot coexist with the forced schema (JEV-03.1).

    `reason` is the short machine key for `jev_rejected` logs; `detail`
    completes the client-facing sentence "Jev is active (<name>); …" — the
    gateway fills in the active schema name at the call site, where it is
    known. The HTTP mapping is fixed: 400 with `error.code` set to `code`.
    """

    reason: str
    detail: str
    type: str = "invalid_request_error"
    code: str = "jev_incompatible_request"


def check_compatible(path: str, data: dict[str, Any]) -> ErrorSpec | None:
    """Reject requests that cannot be served under a forced schema (doc 03).

    - any `/v1/completions` request (a different vLLM param family; out of
      scope for v1),
    - non-empty `tools`,
    - `tool_choice` set to anything other than absent or `"none"`.

    Returns `None` when the request may be proxied with the schema injected.
    """
    if path == "/v1/completions":
        return ErrorSpec(
            reason="path",
            detail=(
                "/v1/completions is not supported in Jev mode. "
                "Detach with `vibe jev off`."
            ),
        )
    if data.get("tools"):
        return ErrorSpec(
            reason="tools",
            detail="tool calling is disabled. Detach with `vibe jev off`.",
        )
    if data.get("tool_choice") not in (None, "none"):
        return ErrorSpec(
            reason="tool_choice",
            detail="tool calling is disabled. Detach with `vibe jev off`.",
        )
    return None


def apply_jev(data: dict[str, Any], schema: dict[str, Any], name: str) -> JevRewrite:
    """Force the Jev `response_format` onto a chat-completions body (doc 03).

    Deep-copies the request; removes every client structured-output key
    (`response_format`, `structured_outputs`, any `guided_*`); injects the
    OpenAI `json_schema` form of the wire schema with `strict: true`. The
    `response_format.json_schema.name` prefers the schema title and falls
    back to the store name. Never raises.
    """
    out = deepcopy(data)
    overrode: list[str] = []
    for key in list(out):
        if key in _REMOVED_KEYS or key.startswith(_GUIDED_PREFIX):
            del out[key]
            overrode.append(key)
    out["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name_for_openai(schema, fallback=name),
            "schema": wire_schema(schema),
            "strict": True,
        },
    }
    return JevRewrite(data=out, overrode=overrode)
