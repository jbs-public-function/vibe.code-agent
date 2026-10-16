"""Quick-builder generator for the Jev admin editor (JEV-05.13, E11).

Doc 05 asks for this generator to live in `vibe.jev.builder`; it lives here
for now because the `vibe.jev` package is owned by the J0/J1 phases. The
module is deliberately tiny and imports only stdlib + one constant from
`vibe.jev.schema`, so it can move to `vibe.jev.builder` unchanged when the
J0/J1 owner picks it up.

With the default options the output is semantically equal to the built-in
`default` schema: the same discriminated `oneOf` of choice/boolean/score
branches, each with `additionalProperties: false` and a 0..1 confidence
(E11 compares canonical hashes).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vibe.jev.schema import DRAFT_2020_12

BUILDER_TITLE = "JevDecisionOutput"

# JEV-05.13: the three decision types, in the built-in's canonical order.
DECISION_TYPES = ("choice", "boolean", "score")


def _number_text(value: float) -> str:
    """0.0/1.0 render as `0`/`1` so the default description matches the built-in."""
    if value == int(value):
        return str(int(value))
    return repr(value)


def build_decision_schema(
    decision_types: Sequence[str],
    choice_options: Sequence[str],
    score_min: float,
    score_max: float,
    include_confidence: bool = True,
    additional_properties_false: bool = True,
) -> str:
    """Compose the discriminated decision schema (choice/boolean/score).

    With `decision_types == ("choice", "boolean", "score")`, options
    `["YES", "NO"]`, score 0..1, confidence on and `additionalProperties:
    false` the output equals the built-in `default` semantically.
    """
    branches: list[dict[str, Any]] = []
    for kind in DECISION_TYPES:
        if kind not in decision_types:
            continue
        properties: dict[str, dict[str, Any]] = {"decision_type": {"const": kind}}
        if kind == "choice":
            properties["result"] = {
                "type": "string",
                "enum": list(choice_options),
                "description": "The chosen action.",
            }
        elif kind == "boolean":
            properties["result"] = {"type": "boolean"}
        else:
            properties["result"] = {
                "type": "number",
                "minimum": score_min,
                "maximum": score_max,
            }
        if include_confidence:
            properties["confidence"] = {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            }
        branch: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
        }
        if additional_properties_false:
            branch["additionalProperties"] = False
        branches.append(branch)

    parts: list[str] = []
    if "choice" in decision_types:
        parts.append(f"a {'/'.join(choice_options)} choice")
    if "boolean" in decision_types:
        parts.append("a boolean verdict")
    if "score" in decision_types:
        parts.append(f"a {_number_text(score_min)}..{_number_text(score_max)} score")
    if len(parts) == 1:
        joined = parts[0]
    elif len(parts) == 2:
        joined = " or ".join(parts)
    else:
        joined = ", ".join(parts[:-1]) + ", or " + parts[-1]

    description = f"Default Jev decision: {joined}"
    if include_confidence:
        description += ", with a confidence"
    description += "."

    schema: dict[str, Any] = {
        "$schema": DRAFT_2020_12,
        "title": BUILDER_TITLE,
        "description": description,
        "oneOf": branches,
    }
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
