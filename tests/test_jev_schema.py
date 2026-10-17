"""Unit tests for `vibe.jev.schema` — validation, hashing, wire form (doc 06 §A)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from vibe.jev import (
    RISKY_KEYWORDS,
    canonical_hash,
    list_builtin,
    load_builtin,
    schema_name_for_openai,
    validate_schema_text,
    wire_schema,
)
from vibe.jev.schema import MAX_SCHEMA_BYTES, MAX_SCHEMA_DEPTH

# The user's original example schema (doc 01 "as given"), used by A3.
ORIGINAL_EXAMPLE = {
    "$schema": "http://json-schema.org",
    "title": "JevDecisionOutput",
    "type": "object",
    "properties": {
        "decision_type": {"type": "string", "enum": ["choice", "score", "boolean"]},
        "result": {
            "anyOf": [
                {"type": "string", "enum": ["option_a", "option_b", "option_c"]},
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {"type": "boolean"},
            ]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["decision_type", "result", "confidence"],
    "additionalProperties": False,
}


def _text(schema: object) -> str:
    return json.dumps(schema)


# -- A1 / A2: built-ins ------------------------------------------------------


def test_builtin_default_validates_with_yes_no_choice() -> None:
    """A1: the built-in `default` has zero errors and a YES/NO choice branch."""
    schema = load_builtin("default")
    result = validate_schema_text(_text(schema))
    assert result.ok, result.errors
    assert result.errors == []
    assert result.schema is not None
    branches = result.schema["oneOf"]
    assert len(branches) == 3
    choice = next(
        branch for branch in branches if branch["properties"]["decision_type"]["const"] == "choice"
    )
    assert choice["properties"]["result"]["enum"] == ["YES", "NO"]
    assert result.sha256 == canonical_hash(schema)


def test_every_preset_validates() -> None:
    """A2: every file in `builtin/` parses and validates with zero errors."""
    names = list_builtin()
    assert names == ["default", "yes-no", "yes-no-unsure"]
    for name in names:
        result = validate_schema_text(_text(load_builtin(name)))
        assert result.ok, f"{name}: {result.errors}"
        assert result.warnings == [], f"{name}: {result.warnings}"


# -- A3: the coupling fix -----------------------------------------------------


def test_original_example_validates_but_does_not_couple_fields() -> None:
    """A3: the original shape validates; the default rejects mismatched pairs."""
    original = validate_schema_text(_text(ORIGINAL_EXAMPLE))
    assert original.ok, original.errors
    assert any("legacy '$schema'" in warning for warning in original.warnings)

    default = load_builtin("default")
    doc = {"decision_type": "score", "result": True, "confidence": 0.5}
    assert Draft202012Validator(default).is_valid(doc) is False
    assert Draft202012Validator(ORIGINAL_EXAMPLE).is_valid(doc) is True


# -- A4: hard errors ----------------------------------------------------------


def test_error_not_json_reports_line_and_column() -> None:
    result = validate_schema_text('{"type": "object",\n "bad"')
    assert not result.ok
    assert result.schema is None
    assert any("not valid JSON" in error and "line" in error and "column" in error for error in result.errors)


def test_error_root_not_object() -> None:
    result = validate_schema_text("[1, 2, 3]")
    assert not result.ok
    assert result.schema is None
    assert any("root must be a JSON object" in error for error in result.errors)


def test_error_unknown_dialect() -> None:
    result = validate_schema_text(
        _text({"$schema": "https://example.com/not-a-dialect", "type": "object"})
    )
    assert not result.ok
    assert any("unsupported '$schema' dialect" in error for error in result.errors)


def test_error_bad_type_keyword() -> None:
    result = validate_schema_text(_text({"type": "strng"}))
    assert not result.ok
    assert any("meta-schema" in error for error in result.errors)


def test_error_empty_enum() -> None:
    result = validate_schema_text(_text({"type": "string", "enum": []}))
    assert not result.ok
    assert any("empty enum" in error for error in result.errors)


def test_error_enum_too_many_values() -> None:
    result = validate_schema_text(_text({"type": "string", "enum": list(range(300))}))
    assert not result.ok
    assert any("maximum is 256" in error for error in result.errors)


def test_error_depth_limit() -> None:
    def nested(wraps: int) -> dict:
        schema = {"type": "string"}
        for _ in range(wraps):
            schema = {"type": "object", "properties": {"a": schema}}
        return schema

    shallow = nested(MAX_SCHEMA_DEPTH // 2 - 1)
    assert validate_schema_text(_text(shallow)).ok
    deep = nested(MAX_SCHEMA_DEPTH)
    result = validate_schema_text(_text(deep))
    assert not result.ok
    assert any("nesting depth" in error for error in result.errors)


def test_error_size_limit() -> None:
    padding = "x" * (MAX_SCHEMA_BYTES + 1)
    result = validate_schema_text(_text({"type": "object", "description": padding}))
    assert not result.ok
    assert any("maximum is" in error for error in result.errors)


def test_error_remote_ref() -> None:
    result = validate_schema_text(_text({"$ref": "https://example.com/other.json"}))
    assert not result.ok
    assert any("remote $ref" in error for error in result.errors)


def test_error_unresolvable_local_ref() -> None:
    result = validate_schema_text(_text({"$ref": "#/defs/missing"}))
    assert not result.ok
    assert any("unresolvable $ref" in error for error in result.errors)


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#"},  # self-reference
        {  # mutual reference through #/defs
            "defs": {
                "a": {"$ref": "#/defs/b"},
                "b": {"$ref": "#/defs/a"},
            },
            "type": "object",
        },
    ],
)
def test_error_circular_ref(schema: dict) -> None:
    result = validate_schema_text(_text(schema))
    assert not result.ok
    assert any("circular $ref" in error for error in result.errors)


# -- A5: warnings -------------------------------------------------------------


def test_warnings_never_flip_ok() -> None:
    result = validate_schema_text(
        _text({"type": "object", "properties": {"a": {"type": "string"}}})
    )
    assert result.ok
    warnings = "\n".join(result.warnings)
    assert "additional properties" in warnings
    assert "may run to max_tokens" in warnings
    assert "no '$schema' declared" in warnings


def test_warning_risky_keywords() -> None:
    subschema: dict[str, object] = {
        "if": {"type": "string"},
        "then": {"type": "string"},
        "else": {"type": "string"},
        "not": {"type": "string"},
        "patternProperties": {"^a": {"type": "string"}},
        "unevaluatedProperties": {"type": "string"},
        "unevaluatedItems": {"type": "string"},
        "dependentSchemas": {"a": {"type": "string"}},
        "dependentRequired": {"a": ["b"]},
    }
    for keyword in RISKY_KEYWORDS:
        result = validate_schema_text(_text({keyword: subschema[keyword]}))
        assert result.ok, (keyword, result.errors)
        assert any(f"'{keyword}'" in warning for warning in result.warnings), keyword


def test_warning_missing_required_on_root_object() -> None:
    result = validate_schema_text(
        _text({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False})
    )
    assert result.ok
    assert any("no 'required' list" in warning for warning in result.warnings)


# -- A6 / A7 / A8: hashing, wire form, naming ---------------------------------


def test_canonical_hash_invariant_and_sensitive() -> None:
    a = {"b": {"c": 2}, "a": 1, "s": "naïve"}
    b = {"a": 1, "s": "naïve", "b": {"c": 2}}
    assert canonical_hash(a) == canonical_hash(b)
    assert len(canonical_hash(a)) == 64
    changed = {**a, "a": 2}
    assert canonical_hash(changed) != canonical_hash(a)


def test_wire_schema_strips_only_dollar_schema() -> None:
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "T", "type": "object"}
    wire = wire_schema(schema)
    assert "$schema" not in wire
    assert wire == {"title": "T", "type": "object"}
    assert schema["$schema"]  # input untouched
    wire["title"] = "mutated"
    assert schema["title"] == "T"


def test_schema_name_for_openai_sanitisation() -> None:
    assert schema_name_for_openai({"title": "JevDecisionOutput"}) == "JevDecisionOutput"
    assert schema_name_for_openai({"title": "My Decision!"}) == "My_Decision_"
    assert schema_name_for_openai({"title": "Décision"}) == "D_cision"
    long_title = "A" * 100
    assert schema_name_for_openai({"title": long_title}) == "A" * 64
    assert schema_name_for_openai({}, fallback="store-name") == "store-name"
    assert schema_name_for_openai({}, fallback="") == "jev"
    assert schema_name_for_openai({}) == "jev"


# -- A11 / A12: packaging and import graph ------------------------------------


def test_builtins_loadable_via_importlib_resources() -> None:
    """A11: package data (builtin/*.json) is shipped with the wheel."""
    for name in list_builtin():
        schema = load_builtin(name)
        assert isinstance(schema, dict) and schema


def test_jev_imports_no_docker_grpc_httpx_fastapi() -> None:
    """A12: `vibe.jev` pulls in none of the control-plane heavyweights."""
    code = (
        "import sys; import vibe.jev;"
        "print(','.join(sorted(m for m in ('docker','grpc','httpx','fastapi')"
        " if m in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""
