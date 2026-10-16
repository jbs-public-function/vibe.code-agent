"""Jev schema validation, hashing, and wire transforms (JEV-01).

Pure-Python and torch-free (NFR-05.1): this module deliberately imports
nothing from `docker`, `grpc`, `httpx`, or `fastapi` so the CLI, the
controller, and the admin UI share one validator and it unit-tests in
isolation (JEV-01.1). This is the single gatekeeper for Jev schema text
(JEV-01.4/JEV-01.5): every caller validates through `validate_schema_text`
so errors are identical everywhere, and the controller's result is the one
that counts.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterator

from jsonschema import Draft201909Validator, Draft202012Validator, Draft7Validator
from jsonschema.exceptions import SchemaError

# Hard limits protecting the controller (which runs `check_schema` on
# untrusted admin input) and vLLM (grammar compilation cost, JEV-01.4).
MAX_SCHEMA_BYTES = 64 * 1024  # 64 KiB
MAX_SCHEMA_DEPTH = 16
MAX_ENUM_VALUES = 256

# Supported `$schema` dialect URIs (JEV-01.4).
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
DRAFT_2019_09 = "https://json-schema.org/draft/2019-09/schema"
DRAFT_07 = "http://json-schema.org/draft-07/schema#"
SUPPORTED_DIALECTS = frozenset({DRAFT_2020_12, DRAFT_2019_09, DRAFT_07})

# Legacy URIs seen in the wild (the user's original example uses the bare
# site). Treated as draft-07 with a warning instead of being rejected.
_LEGACY_DIALECTS = frozenset(
    {
        "http://json-schema.org",
        "http://json-schema.org/schema",
        "http://json-schema.org/schema#",
        "https://json-schema.org/schema",
        "https://json-schema.org/schema#",
    }
)

# Keywords constrained-decoding backends (xgrammar) support poorly or not
# at all (JEV-01.5). Warning-only; refine from runbook evidence.
RISKY_KEYWORDS: tuple[str, ...] = (
    "if",
    "then",
    "else",
    "not",
    "patternProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "dependentSchemas",
    "dependentRequired",
)

_VALIDATORS = {
    DRAFT_2020_12: Draft202012Validator,
    DRAFT_2019_09: Draft201909Validator,
    DRAFT_07: Draft7Validator,
}

# OpenAI structured-output names: up to 64 chars of `[A-Za-z0-9_-]`.
OPENAI_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_BUILTIN_DIR = files("vibe.jev") / "builtin"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one schema text (JEV-01.3).

    Never raises on user input: problems land in `errors` (hard, `ok=False`)
    or `warnings` (non-fatal, never flip `ok`). `schema` is the parsed
    object — `None` only when the text was not valid JSON or not an object —
    and `sha256` is the canonical content hash (`""` when unparsable).
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    schema: dict[str, Any] | None
    sha256: str
    size_bytes: int


# -- structure walking -----------------------------------------------------


def _walk(node: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Yield every `(path, value)` in the document, depth-first."""
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, path + (index,))


def _pointer_repr(path: tuple[Any, ...]) -> str:
    """Render a location path as a `#/a/0/b` JSON-pointer string."""
    return "#" + "".join(f"/{part}" for part in path)


def _max_depth(node: Any) -> int:
    """Structural nesting depth; a lone scalar is 0, the root object is 1."""
    if isinstance(node, dict):
        return 1 + max((_max_depth(value) for value in node.values()), default=0)
    if isinstance(node, list):
        return 1 + max((_max_depth(value) for value in node), default=0)
    return 0


def _iter_refs(node: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], str]]:
    """Yield `(path, ref)` for every string-valued `$ref` keyword."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield path, value
            yield from _iter_refs(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_refs(value, path + (index,))


def _resolve_pointer(schema: dict[str, Any], fragment: str) -> tuple[Any, ...] | None:
    """Resolve a `#/a/b`-style local fragment to a canonical path (or None).

    ``fragment`` is the part after ``#`` (``""`` = the root).
    """
    if fragment == "":
        return ()
    pointer = fragment[1:] if fragment.startswith("/") else fragment
    segments = [
        segment.replace("~1", "/").replace("~0", "~")
        for segment in pointer.split("/")
    ]
    node: Any = schema
    path: list[Any] = []
    for segment in segments:
        if isinstance(node, dict) and segment in node:
            path.append(segment)
            node = node[segment]
        elif isinstance(node, list) and segment.isdigit() and int(segment) < len(node):
            index = int(segment)
            path.append(index)
            node = node[index]
        else:
            return None
    return tuple(path)


def _starts_with(path: tuple[Any, ...], prefix: tuple[Any, ...]) -> bool:
    """`prefix` is an ancestor-or-self location of `path`."""
    return path[: len(prefix)] == prefix


def _ref_cycles(schema: dict[str, Any]) -> list[str]:
    """Local `$ref`s that would make `jsonschema` recurse forever (JEV-01.4).

    Following a `$ref` at P re-validates the target's whole subtree, so P
    points at every ref location inside the target — including P itself when
    the target is an ancestor of P (`{"$ref": "#"}`). Cycles are found with
    a DFS over that graph.
    """
    refs = {path: ref for path, ref in _iter_refs(schema, ())}
    targets: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for path, ref in refs.items():
        if not ref.startswith("#"):
            continue  # remote refs are reported separately
        target = _resolve_pointer(schema, ref[1:])
        if target is not None:
            targets[path] = target

    adjacency: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {
        path: {
            other
            for other in refs
            if other == target or _starts_with(other, target)
        }
        for path, target in targets.items()
    }

    color: dict[tuple[Any, ...], int] = {}

    def visit(path: tuple[Any, ...]) -> bool:
        color[path] = 1  # in current DFS stack
        for nxt in adjacency.get(path, ()):
            if color.get(nxt, 0) == 0:
                if visit(nxt):
                    return True
            elif color[nxt] == 1:
                return True
        color[path] = 2
        return False

    cycles: list[str] = []
    for path in sorted(adjacency):
        if color.get(path, 0) == 0 and visit(path):
            cycles.append(
                f"circular $ref {refs[path]!r} at {_pointer_repr(path)}"
            )
    return cycles


def _enum_violations(schema: dict[str, Any]) -> list[str]:
    """Empty or oversized `enum`s as first-class errors (JEV-01.4)."""
    errors: list[str] = []
    for path, node in _walk(schema):
        if not isinstance(node, dict) or "enum" not in node:
            continue
        enum = node["enum"]
        if not isinstance(enum, list):
            continue  # the meta-schema reports non-array enums
        where = _pointer_repr(path + ("enum",))
        if not enum:
            errors.append(f"empty enum at {where}")
        elif len(enum) > MAX_ENUM_VALUES:
            errors.append(
                f"enum at {where} has {len(enum)} values; maximum is "
                f"{MAX_ENUM_VALUES}"
            )
    return errors


def _type_names(node: dict[str, Any]) -> set[str]:
    """The `type` keyword as a set of names (`"object"`/`["object","null"]`)."""
    raw = node.get("type")
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {value for value in raw if isinstance(value, str)}
    return set()


def _collect_warnings(schema: dict[str, Any]) -> list[str]:
    """Non-fatal hints about constrained-decoding friendliness (JEV-01.5)."""
    warnings: list[str] = []
    for path, node in _walk(schema):
        if not isinstance(node, dict):
            continue
        where = _pointer_repr(path)
        for keyword in RISKY_KEYWORDS:
            if keyword in node:
                warnings.append(
                    f"'{keyword}' at {where} is poorly supported by "
                    "constrained-decoding backends"
                )
        types = _type_names(node)
        if (
            "string" in types
            and "maxLength" not in node
            and "minLength" not in node
            and "enum" not in node
            and "const" not in node
        ):
            warnings.append(
                f"string without maxLength at {where} may run to max_tokens"
            )
        if "properties" in node and node.get("additionalProperties") is not False:
            warnings.append(
                f"object at {where} allows additional properties; set "
                "additionalProperties: false for tighter constrained decoding"
            )
    root_types = _type_names(schema)
    if (
        "object" in root_types
        and "required" not in schema
        and not any(key in schema for key in ("oneOf", "anyOf", "allOf", "$ref"))
    ):
        warnings.append("root object has no 'required' list")
    return warnings


# -- public API ------------------------------------------------------------


def validate_schema_text(text: str) -> ValidationResult:
    """Validate one schema text; **never raises** on user input (JEV-01.3).

    Hard errors (JEV-01.4): not valid JSON (message carries line/column),
    non-object root, unsupported `$schema` dialect, meta-schema failure,
    > 64 KiB, nesting deeper than 16, remote/unresolvable/circular `$ref`,
    empty or oversized `enum`. Warnings (JEV-01.5) never flip `ok`.
    """
    errors: list[str] = []
    warnings: list[str] = []
    size_bytes = len(text.encode("utf-8"))
    if size_bytes > MAX_SCHEMA_BYTES:
        errors.append(
            f"schema is {size_bytes} bytes; maximum is {MAX_SCHEMA_BYTES} bytes"
        )

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(
            f"not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}"
        )
        return ValidationResult(False, errors, warnings, None, "", size_bytes)
    if not isinstance(parsed, dict):
        errors.append(
            f"schema root must be a JSON object, got {type(parsed).__name__}"
        )
        return ValidationResult(False, errors, warnings, None, "", size_bytes)

    dialect = DRAFT_2020_12
    if "$schema" in parsed:
        declared = parsed["$schema"]
        if not isinstance(declared, str):
            errors.append("'$schema' must be a string dialect URI")
        elif declared in SUPPORTED_DIALECTS:
            dialect = declared
        elif declared in _LEGACY_DIALECTS:
            dialect = DRAFT_07
            warnings.append(
                f"legacy '$schema' URI {declared!r} treated as draft-07; "
                f"use {DRAFT_07} or newer"
            )
        else:
            errors.append(
                f"unsupported '$schema' dialect {declared!r}; supported: "
                + ", ".join(sorted(SUPPORTED_DIALECTS))
            )
    else:
        warnings.append(f"no '$schema' declared; assumed {DRAFT_2020_12}")

    if not errors:
        try:
            _VALIDATORS[dialect].check_schema(parsed)
        except SchemaError as exc:
            errors.append(f"schema fails meta-schema validation: {exc.message}")

    depth = _max_depth(parsed)
    if depth > MAX_SCHEMA_DEPTH:
        errors.append(
            f"schema nesting depth {depth} exceeds the maximum of {MAX_SCHEMA_DEPTH}"
        )

    for path, ref in _iter_refs(parsed, ()):
        if not ref.startswith("#"):
            errors.append(
                f"remote $ref {ref!r} at {_pointer_repr(path + ('$ref',))} "
                "is not allowed (only local '#/...' refs)"
            )
    for path, ref in _iter_refs(parsed, ()):
        if ref.startswith("#") and _resolve_pointer(parsed, ref[1:]) is None:
            errors.append(
                f"unresolvable $ref {ref!r} at {_pointer_repr(path + ('$ref',))}"
            )
    errors.extend(_ref_cycles(parsed))
    errors.extend(_enum_violations(parsed))

    warnings.extend(_collect_warnings(parsed))
    sha256 = canonical_hash(parsed)
    return ValidationResult(not errors, errors, warnings, parsed, sha256, size_bytes)


def canonical_hash(schema: dict[str, Any]) -> str:
    """Content hash: key order and whitespace do not change it (JEV-01.6)."""
    text = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Deep copy with `$schema` removed; nothing else touched (JEV-01.7).

    Some structured-output backends choke on the `$schema` keyword, so the
    wire form (gateway injection, probe) drops only that key.
    """
    out = deepcopy(schema)
    out.pop("$schema", None)
    return out


def schema_name_for_openai(schema: dict[str, Any], fallback: str = "jev") -> str:
    """The OpenAI `response_format.json_schema.name` for a schema (JEV-01.8).

    Prefers the schema `title` (sanitised to `^[A-Za-z0-9_-]{1,64}$`), then
    the store name, then ``"jev"``.
    """

    def sanitise(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "_", value.strip())[:64]

    title = schema.get("title")
    if isinstance(title, str):
        cleaned = sanitise(title)
        if OPENAI_NAME_RE.match(cleaned):
            return cleaned
    cleaned = sanitise(fallback or "")
    if OPENAI_NAME_RE.match(cleaned):
        return cleaned
    return "jev"


def instance_validator_for(schema: dict[str, Any]) -> Any:
    """The `jsonschema` validator class matching the schema's dialect.

    Used to validate *instances* (probe output) against a stored schema;
    legacy `$schema` URIs are treated as draft-07 like
    `validate_schema_text` does.
    """
    declared = schema.get("$schema")
    if declared in _LEGACY_DIALECTS:
        return Draft7Validator
    for dialect, validator in _VALIDATORS.items():
        if declared == dialect:
            return validator
    return Draft202012Validator


# -- built-ins -------------------------------------------------------------


def list_builtin() -> list[str]:
    """Names of every built-in schema shipped as package data (JEV-01.2)."""
    names: list[str] = []
    for path in _BUILTIN_DIR.iterdir():
        if path.is_file() and path.name.endswith(".json"):
            names.append(path.name[: -len(".json")])
    return sorted(names)


def load_builtin(name: str) -> dict[str, Any]:
    """Read one built-in schema from package data; `KeyError` if unknown."""
    try:
        text = (_BUILTIN_DIR / f"{name}.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        raise KeyError(name) from None
    data = json.loads(text)
    if not isinstance(data, dict):
        raise KeyError(name)
    return data
