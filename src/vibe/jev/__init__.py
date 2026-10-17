"""Jev: schema-constrained decision output (fable-5 improvements-2).

This package owns everything about a Jev *schema as data*: built-ins, the
single validator shared by CLI/controller/admin (JEV-01.1), content hashing,
and the on-disk user store. It imports no `docker`, `grpc`, `httpx`, or
`fastapi` — pure Python + `jsonschema` (J8/NFR-05.1).
"""

from __future__ import annotations

from vibe.jev.errors import JevError, StoreError
from vibe.jev.schema import (
    DRAFT_07,
    DRAFT_2019_09,
    DRAFT_2020_12,
    MAX_ENUM_VALUES,
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_DEPTH,
    RISKY_KEYWORDS,
    ValidationResult,
    canonical_hash,
    instance_validator_for,
    list_builtin,
    load_builtin,
    schema_name_for_openai,
    validate_schema_text,
    wire_schema,
)
from vibe.jev.store import (
    MAX_USER_SCHEMAS,
    SchemaEntry,
    SchemaStore,
    StoredSchema,
)

__all__ = [
    "DRAFT_07",
    "DRAFT_2019_09",
    "DRAFT_2020_12",
    "JevError",
    "MAX_ENUM_VALUES",
    "MAX_SCHEMA_BYTES",
    "MAX_SCHEMA_DEPTH",
    "MAX_USER_SCHEMAS",
    "RISKY_KEYWORDS",
    "SchemaEntry",
    "SchemaStore",
    "StoreError",
    "StoredSchema",
    "ValidationResult",
    "canonical_hash",
    "instance_validator_for",
    "list_builtin",
    "load_builtin",
    "schema_name_for_openai",
    "validate_schema_text",
    "wire_schema",
]
