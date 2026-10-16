"""On-disk store for user Jev schemas (JEV-01.9/JEV-01.10).

User schemas live at `<root>/<name>.json`; a user file whose name matches a
built-in **shadows** it (source ``override``) and deleting it reverts to the
built-in. Writes are atomic (temp file + `os.replace`) so a crash can never
leave a truncated schema behind. The controller's Jev state file
(`.active.json`) lives in the same directory but is never addressed by name.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vibe.jev.errors import StoreError
from vibe.jev.schema import (
    ValidationResult,
    canonical_hash,
    list_builtin,
    load_builtin,
    validate_schema_text,
)

# Names match this (JEV-01.9): lowercase start, then lowercase/digit/._-,
# at most 64 chars. Leading dots and path separators are structurally
# impossible, so `.active.json` can never be addressed.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# Reserved names collide with admin routes (doc 05).
RESERVED_NAMES = frozenset({"new", "launch", "detach", "build"})

# Refuse to exceed this many user schema files (JEV-01.10).
MAX_USER_SCHEMAS = 256

# The controller's Jev state file — inside the store dir but not a schema.
ACTIVE_FILE = ".active.json"


@dataclass(frozen=True)
class SchemaEntry:
    """One visible schema: name + origin + content fingerprint (JEV-01.9)."""

    name: str
    source: str  # "builtin" | "user" | "override"
    sha256: str
    size: int
    mtime: float  # unix seconds; 0.0 for built-ins
    title: str = ""


@dataclass(frozen=True)
class StoredSchema:
    """A resolved schema body plus provenance (result of `get`)."""

    name: str
    source: str  # "builtin" | "user" | "override"
    schema: dict[str, Any]
    sha256: str
    size: int
    mtime: float


def _check_name(name: str) -> str:
    """Validate a store name, raising `StoreError("invalid_name", …)`."""
    if not NAME_RE.match(name):
        raise StoreError(
            "invalid_name",
            f"invalid schema name {name!r}: must match {NAME_RE.pattern} "
            "(lowercase letters/digits then lowercase/digit/'.'/'_'/'-', "
            "max 64 chars; no leading dot or path separators)",
        )
    if name in RESERVED_NAMES:
        raise StoreError(
            "invalid_name", f"schema name {name!r} is reserved"
        )
    return name


def _json_size(schema: dict[str, Any]) -> int:
    """Byte size of the canonical form (built-ins have no file on disk)."""
    return len(
        json.dumps(
            schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )


def _title(schema: dict[str, Any] | None) -> str:
    """The schema's `title`, or "" (invalid user files have no title)."""
    if schema is None:
        return ""
    title = schema.get("title")
    return title if isinstance(title, str) else ""


class SchemaStore:
    """User schemas shadowing built-ins; the controller owns this store."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)  # created on demand (JEV-01.11)

    @staticmethod
    def check_name(name: str) -> str:
        """Validate a store name, raising `StoreError("invalid_name", …)`."""
        return _check_name(name)

    @property
    def active_path(self) -> Path:
        """The controller's `.active.json` Jev state file (doc 02)."""
        return self.root / ACTIVE_FILE

    def _user_files(self) -> list[Path]:
        """Valid user schema files, sorted by name; skips junk and state."""
        return sorted(
            path
            for path in self.root.glob("*.json")
            if path.name != ACTIVE_FILE
            and not path.name.startswith(".")
            and NAME_RE.match(path.stem)
            and path.stem not in RESERVED_NAMES
        )

    def _user_path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    # -- queries -----------------------------------------------------------

    def list(self) -> list[SchemaEntry]:
        """Every visible schema, built-ins first, overridden ones merged."""
        builtin_names = list_builtin()
        entries: dict[str, SchemaEntry] = {}
        for name in builtin_names:
            schema = load_builtin(name)
            entries[name] = SchemaEntry(
                name,
                "builtin",
                canonical_hash(schema),
                _json_size(schema),
                0.0,
                _title(schema),
            )
        for path in self._user_files():
            name = path.stem
            stat = path.stat()
            text = path.read_text(encoding="utf-8")
            result = validate_schema_text(text)
            source = "override" if name in entries else "user"
            entries[name] = SchemaEntry(
                name,
                source,
                result.sha256 or "",
                stat.st_size,
                stat.st_mtime,
                _title(result.schema),
            )
        return [entries[name] for name in sorted(entries)]

    def get(self, name: str) -> StoredSchema:
        """Resolve one schema (user file wins over built-in).

        Raises `StoreError("not_found", …)` for unknown names and
        `StoreError("corrupt", …)` when the user file is unreadable.
        """
        _check_name(name)
        builtin = name in list_builtin()
        path = self._user_path(name)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            result = validate_schema_text(text)
            if result.schema is None:
                raise StoreError(
                    "corrupt",
                    f"user schema {name!r} is unreadable: "
                    + "; ".join(result.errors),
                )
            stat = path.stat()
            return StoredSchema(
                name,
                "override" if builtin else "user",
                result.schema,
                result.sha256,
                stat.st_size,
                stat.st_mtime,
            )
        if builtin:
            schema = load_builtin(name)
            return StoredSchema(
                name, "builtin", schema, canonical_hash(schema), _json_size(schema), 0.0
            )
        raise StoreError("not_found", f"no Jev schema named {name!r}")

    # -- mutations ---------------------------------------------------------

    def put(self, name: str, text: str) -> ValidationResult:
        """Validate then atomically write `<name>.json` (JEV-01.10).

        Validation failures are returned as data (not raised) so the admin
        UI can show them inline. Store-level refusals (bad name, too many
        schemas) raise `StoreError`.
        """
        _check_name(name)
        result = validate_schema_text(text)
        if not result.ok:
            return result
        path = self._user_path(name)
        if not path.exists() and len(self._user_files()) >= MAX_USER_SCHEMAS:
            raise StoreError(
                "limit",
                f"at most {MAX_USER_SCHEMAS} user schemas are allowed",
            )
        self._atomic_write(path, text)
        return result

    def delete(self, name: str) -> str:
        """Remove a user schema; returns its source ("user" | "override").

        Deleting an override reverts to the built-in. Built-in-only names
        raise `StoreError("readonly", …)`; unknown names raise
        `StoreError("not_found", …)`.
        """
        _check_name(name)
        path = self._user_path(name)
        if not path.is_file():
            if name in list_builtin():
                raise StoreError("readonly", "built-in schemas are read-only")
            raise StoreError("not_found", f"no user schema named {name!r}")
        source = "override" if name in list_builtin() else "user"
        path.unlink()
        return source

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Temp file in the same dir + `os.replace` (atomic, JEV-01.10)."""
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=".jev-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
