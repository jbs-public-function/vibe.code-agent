"""Controller-side Jev logic: state, resolution, and the activation probe.

JEV-02.11/JEV-02.14/JEV-02.15. The docs name this module
`vibe/controller/jev.py`; it lives here (a re-export shim keeps that path
valid) because `vibe.jobs` needs these symbols and importing anything under
`vibe.controller` would pull in the servicer via the package `__init__`
(circular import).

`JevState` persists the active attachment to `${VIBE_JEV_DIR}/.active.json`
(atomic write) so it survives controller restarts; `resolve_jev` implements
the J4 precedence chain; `probe_jev` runs a one-shot chat completion with the
schema so backend-unsupported schemas fail at launch, not on the first real
request (ADR-J6).

This module never imports `docker` (XR-5): container discovery goes through
the runner's `DockerCtl`, and tests pass duck-typed fakes.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from vibe.jev import (
    JevError,
    SchemaStore,
    StoreError,
    canonical_hash,
    instance_validator_for,
    schema_name_for_openai,
    validate_schema_text,
    wire_schema,
)
from vibe.registry import ModelSpec

# Probe parameters (JEV-02.15): separate from VIBE_LOAD_TIMEOUT.
PROBE_TIMEOUT_S = 20.0
PROBE_MAX_TOKENS = 128
PROBE_PROMPT = "Return a valid example."


@dataclass(frozen=True)
class JevLoadSpec:
    """Proto-free launch parameters for a Jev load (JEV-02.2)."""

    schema_name: str = ""
    schema_json: str = ""
    save_as: str = ""


@dataclass(frozen=True)
class ResolvedJev:
    """The outcome of `resolve_jev`: a validated schema + provenance."""

    name: str
    source: str  # "include-json" | "store" | "model-default" | "default"
    schema: dict[str, Any]
    sha256: str
    save_as: str = ""


@dataclass(frozen=True)
class ActiveJev:
    """The currently attached schema (persisted in `.active.json`)."""

    alias: str
    name: str
    source: str
    schema: dict[str, Any]
    sha256: str
    applied_at_unix_ms: int


class JevState:
    """Persistent holder of the active Jev attachment (JEV-02.11)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._active: ActiveJev | None = None

    def load(self) -> list[str]:
        """Read the state file; corrupt/missing files yield `[]`/warnings."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._active = None
            return []
        except OSError as exc:
            self._active = None
            return [f"cannot read Jev state file {self._path}: {exc}"]
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            schema = data["schema"]
            if not isinstance(schema, dict):
                raise ValueError("'schema' is not an object")
            self._active = ActiveJev(
                alias=str(data["alias"]),
                name=str(data["name"]),
                source=str(data.get("source") or "default"),
                schema=schema,
                sha256=str(data.get("sha256") or canonical_hash(schema)),
                applied_at_unix_ms=int(data.get("applied_at_unix_ms") or 0),
            )
            return []
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._active = None
            return [f"ignoring corrupt Jev state file {self._path}: {exc}"]

    def current(self) -> ActiveJev | None:
        return self._active

    def set(self, alias: str, name: str, source: str, schema: dict[str, Any]) -> None:
        """Attach a schema; persists atomically (temp file + `os.replace`)."""
        self._active = ActiveJev(
            alias=alias,
            name=name,
            source=source,
            schema=schema,
            sha256=canonical_hash(schema),
            applied_at_unix_ms=int(datetime.now(UTC).timestamp() * 1000),
        )
        self._write()

    def clear(self) -> None:
        """Detach; the missing state file is the canonical "inactive"."""
        self._active = None
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def reconcile(self, running_aliases: set[str]) -> bool:
        """Keep state only while a container for its alias is running."""
        active = self._active
        if active is None:
            return False
        if active.alias in running_aliases:
            return False
        self.clear()
        return True

    def _write(self) -> None:
        active = self._active
        if active is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alias": active.alias,
            "name": active.name,
            "source": active.source,
            "sha256": active.sha256,
            "applied_at_unix_ms": active.applied_at_unix_ms,
            "schema": active.schema,
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".active-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.chmod(tmp, 0o644)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def resolve_jev(
    spec: JevLoadSpec, model: ModelSpec, store: SchemaStore
) -> ResolvedJev:
    """Pick the schema per the J4 precedence chain (JEV-02.14).

    `schema_json` (inline) > `schema_name` > `ModelSpec.jev_schema` >
    `default` (user override, else built-in). Raises `JevError` with a
    taxonomy code on any failure; never downgrades silently.
    """
    if spec.schema_json:
        if spec.save_as:
            try:
                store.check_name(spec.save_as)  # fail before the probe
            except StoreError as exc:
                raise JevError("jev_schema_invalid", f"save_as: {exc}") from exc
        result = validate_schema_text(spec.schema_json)
        if not result.ok or result.schema is None:
            raise JevError("jev_schema_invalid", "; ".join(result.errors))
        name = spec.schema_name or schema_name_for_openai(result.schema)
        return ResolvedJev(
            name=name,
            source="include-json",
            schema=result.schema,
            sha256=result.sha256,
            save_as=spec.save_as,
        )
    if spec.schema_name:
        return _from_store(spec.schema_name, "store", store)
    if model.jev_schema:
        return _from_store(model.jev_schema, "model-default", store)
    return _from_store("default", "default", store)


def _from_store(name: str, source: str, store: SchemaStore) -> ResolvedJev:
    try:
        stored = store.get(name)
    except StoreError as exc:
        if exc.code == "not_found":
            raise JevError(
                "jev_schema_not_found",
                f"no Jev schema named {name!r} — run `vibe jev schema put {name} -` "
                "or pick another name",
            ) from exc
        raise JevError("jev_schema_invalid", f"schema {name!r}: {exc}") from exc
    return ResolvedJev(
        name=stored.name,
        source=source,
        schema=stored.schema,
        sha256=stored.sha256,
    )


async def probe_jev(
    host_port: int, alias: str, schema: dict[str, Any]
) -> None:
    """One-shot chat completion that must yield schema-valid JSON (JEV-02.15).

    Raises `JevError("jev_probe_failed", …)` on any failure; the model stays
    loaded in plain mode and the caller does not attach the schema.
    """
    payload: dict[str, Any] = {
        "model": alias,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "stream": False,
        "max_tokens": PROBE_MAX_TOKENS,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name_for_openai(schema),
                "schema": wire_schema(schema),
                "strict": True,
            },
        },
    }
    url = f"http://127.0.0.1:{host_port}/v1/chat/completions"
    timeout = httpx.Timeout(connect=5.0, read=PROBE_TIMEOUT_S, write=30.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise JevError("jev_probe_failed", f"probe request failed: {exc}") from exc
    if response.status_code != 200:
        body = response.text[:500]
        raise JevError(
            "jev_probe_failed",
            f"vLLM rejected the schema (HTTP {response.status_code}): {body}",
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise JevError("jev_probe_failed", "probe response was not JSON") from exc
    choices = data.get("choices") or []
    if not choices:
        raise JevError("jev_probe_failed", "probe response had no choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise JevError(
            "jev_probe_failed",
            "probe output hit max_tokens — the schema may permit unbounded "
            "output; add maxLength/bounds (JEV-02.16)",
        )
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise JevError("jev_probe_failed", "probe response had no message content")
    try:
        instance = json.loads(content)
    except ValueError as exc:
        raise JevError(
            "jev_probe_failed", f"probe output was not valid JSON: {exc}"
        ) from exc
    validator = instance_validator_for(schema)
    if not validator(schema).is_valid(instance):
        errors = sorted(
            validator(schema).iter_errors(instance), key=lambda err: list(err.path)
        )[:3]
        raise JevError(
            "jev_probe_failed",
            "probe output did not validate against the schema: "
            + "; ".join(error.message for error in errors),
        )
