"""Unit tests for `vibe.jev.store.SchemaStore` (doc 06 §A, A9–A10)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vibe.jev import (
    MAX_USER_SCHEMAS,
    SchemaStore,
    StoreError,
    canonical_hash,
    load_builtin,
)

GOOD_TEXT = json.dumps(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Alpha",
        "type": "object",
        "properties": {"result": {"type": "string", "enum": ["YES", "NO"]}},
        "required": ["result"],
        "additionalProperties": False,
    }
)
CHANGED_TEXT = GOOD_TEXT.replace('"Alpha"', '"AlphaTwo"')


def test_put_get_list_delete_round_trip(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    result = store.put("alpha", GOOD_TEXT)
    assert result.ok, result.errors

    stored = store.get("alpha")
    assert stored.source == "user"
    assert stored.sha256 == canonical_hash(json.loads(GOOD_TEXT))
    assert stored.size > 0

    entries = store.list()
    names = {entry.name for entry in entries}
    assert "alpha" in names
    # built-ins are always visible
    assert {"default", "yes-no", "yes-no-unsure"} <= names
    alpha = next(entry for entry in entries if entry.name == "alpha")
    assert alpha.source == "user"
    assert alpha.sha256 == stored.sha256

    assert store.delete("alpha") == "user"
    with pytest.raises(StoreError) as excinfo:
        store.get("alpha")
    assert excinfo.value.code == "not_found"


def test_put_invalid_text_returns_failed_result_and_writes_nothing(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    result = store.put("bad", "{not json")
    assert not result.ok
    assert not (tmp_path / "bad.json").exists()


def test_put_atomic_crash_leaves_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SchemaStore(tmp_path)
    store.put("alpha", GOOD_TEXT)

    real_replace = os.replace

    def boom(source: str, destination: str) -> None:
        raise OSError("simulated crash between temp write and replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        store.put("alpha", CHANGED_TEXT)
    monkeypatch.setattr(os, "replace", real_replace)

    assert (tmp_path / "alpha.json").read_text(encoding="utf-8") == GOOD_TEXT
    assert list(tmp_path.glob(".jev-*.tmp")) == []


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "a/b",
        "a\\b",
        "A",
        ".active",
        ".active.json",
        "has space",
        "a" * 65,
        "new",
        "launch",
        "detach",
        "build",
    ],
)
def test_invalid_names_rejected(tmp_path: Path, name: str) -> None:
    store = SchemaStore(tmp_path)
    with pytest.raises(StoreError) as excinfo:
        store.put(name, GOOD_TEXT)
    assert excinfo.value.code == "invalid_name"
    with pytest.raises(StoreError) as excinfo:
        store.get(name)
    assert excinfo.value.code == "invalid_name"
    assert list(tmp_path.glob("*.json")) == []


def test_max_user_schemas_enforced(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    for index in range(MAX_USER_SCHEMAS):
        store.put(f"s{index:03d}", GOOD_TEXT)
    with pytest.raises(StoreError) as excinfo:
        store.put("one-too-many", GOOD_TEXT)
    assert excinfo.value.code == "limit"
    # replacing an existing schema still works at the cap
    assert store.put("s000", CHANGED_TEXT).ok


def test_shadowing_and_revert(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    builtin = load_builtin("default")
    assert store.get("default").source == "builtin"
    assert store.get("default").schema == builtin

    store.put("default", CHANGED_TEXT)
    entry = next(entry for entry in store.list() if entry.name == "default")
    assert entry.source == "override"
    assert store.get("default").source == "override"
    assert store.get("default").schema != builtin

    assert store.delete("default") == "override"
    assert store.get("default").source == "builtin"
    assert store.get("default").schema == builtin


def test_delete_builtin_only_is_readonly(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    with pytest.raises(StoreError) as excinfo:
        store.delete("yes-no")
    assert excinfo.value.code == "readonly"


def test_delete_unknown_is_not_found(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    with pytest.raises(StoreError) as excinfo:
        store.delete("never-existed")
    assert excinfo.value.code == "not_found"


def test_corrupt_user_file_raises(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    (tmp_path / "broken.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(StoreError) as excinfo:
        store.get("broken")
    assert excinfo.value.code == "corrupt"


def test_store_creates_root_on_demand(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "jev"
    store = SchemaStore(root)
    assert root.is_dir()
    assert store.list()  # built-ins visible even in a fresh dir


def test_active_path_inside_store(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    assert store.active_path == tmp_path / ".active.json"
    # the state file is never listed or addressable as a schema
    store.active_path.write_text("{}", encoding="utf-8")
    assert all(entry.name != ".active.json" for entry in store.list())
    with pytest.raises(StoreError):
        store.get(".active.json")
