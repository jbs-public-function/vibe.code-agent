"""Registry validation and lookup tests (FR-00.2), plus the Phase 4
registration/reload surface (FR-04.5/FR-04.6, NFR-04.3).

The propose/write tests synthesize tmp HF cache layouts (`hub/models--org--
repo/snapshots/<40-hex>/`) with sparse safetensors files, so the heuristics
see real sizes without writing gigabytes.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from vibe.client import ControllerClient
from vibe.config import Settings
from vibe.controller import RingBuffer, build_server
from vibe.controller.cache import CacheScanner
from vibe.gen.vibe.v1 import control_pb2
from vibe.gpu import GpuProcess, GpuStats
from vibe.jobs import JobRunner
from vibe.registry import ModelSpec, Registry, RegistryError

REV = "a" * 40
GIB = 1024**3


def _entry(alias: str, **overrides) -> dict:
    """A single valid-ish entry; `overrides` tweak one field at a time."""
    entry = {
        "alias": alias,
        "display_name": f"Display {alias}",
        "hf_repo": f"org/{alias}",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "container_name": f"vibe-{alias}",
        "host_port": 8000,
    }
    entry.update(overrides)
    return entry


def _registry(models: list[dict]) -> Registry:
    return Registry.from_dict({"models": models})


# -- valid input ---------------------------------------------------------


def test_valid_registry_shape(valid_registry_dict):
    registry = Registry.from_dict(valid_registry_dict)
    assert len(registry) == 3
    assert registry.aliases == [
        "qwen2.5-coder-7b",
        "llama-3.1-8b",
        "deepseek-r1-14b",
    ]
    assert [spec.alias for spec in registry] == registry.aliases

    spec = registry.get("qwen2.5-coder-7b")
    assert isinstance(spec, ModelSpec)
    assert spec.host_port == 8030
    assert spec.max_model_len == 32768
    assert spec.quantization == "awq"
    assert spec.extra_args == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    assert spec.tags == ["code", "tools"]


def test_by_port_lookup(valid_registry_dict):
    registry = Registry.from_dict(valid_registry_dict)
    qwen = registry.by_port(8030)
    llama = registry.by_port(8020)
    deepseek = registry.by_port(8010)
    assert qwen is not None
    assert llama is not None
    assert deepseek is not None
    assert qwen.alias == "qwen2.5-coder-7b"
    assert llama.alias == "llama-3.1-8b"
    assert deepseek.alias == "deepseek-r1-14b"
    assert registry.by_port(9999) is None


def test_get_unknown_alias_raises_keyerror(valid_registry_dict):
    registry = Registry.from_dict(valid_registry_dict)
    with pytest.raises(KeyError):
        registry.get("no-such-alias")


# -- duplicate rejection ---------------------------------------------------


def test_duplicate_alias_rejected():
    models = [
        _entry("dup", host_port=8001, container_name="vibe-dup-1"),
        _entry("dup", host_port=8002, container_name="vibe-dup-2"),
    ]
    with pytest.raises(RegistryError) as excinfo:
        _registry(models)
    assert any("duplicate alias: dup" in error for error in excinfo.value.errors)


def test_duplicate_host_port_rejected():
    models = [
        _entry("model-a", host_port=9000, container_name="vibe-a"),
        _entry("model-b", host_port=9000, container_name="vibe-b"),
    ]
    with pytest.raises(RegistryError) as excinfo:
        _registry(models)
    assert any("duplicate host_port: 9000" in error for error in excinfo.value.errors)


def test_duplicate_container_name_rejected():
    models = [
        _entry("model-a", host_port=9001, container_name="vibe-shared"),
        _entry("model-b", host_port=9002, container_name="vibe-shared"),
    ]
    with pytest.raises(RegistryError) as excinfo:
        _registry(models)
    assert any(
        "duplicate container_name: vibe-shared" in error
        for error in excinfo.value.errors
    )


def test_all_problems_collected_in_one_error():
    models = [
        _entry("dup", host_port=8001, container_name="vibe-dup-1"),
        _entry("dup", host_port=8002, container_name="vibe-dup-2"),
        _entry("Bad_Alias", host_port=8003, container_name="vibe-bad"),
    ]
    with pytest.raises(RegistryError) as excinfo:
        _registry(models)
    errors = excinfo.value.errors
    assert any("duplicate alias" in error for error in errors)
    assert any("Bad_Alias" in error for error in errors)
    assert len(errors) >= 2


# -- alias regex ----------------------------------------------------------


@pytest.mark.parametrize("alias", ["Bad_Alias", "-leading", "", "two words"])
def test_alias_regex_rejects(alias):
    with pytest.raises(RegistryError):
        _registry([_entry(alias)])


@pytest.mark.parametrize("alias", ["a-b.c", "qwen2.5-coder-7b"])
def test_alias_regex_accepts(alias):
    registry = _registry([_entry(alias)])
    assert registry.get(alias).alias == alias


# -- gpu_memory_utilization ------------------------------------------------


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_gpu_memory_utilization_rejects(value):
    with pytest.raises(RegistryError):
        _registry([_entry("ok", gpu_memory_utilization=value)])


@pytest.mark.parametrize("value", [0.01, 1.0])
def test_gpu_memory_utilization_accepts_boundaries(value):
    registry = _registry([_entry("ok", gpu_memory_utilization=value)])
    assert registry.get("ok").gpu_memory_utilization == value


# -- max_model_len ----------------------------------------------------------


@pytest.mark.parametrize("value", [0, -5])
def test_max_model_len_rejects(value):
    with pytest.raises(RegistryError):
        _registry([_entry("ok", max_model_len=value)])


def test_max_model_len_accepts_one():
    registry = _registry([_entry("ok", max_model_len=1)])
    assert registry.get("ok").max_model_len == 1


# -- host_port ---------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 65536])
def test_host_port_rejects(value):
    with pytest.raises(RegistryError):
        _registry([_entry("ok", host_port=value)])


@pytest.mark.parametrize("value", [1, 65535])
def test_host_port_accepts_boundaries(value):
    registry = _registry([_entry("ok", host_port=value)])
    assert registry.get("ok").host_port == value


# -- extra fields -------------------------------------------------------------


def test_unknown_field_rejected():
    with pytest.raises(RegistryError) as excinfo:
        _registry([_entry("ok", bogus_field=123)])
    assert any("bogus_field" in error for error in excinfo.value.errors)


# -- Phase 4: multi-document load / to_dict / reload --------------------------


def test_load_merges_multiple_documents(tmp_path):
    """`load` merges every YAML document's models in order (NFR-04.3)."""
    path = tmp_path / "models.yaml"
    first = {"models": [_entry("first", host_port=8001, container_name="vibe-first")]}
    second = {
        "models": [_entry("second", host_port=8002, container_name="vibe-second")]
    }
    path.write_text(
        yaml.safe_dump(first, sort_keys=False)
        + "\n---\n# --- added by vibe register 2026-09-15 ---\n"
        + yaml.safe_dump(second, sort_keys=False)
    )
    registry = Registry.load(path)
    assert registry.aliases == ["first", "second"]
    assert registry.path == path


def test_load_rejects_bad_yaml_in_any_document(tmp_path):
    path = tmp_path / "models.yaml"
    good = yaml.safe_dump(
        {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
        sort_keys=False,
    )
    path.write_text(good + "\n---\nmodels: [unclosed\n")
    with pytest.raises(RegistryError) as excinfo:
        Registry.load(path)
    assert any("not valid YAML" in error for error in excinfo.value.errors)


def test_load_rejects_duplicate_alias_across_documents(tmp_path):
    """Uniqueness is enforced on the merged stream, not per document."""
    path = tmp_path / "models.yaml"
    first = {"models": [_entry("dup", host_port=8001, container_name="vibe-dup-1")]}
    second = {"models": [_entry("dup", host_port=8002, container_name="vibe-dup-2")]}
    path.write_text(
        yaml.safe_dump(first, sort_keys=False)
        + "\n---\n"
        + yaml.safe_dump(second, sort_keys=False)
    )
    with pytest.raises(RegistryError) as excinfo:
        Registry.load(path)
    assert any("duplicate alias: dup" in error for error in excinfo.value.errors)


def test_load_rejects_duplicate_port_across_documents(tmp_path):
    path = tmp_path / "models.yaml"
    first = {"models": [_entry("a", host_port=8001, container_name="vibe-a")]}
    second = {"models": [_entry("b", host_port=8001, container_name="vibe-b")]}
    path.write_text(
        yaml.safe_dump(first, sort_keys=False)
        + "\n---\n"
        + yaml.safe_dump(second, sort_keys=False)
    )
    with pytest.raises(RegistryError) as excinfo:
        Registry.load(path)
    assert any("duplicate host_port: 8001" in error for error in excinfo.value.errors)


def test_load_rejects_non_mapping_document(tmp_path):
    path = tmp_path / "models.yaml"
    good = yaml.safe_dump(
        {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
        sort_keys=False,
    )
    path.write_text(good + "\n---\n- just\n- a\n- list\n")
    with pytest.raises(RegistryError) as excinfo:
        Registry.load(path)
    assert any("must be a mapping" in error for error in excinfo.value.errors)


def test_to_dict_round_trips(valid_registry_dict):
    registry = Registry.from_dict(valid_registry_dict)
    again = Registry.from_dict(registry.to_dict())
    assert again.aliases == registry.aliases
    qwen = again.get("qwen2.5-coder-7b")
    assert qwen.host_port == 8030
    assert qwen.extra_args == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    assert qwen.tags == ["code", "tools"]


def test_reload_swaps_registry_in_place(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    registry = Registry.load(path)
    assert registry.aliases == ["first"]

    path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    _entry("first", host_port=8001, container_name="vibe-first"),
                    _entry("second", host_port=8002, container_name="vibe-second"),
                ]
            },
            sort_keys=False,
        )
    )
    assert registry.reload() == []
    assert registry.aliases == ["first", "second"]
    assert registry.path == path  # path is kept across swaps
    assert registry.get("second").host_port == 8002


def test_reload_invalid_keeps_old_snapshot(tmp_path):
    """Validate-before-swap: bad YAML leaves the live registry intact (FR-04.6)."""
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    registry = Registry.load(path)
    path.write_text("models: [unclosed\n")
    errors = registry.reload()
    assert any("not valid YAML" in error for error in errors)
    assert registry.aliases == ["first"]


def test_reload_missing_file_returns_errors(tmp_path):
    registry = Registry.from_dict(
        {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
        path=tmp_path / "nope.yaml",
    )
    errors = registry.reload()
    assert any("not found" in error for error in errors)
    assert registry.aliases == ["first"]


# -- Phase 4: propose heuristics ----------------------------------------------


def _make_hub_repo(
    hf_home: Path, repo_id: str, files: dict[str, bytes], rev: str = REV
) -> Path:
    """A minimal HF cache layout for one repo; returns the repo dir."""
    hub = hf_home / "hub"
    repo_dir = hub / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    refs = repo_dir / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(rev)
    for name, content in files.items():
        (snap / name).write_bytes(content)
    return repo_dir


def _sparse(path: Path, size: int) -> None:
    """A sparse file of `size` bytes: st_size reports size, no disk I/O."""
    with path.open("wb") as fh:
        fh.truncate(size)


def _weights(repo_dir: Path, gib: int) -> None:
    _sparse(repo_dir / "snapshots" / REV / "model.safetensors", gib * GIB)


def test_propose_fills_defaults_and_small_heuristic(tmp_path, registry):
    """Defaults per FR-04.5: display, container, port, 0.60/32768 bucket."""
    repo_dir = _make_hub_repo(tmp_path, "org/small", {"config.json": b"{}"})
    _weights(repo_dir, 2)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small"}, CacheScanner(tmp_path)
    )
    assert errors == []
    assert entry["gpu_memory_utilization"] == 0.60
    assert entry["max_model_len"] == 8192  # no config mml → default 8192 < cap
    assert entry["container_name"] == "vibe-small"
    assert entry["host_port"] == 8011  # first free after 8010/8020/8030
    assert entry["quantization"] == "none"
    assert entry["display_name"] == "small"
    assert entry["extra_args"] == []
    assert entry["tags"] == []


@pytest.mark.parametrize(
    ("gib", "gpu", "cap"),
    [(2, 0.60, 32768), (4, 0.72, 32768), (8, 0.85, 8192)],
)
def test_propose_heuristic_buckets(tmp_path, registry, gib, gpu, cap):
    repo_dir = _make_hub_repo(tmp_path, "org/bucket", {"config.json": b"{}"})
    _weights(repo_dir, gib)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/bucket", "alias": f"bucket-{gib}"}, CacheScanner(tmp_path)
    )
    assert errors == []
    assert entry["gpu_memory_utilization"] == gpu
    assert entry["max_model_len"] == min(8192, cap)


def test_propose_refuses_over_10_gib(tmp_path, registry):
    repo_dir = _make_hub_repo(tmp_path, "org/huge", {"config.json": b"{}"})
    _weights(repo_dir, 12)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/huge", "alias": "huge"}, CacheScanner(tmp_path)
    )
    assert any("16 GB card" in error for error in errors)


def test_propose_requires_downloaded_repo(tmp_path, registry):
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/missing", "alias": "missing"}, CacheScanner(tmp_path)
    )
    assert any("download the repo first" in error for error in errors)


def test_propose_max_model_len_from_config(tmp_path, registry):
    config = json.dumps({"max_position_embeddings": 16384}).encode()
    repo_dir = _make_hub_repo(tmp_path, "org/m", {"config.json": config})
    _weights(repo_dir, 2)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/m", "alias": "m"}, CacheScanner(tmp_path)
    )
    assert errors == []
    assert entry["max_model_len"] == 16384  # 16384 ≤ 32768 cap


def test_propose_max_model_len_capped_by_bucket(tmp_path, registry):
    config = json.dumps({"max_position_embeddings": 32768}).encode()
    repo_dir = _make_hub_repo(tmp_path, "org/m", {"config.json": config})
    _weights(repo_dir, 8)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/m", "alias": "m"}, CacheScanner(tmp_path)
    )
    assert errors == []
    assert entry["max_model_len"] == 8192  # capped by the 6–10 GiB bucket


def test_propose_caller_max_model_len_wins(tmp_path, registry):
    repo_dir = _make_hub_repo(tmp_path, "org/m", {"config.json": b"{}"})
    _weights(repo_dir, 2)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/m", "alias": "m", "max_model_len": 4096},
        CacheScanner(tmp_path),
    )
    assert errors == []
    assert entry["max_model_len"] == 4096


def test_propose_rejects_gguf_only_repo(tmp_path, registry):
    _make_hub_repo(tmp_path, "org/gguf", {"model-Q4_K_M.gguf": b"gguf"})
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/gguf", "alias": "gguf"}, CacheScanner(tmp_path)
    )
    assert any("laboratory/inference_gguf.py" in error for error in errors)


def test_propose_rejects_realistic_gguf_repo(tmp_path, registry):
    """IMP-02.21: real GGUF repos ship README + config.json alongside the
    .gguf weights, so 'every file is .gguf' never matches them."""
    _make_hub_repo(
        tmp_path,
        "org/gguf-real",
        {
            "README.md": b"# gguf repo",
            "config.json": b"{}",
            "model-Q4_K_M.gguf": b"gguf",
        },
    )
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/gguf-real", "alias": "gguf-real"}, CacheScanner(tmp_path)
    )
    assert any("laboratory/inference_gguf.py" in error for error in errors)
    # The GGUF hint replaces the generic weightless-refusal (IMP-02.22).
    assert all("no *.safetensors" not in error for error in errors)


def test_propose_zero_weights_rejected(tmp_path, registry):
    """IMP-02.22: cached but weightless repos are refused, never bucketed
    into the ≤3 GiB heuristic."""
    _make_hub_repo(tmp_path, "org/empty", {"config.json": b"{}"})
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/empty", "alias": "empty"}, CacheScanner(tmp_path)
    )
    assert any("no *.safetensors" in error for error in errors)


def test_propose_rejects_unexpanded_tilde(tmp_path, registry):
    """IMP-02.11: vLLM cannot see an unexpanded `~` — demand the full path."""
    entry, warnings, errors = registry.propose(
        {"hf_repo": "~/models/x", "alias": "x"}, CacheScanner(tmp_path)
    )
    assert any("unexpanded" in error for error in errors)


def test_propose_allows_mixed_gguf_and_safetensors(tmp_path, registry):
    repo_dir = _make_hub_repo(
        tmp_path, "org/mixed", {"extra.gguf": b"gguf"}
    )
    _weights(repo_dir, 2)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/mixed", "alias": "mixed"}, CacheScanner(tmp_path)
    )
    assert errors == []


def test_propose_port_exhaustion(tmp_path):
    specs = [
        ModelSpec(
            alias=f"m{i}",
            display_name="M",
            hf_repo=f"org/m{i}",
            max_model_len=4096,
            gpu_memory_utilization=0.5,
            container_name=f"vibe-m{i}",
            host_port=port,
        )
        for i, port in enumerate(range(8010, 8100))
    ]
    full = Registry(specs)
    repo_dir = _make_hub_repo(tmp_path, "org/small", {"config.json": b"{}"})
    _weights(repo_dir, 2)
    entry, warnings, errors = full.propose(
        {"hf_repo": "org/small", "alias": "small"}, CacheScanner(tmp_path)
    )
    assert any("no free port" in error for error in errors)


def test_propose_rejects_duplicates(tmp_path, registry):
    repo_dir = _make_hub_repo(tmp_path, "org/small", {"config.json": b"{}"})
    _weights(repo_dir, 2)
    scanner = CacheScanner(tmp_path)

    _, _, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "qwen2.5-coder-7b"}, scanner
    )
    assert any("duplicate alias: qwen2.5-coder-7b" in e for e in errors)

    _, _, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small", "host_port": 8030}, scanner
    )
    assert any("duplicate host_port: 8030" in e for e in errors)

    _, _, errors = registry.propose(
        {
            "hf_repo": "org/small",
            "alias": "small",
            "container_name": "vibe-qwen2.5-coder-7b",
        },
        scanner,
    )
    assert any(
        "duplicate container_name: vibe-qwen2.5-coder-7b" in e for e in errors
    )


def test_propose_local_path(tmp_path, registry):
    """Absolute local paths skip the cache scan (Phase 5 round-trip)."""
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    _sparse(model_dir / "model.safetensors", 2 * GIB)
    entry, warnings, errors = registry.propose(
        {"hf_repo": str(model_dir), "alias": "local"}, CacheScanner(tmp_path)
    )
    assert errors == []
    assert entry["gpu_memory_utilization"] == 0.60
    assert entry["host_port"] == 8011


def test_propose_local_path_must_exist(tmp_path, registry):
    entry, warnings, errors = registry.propose(
        {"hf_repo": str(tmp_path / "nope"), "alias": "local"},
        CacheScanner(tmp_path),
    )
    assert any("does not exist" in error for error in errors)


def test_propose_quantization_config_warnings(tmp_path, registry):
    repo_dir = _make_hub_repo(
        tmp_path,
        "org/quant",
        {"config.json": json.dumps({"quantization_config": {"bits": 4}}).encode()},
    )
    _weights(repo_dir, 2)
    scanner = CacheScanner(tmp_path)

    # Default none + config declares it → warn.
    _, warnings, errors = registry.propose(
        {"hf_repo": "org/quant", "alias": "quant"}, scanner
    )
    assert errors == []
    assert any("declares quantization_config" in w for w in warnings)

    # Given awq + config declares it → no warning.
    _, warnings, errors = registry.propose(
        {"hf_repo": "org/quant", "alias": "quant", "quantization": "awq"}, scanner
    )
    assert all("quantization_config" not in w for w in warnings)

    # Bare config + requested awq → warn.
    bare = _make_hub_repo(tmp_path, "org/bare", {"config.json": b"{}"})
    _weights(bare, 2)
    _, warnings, errors = registry.propose(
        {"hf_repo": "org/bare", "alias": "bare", "quantization": "awq"}, scanner
    )
    assert any("no quantization_config" in w for w in warnings)


def test_propose_tool_parser_extra_args(tmp_path, registry):
    repo_dir = _make_hub_repo(tmp_path, "org/small", {"config.json": b"{}"})
    _weights(repo_dir, 2)
    scanner = CacheScanner(tmp_path)

    entry, _, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small", "tool_parser": "hermes"}, scanner
    )
    assert errors == []
    assert entry["extra_args"] == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]

    entry, _, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small", "tool_parser": "llama3_json"},
        scanner,
    )
    assert entry["extra_args"] == [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "llama3_json",
    ]

    _, _, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small", "tool_parser": "bogus"}, scanner
    )
    assert any("tool_parser must be one of" in e for e in errors)


def test_propose_gpu_override_and_warning(tmp_path, registry):
    repo_dir = _make_hub_repo(tmp_path, "org/small", {"config.json": b"{}"})
    _weights(repo_dir, 2)  # heuristic: 0.60
    scanner = CacheScanner(tmp_path)

    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small", "gpu_memory_utilization": 0.50},
        scanner,
    )
    assert errors == []
    assert entry["gpu_memory_utilization"] == 0.50
    assert warnings == []

    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small", "gpu_memory_utilization": 0.90},
        scanner,
    )
    assert entry["gpu_memory_utilization"] == 0.90
    assert any("exceeds the recommended 0.6" in w for w in warnings)

    _, _, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small", "gpu_memory_utilization": 1.5},
        scanner,
    )
    assert any("must be in (0, 1]" in e for e in errors)


def test_propose_vram_warning_near_limit(tmp_path, registry):
    repo_dir = _make_hub_repo(tmp_path, "org/big", {"config.json": b"{}"})
    _weights(repo_dir, 9)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/big", "alias": "big"}, CacheScanner(tmp_path)
    )
    assert errors == []
    assert any("16 GB" in w for w in warnings)


def test_propose_collects_all_errors(tmp_path, registry):
    """Every violation is listed in one pass (NFR-04.4)."""
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/missing", "alias": "Bad_Alias", "tool_parser": "bogus"},
        CacheScanner(tmp_path),
    )
    assert any("Bad_Alias" in e for e in errors)
    assert any("download the repo first" in e for e in errors)
    assert any("tool_parser" in e for e in errors)


# -- Phase 4: write_entry ------------------------------------------------------


def test_write_entry_appends_marked_document(tmp_path):
    """Atomic append with the `vibe register` marker (NFR-04.3)."""
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    registry = Registry.load(path)
    assert registry.write_entry(
        _entry("second", host_port=8002, container_name="vibe-second")
    ) == []

    text = path.read_text()
    assert "# --- added by vibe register" in text
    assert "\n---\n" in text
    on_disk = Registry.load(path)
    assert on_disk.aliases == ["first", "second"]
    assert on_disk.get("second").host_port == 8002


def test_write_entry_rejects_duplicates_and_leaves_file_untouched(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    before = path.read_text()
    registry = Registry.load(path)

    errors = registry.write_entry(
        _entry("first", host_port=8090, container_name="vibe-other")
    )
    assert any("duplicate alias: first" in e for e in errors)
    assert path.read_text() == before

    errors = registry.write_entry(
        _entry("second", host_port=8001, container_name="vibe-second")
    )
    assert any("duplicate host_port: 8001" in e for e in errors)
    assert path.read_text() == before

    errors = registry.write_entry(
        _entry("second", host_port=8002, container_name="vibe-first")
    )
    assert any("duplicate container_name: vibe-first" in e for e in errors)
    assert path.read_text() == before


def test_write_entry_rejects_invalid_spec(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    before = path.read_text()
    registry = Registry.load(path)
    errors = registry.write_entry(
        _entry("bad", host_port=70000, container_name="vibe-bad")
    )
    assert any("host_port" in e for e in errors)
    assert path.read_text() == before


def test_write_entry_missing_file(tmp_path):
    registry = Registry.from_dict({"models": []}, path=tmp_path / "nope.yaml")
    errors = registry.write_entry(
        _entry("x", host_port=8001, container_name="vibe-x")
    )
    assert any("not found" in e for e in errors)


def test_write_entry_round_trips_special_characters(tmp_path):
    """`use_case`/`display_name` with quotes, semicolons and unicode survive
    the append → reload round-trip (IMP-02.20 re-validation path)."""
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    registry = Registry.load(path)
    entry = _entry(
        "special",
        host_port=8002,
        container_name="vibe-special",
        use_case='quotes " and ; semicolons',
        display_name='Display "special" — 中文 🎉',
    )
    assert registry.write_entry(entry) == []
    on_disk = Registry.load(path)
    spec = on_disk.get("special")
    assert spec.use_case == entry["use_case"]
    assert spec.display_name == entry["display_name"]


def test_write_entry_preserves_file_mode(tmp_path):
    """IMP-02.23: the atomic replace keeps the file's mode."""
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    path.chmod(0o640)
    registry = Registry.load(path)
    assert (
        registry.write_entry(
            _entry("second", host_port=8002, container_name="vibe-second")
        )
        == []
    )
    assert path.stat().st_mode & 0o777 == 0o640


def test_write_entry_concurrent_single_winner(tmp_path):
    """IMP-02.20: two concurrent writes of the same entry — exactly one
    succeeds, the other reports the duplicate, and the file re-loads."""
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    registry = Registry.load(path)
    entry = _entry("second", host_port=8002, container_name="vibe-second")
    barrier = threading.Barrier(2)
    results: list[list[str]] = []

    def attempt() -> None:
        barrier.wait()
        results.append(registry.write_entry(dict(entry)))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [
        [],
        [
            "duplicate alias: second",
            "duplicate host_port: 8002",
            "duplicate container_name: vibe-second",
        ],
    ]
    on_disk = Registry.load(path)
    assert on_disk.aliases == ["first", "second"]
    assert "added by vibe register" in path.read_text()


def test_propose_auto_port_from_on_disk_registry(tmp_path):
    """IMP-02.24: with `self.path` set, the auto-port comes from the on-disk
    registry so it matches what `write_entry` re-checks."""
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {"models": [_entry("first", host_port=8001, container_name="vibe-first")]},
            sort_keys=False,
        )
    )
    registry = Registry.load(path)
    # The file grows an entry (port 8010) the in-memory snapshot does not know.
    path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    _entry("first", host_port=8001, container_name="vibe-first"),
                    _entry("second", host_port=8010, container_name="vibe-second"),
                ]
            },
            sort_keys=False,
        )
    )
    repo_dir = _make_hub_repo(tmp_path, "org/small", {"config.json": b"{}"})
    _weights(repo_dir, 2)
    entry, warnings, errors = registry.propose(
        {"hf_repo": "org/small", "alias": "small"}, CacheScanner(tmp_path)
    )
    assert errors == []
    # Memory-only would answer 8010; the on-disk registry says 8011.
    assert entry["host_port"] == 8011


# -- Phase 4: in-process registry RPCs ----------------------------------------


def fake_gpu() -> GpuStats:
    """Deterministic GPU stats for the in-process controller."""
    return GpuStats(
        used_mib=1234,
        total_mib=24576,
        utilization_pct=42,
        processes=(GpuProcess(pid=4321, name="vllm", used_mib=1000),),
    )


def free_port() -> int:
    """Pick a free loopback port by binding to :0 and releasing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _drain[T](stream: AsyncIterator[T]) -> list[T]:
    """Collect every item of an async generator into a list."""
    return [item async for item in stream]


@pytest.fixture(autouse=True)
def _loopback_bypasses_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep gRPC traffic to the in-process controller off any HTTP proxy."""
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("grpc_proxy", raising=False)


async def _start(
    settings: Settings,
    registry: Registry,
    docker: Any,
    gpu_fn: Callable[[], GpuStats],
) -> SimpleNamespace:
    """Start an in-process controller and return handles to it."""
    runner = JobRunner(settings, registry, docker=docker, get_gpu_stats=gpu_fn)
    buffer = RingBuffer()
    port = free_port()
    server = await build_server(
        settings,
        registry,
        runner=runner,
        docker=docker,
        gpu_fn=gpu_fn,
        buffer=buffer,
        port=port,
    )
    await server.start()
    client = ControllerClient(target=f"127.0.0.1:{port}")
    return SimpleNamespace(
        server=server, client=client, runner=runner, buffer=buffer, port=port
    )


async def _stop(ns: SimpleNamespace) -> None:
    """Tear down an in-process controller started by `_start`."""
    await ns.client.close()
    await ns.server.stop(0)


def _models_file(path: Path, aliases: list[str]) -> None:
    """Write a valid models.yaml with one entry per alias."""
    path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    _entry(alias, host_port=8000 + i, container_name=f"vibe-{alias}")
                    for i, alias in enumerate(aliases)
                ]
            },
            sort_keys=False,
        )
    )


async def test_reload_registry_rpc(tmp_path, fake_docker):
    """ReloadRegistry swaps on success and errors keep the old snapshot."""
    models = tmp_path / "models.yaml"
    _models_file(models, ["first"])
    registry = Registry.load(models)
    settings = Settings(vllm_image_tag="v0.11.2", vibe_data_dir=str(tmp_path))
    ns = await _start(settings, registry, fake_docker, fake_gpu)
    try:
        assert await ns.client.reload_registry() == []

        _models_file(models, ["first", "second"])
        assert await ns.client.reload_registry() == []
        aliases = [m.alias for m in await ns.client.list_models()]
        assert "second" in aliases

        models.write_text("models: [unclosed\n")
        errors = await ns.client.reload_registry()
        assert any("not valid YAML" in e for e in errors)
        aliases = [m.alias for m in await ns.client.list_models()]
        assert aliases == ["first", "second"]  # old snapshot intact
    finally:
        await _stop(ns)


async def test_propose_and_write_registration_rpc(tmp_path, fake_docker):
    """Propose fills defaults; Write appends and reloads (FR-04.5/FR-04.6)."""
    repo_dir = _make_hub_repo(tmp_path, "org/small", {"config.json": b"{}"})
    _weights(repo_dir, 2)
    models = tmp_path / "models.yaml"
    _models_file(models, ["first"])
    registry = Registry.load(models)
    settings = Settings(vllm_image_tag="v0.11.2", vibe_data_dir=str(tmp_path))
    ns = await _start(settings, registry, fake_docker, fake_gpu)
    try:
        spec = control_pb2.RegistrationSpec(hf_repo="org/small", alias="small")
        proposed = await ns.client.propose_registration(spec)
        assert proposed.errors == []
        assert proposed.model.alias == "small"
        # proto floats are 32-bit — compare approximately
        assert proposed.model.gpu_memory_utilization == pytest.approx(0.60)
        assert proposed.model.state == control_pb2.AVAILABLE
        assert proposed.model.cached_locally is True
        # IMP-01.5: the response round-trips a full registry entry.
        assert proposed.model.container_name == "vibe-small"
        assert list(proposed.model.extra_args) == []
        assert "# --- added by vibe register" in proposed.model.yaml_preview
        assert "alias: small" in proposed.model.yaml_preview

        written = await ns.client.write_registration(spec)
        assert written.ok is True
        assert written.errors == []
        aliases = [m.alias for m in await ns.client.list_models()]
        assert aliases == ["first", "small"]  # reloaded immediately
        assert "# --- added by vibe register" in models.read_text()
    finally:
        await _stop(ns)


async def test_write_registration_rpc_rejects_duplicate(tmp_path, fake_docker):
    models = tmp_path / "models.yaml"
    _models_file(models, ["first"])
    registry = Registry.load(models)
    settings = Settings(vllm_image_tag="v0.11.2", vibe_data_dir=str(tmp_path))
    ns = await _start(settings, registry, fake_docker, fake_gpu)
    try:
        spec = control_pb2.RegistrationSpec(hf_repo="org/first", alias="first")
        written = await ns.client.write_registration(spec)
        assert written.ok is False
        assert any("duplicate alias: first" in e for e in written.errors)
        assert "vibe register" not in models.read_text()  # nothing written
    finally:
        await _stop(ns)
