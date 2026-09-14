"""Registry validation and lookup tests (FR-00.2)."""

from __future__ import annotations

import pytest

from vibe.registry import ModelSpec, Registry, RegistryError


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
