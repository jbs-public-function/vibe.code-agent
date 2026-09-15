"""CacheScanner unit tests against a real (synthetic) HF cache layout (M4.3).

The layouts use the standard `models--org--repo/{refs,snapshots/<40-hex>}`
shape, which the real `scan_cache_dir` parses offline — no monkeypatched
scanner, no network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vibe.controller import cache as cache_mod
from vibe.controller.cache import CacheScanner, CacheRepo
from vibe.registry import Registry

REV = "a" * 40


def make_cache_layout(
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


def _entry(alias: str, **overrides: Any) -> dict:
    """A single valid registry entry; `overrides` tweak one field."""
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


# -- scanning ----------------------------------------------------------------


def test_scan_lists_repos_sizes_and_timestamps(tmp_path: Path) -> None:
    make_cache_layout(
        tmp_path, "org/repo-a", {"model.safetensors": b"x" * 300, "config.json": b"{}"}
    )
    make_cache_layout(tmp_path, "org/repo-b", {"model.safetensors": b"y" * 50})
    repos = CacheScanner(tmp_path).list()
    assert {repo.repo_id for repo in repos} == {"org/repo-a", "org/repo-b"}
    a = next(repo for repo in repos if repo.repo_id == "org/repo-a")
    assert isinstance(a, CacheRepo)
    assert a.size_bytes == 302
    assert a.last_modified  # ISO 8601 timestamp string
    assert a.aliases == []


def test_scan_ttl_caches_for_30_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Results are cached for 30 s (NFR-04.2)."""
    make_cache_layout(tmp_path, "org/repo", {"model.safetensors": b"x" * 10})
    scanner = CacheScanner(tmp_path)
    real_scan = cache_mod.scan_cache_dir
    calls: list[int] = []

    def counting_scan(**kwargs: Any) -> Any:
        calls.append(1)
        return real_scan(**kwargs)

    monkeypatch.setattr(cache_mod, "scan_cache_dir", counting_scan)
    clock = [1000.0]
    # Replace the module-global `time` inside cache.py only, so the rest of
    # the test process keeps the real monotonic clock.
    monkeypatch.setattr(
        cache_mod, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )

    scanner.list()
    scanner.list()
    assert len(calls) == 1
    clock[0] += 31.0
    scanner.list()
    assert len(calls) == 2


def test_scan_missing_dir_returns_empty(tmp_path: Path) -> None:
    scanner = CacheScanner(tmp_path / "does-not-exist")
    assert scanner.list() == []


def test_scan_hub_as_file_returns_empty(tmp_path: Path) -> None:
    """A broken cache dir (hub is a file) degrades to an empty listing."""
    (tmp_path / "hub").write_text("not a directory")
    scanner = CacheScanner(tmp_path)
    assert scanner.list() == []


# -- aliases -----------------------------------------------------------------


def test_aliases_cross_reference_registry(tmp_path: Path) -> None:
    """Registry aliases whose hf_repo matches are listed (case-insensitive)."""
    make_cache_layout(tmp_path, "org/repo", {"model.safetensors": b"x"})
    registry = Registry.from_dict(
        {
            "models": [
                _entry("alias-a", hf_repo="ORG/Repo", host_port=8001),
                _entry("alias-b", hf_repo="org/repo", host_port=8002),
                _entry("alias-c", hf_repo="org/other", host_port=8003),
            ]
        }
    )
    repos = CacheScanner(tmp_path).list(registry)
    repo = next(r for r in repos if r.repo_id == "org/repo")
    assert repo.aliases == ["alias-a", "alias-b"]


def test_scan_without_registry_has_no_aliases(tmp_path: Path) -> None:
    make_cache_layout(tmp_path, "org/repo", {"model.safetensors": b"x"})
    repos = CacheScanner(tmp_path).scan()
    assert all(repo.aliases == [] for repo in repos)


# -- weights -----------------------------------------------------------------


def test_weights_bytes_sums_across_snapshots(tmp_path: Path) -> None:
    make_cache_layout(tmp_path, "org/repo", {"model.safetensors": b"x" * 300})
    make_cache_layout(tmp_path, "org/repo", {"model.safetensors": b"x" * 100}, rev="b" * 40)
    scanner = CacheScanner(tmp_path)
    assert scanner.weights_bytes("org/repo") == 400
    assert scanner.weights_bytes("org/missing") is None


def test_weights_bytes_zero_for_gguf_only(tmp_path: Path) -> None:
    """Cached without safetensors → 0 (not None): cached but weightless."""
    make_cache_layout(tmp_path, "org/gguf", {"model-Q4_K_M.gguf": b"gguf"})
    assert CacheScanner(tmp_path).weights_bytes("org/gguf") == 0


def test_weights_bytes_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"x" * 700)
    scanner = CacheScanner(tmp_path)
    assert scanner.weights_bytes(str(model_dir)) == 700
    assert scanner.weights_bytes(str(tmp_path / "nope")) is None


# -- files and repo dirs -----------------------------------------------------


def test_repo_files_and_repo_dir(tmp_path: Path) -> None:
    repo_dir = make_cache_layout(
        tmp_path, "org/repo", {"config.json": b"{}", "model.safetensors": b"x"}
    )
    scanner = CacheScanner(tmp_path)
    assert scanner.repo_dir("org/repo") == repo_dir
    assert scanner.repo_dir("org/missing") is None
    names = sorted(path.name for path in scanner.repo_files("org/repo"))
    assert names == ["config.json", "model.safetensors"]


# -- removal -----------------------------------------------------------------


def test_remove_and_idempotency(tmp_path: Path) -> None:
    """Removal deletes the dir; a second remove is (True, "") by design."""
    repo_dir = make_cache_layout(tmp_path, "org/repo", {"model.safetensors": b"x"})
    scanner = CacheScanner(tmp_path)
    assert scanner.remove("org/repo") == (True, "")
    assert not repo_dir.exists()
    assert scanner.repo_dir("org/repo") is None
    assert scanner.remove("org/repo") == (True, "")  # idempotent
    # Removal invalidates the scan TTL: the next list re-reads the disk.
    assert scanner.list() == []


def test_remove_refuses_local_paths(tmp_path: Path) -> None:
    removed, error = CacheScanner(tmp_path).remove(str(tmp_path / "models"))
    assert removed is False
    assert "local paths" in error


def test_is_cached(tmp_path: Path) -> None:
    make_cache_layout(tmp_path, "org/repo", {"model.safetensors": b"x"})
    scanner = CacheScanner(tmp_path)
    assert scanner.is_cached("org/repo") is True
    assert scanner.is_cached("org/missing") is False
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    assert scanner.is_cached(str(model_dir)) is True
    assert scanner.is_cached(str(tmp_path / "nope")) is False


# -- disk usage --------------------------------------------------------------


def test_disk_usage_reports_numbers(tmp_path: Path) -> None:
    total, free = CacheScanner(tmp_path).disk_usage()
    assert total > 0
    assert free > 0


def test_disk_usage_degrades_to_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: Path) -> Any:
        raise OSError("statvfs failed")

    monkeypatch.setattr(cache_mod, "shutil", SimpleNamespace(disk_usage=boom))
    assert CacheScanner(tmp_path).disk_usage() == (0, 0)
