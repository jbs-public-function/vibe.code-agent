"""HF cache scanning and management (M4.3, NFR-04.2, FR-04.9).

`CacheScanner` wraps `huggingface_hub.scan_cache_dir` with a 30 s TTL
(NFR-04.2) so the `/developer` page stays snappy even on a multi-TB volume.
Missing or broken cache dirs degrade to an empty listing rather than
raising. It also computes the `*.safetensors` weight sizes used by the
register heuristics, cross-references registry aliases, reports the
filesystem's free space, and removes whole `models--...` repo dirs.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import scan_cache_dir

if TYPE_CHECKING:
    from vibe.registry import Registry

# NFR-04.2: cache `scan_cache_dir` results for 30 seconds.
SCAN_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class CacheRepo:
    """One cached repo as shown by `ListCache` (FR-04.4)."""

    repo_id: str
    size_bytes: int
    last_modified: str  # ISO 8601
    aliases: list[str] = field(default_factory=list)


class CacheScanner:
    """TTL-cached view of `HF_HOME/hub` plus removal helpers."""

    def __init__(self, hf_home: Path) -> None:
        self.hf_home = Path(hf_home)
        self._repos: list[CacheRepo] = []
        self._scanned_at = 0.0
        # IMP-02.16: `to_thread` callers (ListModels, ListCache, register
        # heuristics) can refresh concurrently; serialize so the scan runs
        # once and both callers share one `scan_cache_dir` traversal.
        self._scan_lock = threading.Lock()

    @property
    def hub_dir(self) -> Path:
        """The HF hub cache dir that `snapshot_download` populates."""
        return self.hf_home / "hub"

    # -- scanning ----------------------------------------------------------

    def scan(self, registry: Registry | None = None) -> list[CacheRepo]:
        """Cached repos, oldest scans reused for `SCAN_TTL_SECONDS` (NFR-04.2).

        With a registry, each repo's `aliases` lists the registry aliases
        whose `hf_repo` points at it (case-insensitive).
        """
        self._ensure_fresh()
        if registry is None:
            return list(self._repos)
        by_repo: dict[str, list[str]] = {}
        for spec in registry.all():
            by_repo.setdefault(spec.hf_repo.lower(), []).append(spec.alias)
        return [
            CacheRepo(
                repo.repo_id,
                repo.size_bytes,
                repo.last_modified,
                by_repo.get(repo.repo_id.lower(), []),
            )
            for repo in self._repos
        ]

    def list(self, registry: Registry | None = None) -> list[CacheRepo]:
        """Alias for `scan` (the servicer calls `list`)."""
        return self.scan(registry)

    def _ensure_fresh(self) -> None:
        with self._scan_lock:
            now = time.monotonic()
            if now - self._scanned_at >= SCAN_TTL_SECONDS:
                self._repos = self._scan_raw()
                self._scanned_at = now

    def _scan_raw(self) -> list[CacheRepo]:
        try:
            info = scan_cache_dir(cache_dir=str(self.hub_dir))
        except Exception:  # noqa: BLE001
            # Missing or malformed cache dirs degrade to an empty listing
            # (FR-04.9: the /models page must still render).
            return []
        repos = [
            CacheRepo(
                repo_id=cached.repo_id,
                size_bytes=int(cached.size_on_disk),
                # IMP-02.18: UTC — every other timestamp in the codebase
                # is emitted as a UTC ISO 8601 string.
                last_modified=datetime.fromtimestamp(
                    cached.last_modified, UTC
                ).isoformat(),
                aliases=[],
            )
            for cached in info.repos
        ]
        repos.sort(key=lambda repo: repo.repo_id)
        return repos

    def is_cached(self, repo_id: str) -> bool:
        """Whether the repo has a snapshot locally (feeds `cached_locally`).

        Local paths (absolute or `~`, IMP-02.11) are checked directly; hub
        repo ids compare case-insensitively against the TTL-cached scan.
        """
        path = Path(repo_id).expanduser()
        if path.is_absolute():
            return path.is_dir()  # local-path models (Phase 5 round-trip)
        self._ensure_fresh()
        key = repo_id.lower()
        return any(repo.repo_id.lower() == key for repo in self._repos)

    # -- sizes and files ---------------------------------------------------

    def weights_bytes(self, repo_id: str) -> int | None:
        """Sum of `*.safetensors` sizes in the repo's newest snapshot.

        IMP-02.17: only the newest snapshot is measured — summing across
        revisions double-counts the same weights and can trip the
        > 10 GiB register refusal. `None` when the repo is not cached; `0`
        when cached without weights (e.g. a GGUF-only snapshot). Local
        absolute paths are measured in place.
        """
        base = self._snapshot_root(repo_id)
        if base is None:
            return None
        if base.name == "snapshots":
            snapshot = self._newest_snapshot(base)
            if snapshot is None:
                return 0
        else:
            snapshot = base  # local path: the dir itself
        total = 0
        for path in snapshot.rglob("*.safetensors"):
            try:
                total += path.stat().st_size  # follows symlinks into blobs/
            except OSError:
                continue  # vanished mid-scan — skip
        return total

    @staticmethod
    def _newest_snapshot(snapshots: Path) -> Path | None:
        """The snapshot the `main` ref points at, else newest by mtime."""
        ref = snapshots.parent / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text().strip()
            candidate = snapshots / revision
            if candidate.is_dir():
                return candidate
        dirs = [path for path in snapshots.iterdir() if path.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda path: path.stat().st_mtime)

    def repo_files(self, repo_id: str) -> list[Path]:
        """Every regular file under the repo's snapshot dirs (symlinks)."""
        base = self._snapshot_root(repo_id)
        if base is None:
            return []
        return [
            path
            for path in base.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        ]

    def repo_dir(self, repo_id: str) -> Path | None:
        """The repo's `models--...` dir, or None when not cached."""
        if Path(repo_id).expanduser().is_absolute():
            return None
        path = self.hub_dir / f"models--{repo_id.replace('/', '--')}"
        return path if path.is_dir() else None

    def _snapshot_root(self, repo_id: str) -> Path | None:
        """Snapshots dir for a hub repo, or the dir itself for a local path."""
        path = Path(repo_id).expanduser()
        if path.is_absolute():
            return path if path.is_dir() else None
        snapshots = self.hub_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        return snapshots if snapshots.is_dir() else None

    # -- disk and removal --------------------------------------------------

    def disk_usage(self) -> tuple[int, int]:
        """(total, free) bytes of the HF_HOME filesystem; (0, 0) on error."""
        try:
            usage = shutil.disk_usage(self.hf_home)
        except OSError:
            return (0, 0)
        return (usage.total, usage.free)

    def remove(self, repo_id: str) -> tuple[bool, str]:
        """Remove one repo's cache dir (FR-04.4).

        Idempotent by design: removing an absent entry is `(True, "")`.
        Local absolute paths are not managed by the HF cache and are
        refused, never deleted.
        """
        if Path(repo_id).expanduser().is_absolute():
            return (False, "local paths are not managed by the HF cache")
        path = self.hub_dir / f"models--{repo_id.replace('/', '--')}"
        if not path.exists():
            return (True, "")
        try:
            shutil.rmtree(path)
        except OSError as exc:
            return (False, str(exc))
        self._scanned_at = 0.0  # the next scan re-reads the disk
        return (True, "")
