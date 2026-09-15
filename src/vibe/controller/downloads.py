"""HF model downloads with streamed progress and cooperative cancel (M4.1).

`DownloadManager` runs `huggingface_hub.snapshot_download` in worker threads
from a bounded `ThreadPoolExecutor` (NFR-04.1: max 2 concurrent downloads).
Before downloading it preflights `HfApi.model_info(files_metadata=True)` so
`bytes_total` is meaningful (spec risk note); a failed preflight degrades to
`bytes_total == 0` rather than aborting.

Progress crosses the thread boundary through a custom `tqdm.std.tqdm`
subclass: every `update(n)` accumulates bytes on the handle and schedules a
snapshot push onto the event loop captured at `start()` time via
`loop.call_soon_threadsafe`. Cancellation is cooperative (FR-04.2): a
`threading.Event` per handle is checked inside `update()`, which raises the
private `_Cancelled` exception to unwind `snapshot_download`. Partial blobs
are never deleted here — they stay resumable in HF's cache.

Gated/401/403/404 errors from huggingface_hub map to terminal error strings
telling the user to check HF_TOKEN / accept the license (spec risks).

This module never imports the proto: it ships plain `DownloadHandle` /
`DownloadProgress` objects and the servicer translates them to
`control_pb2`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import uuid
from collections import deque
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, cast

import tqdm.std
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from vibe.config import Settings

# FR-04.2: keep the last 20 completed downloads.
HISTORY_LIMIT = 20

# NFR-04.1: at most two downloads run concurrently.
MAX_WORKERS = 2

# Per-subscriber progress queue capacity; writes beyond it are dropped so a
# slow/disconnected client can never make the queue grow unbounded
# (FR-02.11 semantics, mirroring servicer.FANOUT_CAPACITY).
QUEUE_CAPACITY = 200

# How often the event pump re-checks `handle.done` while the queue is idle.
DONE_POLL_SECONDS = 0.5

CANCELLED_ERROR = "cancelled"

# Aggregate bar descriptions used by huggingface_hub >= 1.28's
# snapshot_download: these never name a file, so they are not shown as
# `current_file`.
_AGGREGATE_DESCS = ("Downloading bytes", "Reconstructing", "Fetching ")


class _Cancelled(Exception):
    """Raised inside the tqdm hook to unwind `snapshot_download` (FR-04.2)."""


@dataclass
class DownloadProgress:
    """One progress snapshot shipped to RPC subscribers."""

    id: str
    progress: float
    bytes_done: int
    bytes_total: int
    current_file: str
    terminal: bool
    error: str


@dataclass
class DownloadHandle:
    """State of one download (FR-04.1).

    `done`/`ok`/`error`/`progress`/`bytes_*`/`current_file` are written only
    by the worker thread; the event loop reads them for terminal synthesis
    and `list()` snapshots. `subscribers` is mutated only on the loop.
    """

    id: str
    hf_repo: str
    include: list[str]
    revision: str
    done: bool = False
    ok: bool = False
    error: str = ""
    progress: float = 0.0
    bytes_done: int = 0
    bytes_total: int = 0
    current_file: str = ""
    cancel_event: Event = field(default_factory=Event)
    subscribers: set[asyncio.Queue[DownloadProgress]] = field(
        default_factory=set
    )
    loop: asyncio.AbstractEventLoop | None = None


def _repo_key(hf_repo: str) -> str:
    """Normalize a repo id for attach-to-in-flight matching (FR-04.1).

    Hub repo ids compare case-insensitively; local paths normalize through
    their absolute form.
    """
    repo = hf_repo.strip()
    if repo.startswith(("/", "~")):
        return str(Path(repo).expanduser().resolve())
    return repo.lower()


def _friendly_hf_error(exc: Exception, repo: str) -> str:
    """Map gated/401/403/404 huggingface_hub errors to actionable text."""
    if isinstance(exc, GatedRepoError):
        return (
            f"'{repo}' is gated — accept the license on huggingface.co "
            "and make sure HF_TOKEN is set"
        )
    if isinstance(exc, RevisionNotFoundError):
        return f"revision not found for '{repo}'"
    if isinstance(exc, RepositoryNotFoundError):
        return f"repository '{repo}' not found on huggingface.co"
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", 0)
        if status in (401, 403):
            return (
                f"access denied for '{repo}' ({status}) — check HF_TOKEN and "
                "accept the license on huggingface.co"
            )
        if status == 404:
            return f"repository '{repo}' not found on huggingface.co"
    return str(exc)


class DownloadManager:
    """Runs HF downloads in worker threads and fans progress out (M4.1)."""

    def __init__(self, settings: Settings) -> None:
        self._hf_token = settings.hf_token
        self._cache_dir = settings.hf_home / "hub"
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._inflight: dict[str, DownloadHandle] = {}
        self._history: deque[DownloadHandle] = deque(maxlen=HISTORY_LIMIT)

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        hf_repo: str,
        include: Sequence[str] | None = None,
        revision: str | None = None,
    ) -> DownloadHandle:
        """Start a download, or attach to an in-flight one (FR-04.1).

        Must be called on the event loop so the progress-push loop can be
        captured. Raises `ValueError` for an empty repo id.
        """
        repo = hf_repo.strip()
        if not repo:
            raise ValueError("hf_repo is required")
        self._reap_done()
        existing = self._inflight.get(_repo_key(repo))
        if existing is not None:
            return existing  # FR-04.1: duplicates attach to the running one
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None  # no running loop — progress pushes are skipped
        handle = DownloadHandle(
            id=uuid.uuid4().hex[:8],
            hf_repo=repo,
            include=list(include or []),
            revision=(revision or "").strip(),
            loop=loop,
        )
        self._inflight[_repo_key(repo)] = handle
        self._executor.submit(self._run, handle)
        return handle

    def cancel(self, download_id: str) -> bool:
        """Request cooperative cancellation; False when unknown or done."""
        self._reap_done()
        for handle in (*self._inflight.values(), *self._history):
            if handle.id == download_id and not handle.done:
                handle.cancel_event.set()
                return True
        return False

    def list(self) -> list[DownloadHandle]:
        """In-flight first, then completed history, newest first (FR-04.2)."""
        self._reap_done()
        handles = list(self._inflight.values())
        handles.extend(self._history)
        return handles

    # -- event stream ------------------------------------------------------

    async def events(
        self, handle: DownloadHandle
    ) -> AsyncIterator[DownloadProgress]:
        """Stream one download's progress until its terminal event.

        The download lives in the manager, not this generator: when the
        subscriber goes away the worker keeps running (FR-02.11 semantics).
        The terminal event is synthesized from the handle's final state, so
        it is delivered even if intermediate snapshots were dropped.
        """
        queue: asyncio.Queue[DownloadProgress] = asyncio.Queue(
            maxsize=QUEUE_CAPACITY
        )
        handle.subscribers.add(queue)
        try:
            while True:
                if handle.done:
                    yield self._terminal_progress(handle)
                    self._reap(handle)
                    return
                try:
                    snapshot = await asyncio.wait_for(
                        queue.get(), timeout=DONE_POLL_SECONDS
                    )
                except TimeoutError:
                    continue
                yield snapshot
        finally:
            handle.subscribers.discard(queue)

    # -- internal ----------------------------------------------------------

    def _reap_done(self) -> None:
        """Move finished handles from in-flight to the history (loop-side)."""
        for handle in list(self._inflight.values()):
            if handle.done:
                self._reap(handle)

    def _reap(self, handle: DownloadHandle) -> None:
        key = _repo_key(handle.hf_repo)
        if self._inflight.get(key) is handle:
            del self._inflight[key]
        if not any(h.id == handle.id for h in self._history):
            self._history.appendleft(handle)  # newest first on iteration

    def _run(self, handle: DownloadHandle) -> None:
        """Worker thread: preflight totals, download, finalize (FR-04.1)."""
        try:
            try:
                handle.bytes_total = self._preflight_total(handle)
            except (
                GatedRepoError,
                HfHubHTTPError,
                RepositoryNotFoundError,
            ) as exc:
                self._fail(handle, _friendly_hf_error(exc, handle.hf_repo))
                return
            except Exception:  # noqa: BLE001
                # Preflight is best-effort: download anyway without a total.
                handle.bytes_total = 0
            try:
                snapshot_download(
                    repo_id=handle.hf_repo,
                    allow_patterns=handle.include or None,
                    revision=handle.revision or None,
                    token=self._hf_token or None,
                    cache_dir=str(self._cache_dir),
                    tqdm_class=cast(Any, _progress_tqdm_class(self, handle)),
                )
            except _Cancelled:
                self._fail(handle, CANCELLED_ERROR)
                return
            except (
                GatedRepoError,
                HfHubHTTPError,
                RepositoryNotFoundError,
            ) as exc:
                self._fail(handle, _friendly_hf_error(exc, handle.hf_repo))
                return
            except Exception as exc:  # noqa: BLE001
                self._fail(handle, str(exc))
                return
            if handle.cancel_event.is_set():
                # Defensive: snapshot_download normally re-raises _Cancelled,
                # but never report success for a download the user cancelled.
                self._fail(handle, CANCELLED_ERROR)
                return
            handle.error = ""
            handle.ok = True
            handle.progress = 1.0
            handle.done = True
        finally:
            self._push(handle)

    def _fail(self, handle: DownloadHandle, error: str) -> None:
        handle.done = True
        handle.ok = False
        handle.error = error

    def _preflight_total(self, handle: DownloadHandle) -> int:
        """Sum the sizes of the matched files via model_info (spec risk)."""
        if handle.hf_repo.startswith(("/", "~")):
            return 0  # local paths are a Phase-5 concern — nothing to query
        info = HfApi(token=self._hf_token or None).model_info(
            handle.hf_repo,
            revision=handle.revision or None,
            files_metadata=True,
        )
        total = 0
        for sibling in info.siblings or []:
            name = getattr(sibling, "rfilename", "")
            if handle.include and not any(
                fnmatch.fnmatch(name, pattern) for pattern in handle.include
            ):
                continue
            size = getattr(sibling, "size", None)
            if size:
                total += int(size)
        return total

    def _record_bytes(self, handle: DownloadHandle, n: int, desc: str) -> None:
        """Accumulate one byte delta and push a snapshot (worker thread)."""
        if n <= 0:
            return
        handle.bytes_done += n
        if handle.bytes_total and handle.bytes_done > handle.bytes_total:
            # The preflight can undercount: grow the total, never shrink it.
            handle.bytes_total = handle.bytes_done
        handle.progress = (
            min(handle.bytes_done / handle.bytes_total, 1.0)
            if handle.bytes_total
            else 0.0
        )
        if desc and not desc.startswith(_AGGREGATE_DESCS):
            handle.current_file = (
                desc.removesuffix(": downloading bytes")
                .removesuffix(": reconstructing file")
                .strip()
            )
        self._push(handle)

    def _push(self, handle: DownloadHandle) -> None:
        """Ship a snapshot to every subscriber via the captured loop."""
        loop = handle.loop
        if loop is None or loop.is_closed():
            return
        snapshot = DownloadProgress(
            id=handle.id,
            progress=handle.progress,
            bytes_done=handle.bytes_done,
            bytes_total=handle.bytes_total,
            current_file=handle.current_file,
            terminal=False,
            error=handle.error,
        )
        loop.call_soon_threadsafe(self._offer_to_subscribers, handle, snapshot)

    def _offer_to_subscribers(
        self, handle: DownloadHandle, snapshot: DownloadProgress
    ) -> None:
        """Runs on the event loop: bounded fan-out, dropping when full."""
        for queue in list(handle.subscribers):
            try:
                queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass

    @staticmethod
    def _terminal_progress(handle: DownloadHandle) -> DownloadProgress:
        return DownloadProgress(
            id=handle.id,
            progress=handle.progress,
            bytes_done=handle.bytes_done,
            bytes_total=handle.bytes_total,
            current_file=handle.current_file,
            terminal=True,
            error=handle.error,
        )


def _progress_tqdm_class(
    manager: DownloadManager, handle: DownloadHandle
) -> type[tqdm.std.tqdm]:
    """Build the tqdm subclass feeding one handle (FR-04.1 progress hook).

    huggingface_hub instantiates the class a few times per snapshot
    (transfer bar, reconstruction bar, file-count bar). Only `unit="B"`
    bars other than the network-transfer bar carry disk bytes, so those
    drive the accumulator; the file-count bar and the transfer bar only
    serve the cooperative-cancel check.
    """

    class _ProgressTqdm(tqdm.std.tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Progress flows through the RPC stream, not the terminal.
            kwargs["disable"] = True
            desc = str(kwargs.get("desc") or "")
            super().__init__(*args, **kwargs)
            # tqdm's disabled constructor path skips setting `desc`.
            self.desc = desc
            self._is_bytes_bar = str(kwargs.get("unit") or "") == "B"
            # Skip network-transfer bars ("Downloading bytes", or XET's
            # per-file "<name>: downloading bytes") so bytes are never
            # double-counted against the reconstruction bars.
            self._is_transfer_bar = desc.endswith("downloading bytes")

        def update(self, n: float | None = 1) -> bool | None:
            if handle.cancel_event.is_set():
                raise _Cancelled()
            if self._is_bytes_bar and not self._is_transfer_bar:
                manager._record_bytes(handle, int(n or 0), str(self.desc or ""))
            return super().update(n)

    return _ProgressTqdm
