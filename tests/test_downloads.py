"""DownloadManager + in-process download/cache RPC tests (M4.1/M4.3).

Network-free: `vibe.controller.downloads.snapshot_download` and
`vibe.controller.downloads.HfApi` are monkeypatched with fakes that drive
the custom tqdm class through synthetic per-file progress. The RPC tests
exercise `build_server` over a real gRPC channel on a free loopback port
(the `_start`/`_stop` pattern from `tests/test_client.py`), with the HF
cache pointed at a tmp dir via `Settings(vibe_data_dir=...)`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import socket
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from vibe.client import ControllerClient
from vibe.config import Settings
from vibe.controller import RingBuffer, build_server
from vibe.controller import downloads as downloads_mod
from vibe.controller.downloads import DownloadManager
from vibe.gen.vibe.v1 import control_pb2
from vibe.gpu import GpuProcess, GpuStats
from vibe.jobs import JobRunner
from vibe.registry import Registry

FAKE_FILES = [
    ("config.json", 10),
    ("model-00001-of-00002.safetensors", 600),
    ("model-00002-of-00002.safetensors", 400),
]

REV = "a" * 40


# -- fakes ------------------------------------------------------------------


class FakeHfApi:
    """Synthetic file metadata for the preflight `bytes_total` computation.

    Set `error` to make `model_info` raise, exercising the error mapping.
    """

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.error: Exception | None = None

    def model_info(
        self,
        repo_id: str,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename=name, size=size)
                for name, size in FAKE_FILES
            ]
        )


class FakeSnapshotDownload:
    """Fake `snapshot_download` driving the custom tqdm class.

    Instantiates one `unit="B"` bar per file (like huggingface_hub's
    per-file bars) and pushes `chunk`-sized updates; cancellation works
    because the tqdm subclass raises `_Cancelled` from `update()` when the
    handle's event is set, unwinding through this loop untouched.
    """

    def __init__(
        self, files: list[tuple[str, int]], chunk: int, delay: float
    ) -> None:
        self.files = files
        self.chunk = chunk
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def __call__(self, repo_id: str, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        tqdm_class: Any = kwargs.get("tqdm_class")
        patterns = kwargs.get("allow_patterns")
        for name, size in self.files:
            # Mirror snapshot_download's allow_patterns filtering.
            if patterns and not any(
                fnmatch.fnmatch(name, pattern) for pattern in patterns
            ):
                continue
            pbar = tqdm_class(desc=name, total=size, unit="B")
            remaining = size
            while remaining > 0:
                time.sleep(self.delay)
                step = min(self.chunk, remaining)
                pbar.update(step)
                remaining -= step
            pbar.close()
        return str(Path(str(kwargs.get("cache_dir") or ".")) / "snapshots" / REV)


def _gated_error() -> GatedRepoError:
    response = httpx.Response(
        401, request=httpx.Request("GET", "https://huggingface.co/api/models/x")
    )
    return GatedRepoError("gated repo", response=response)


def _http_error(status: int) -> HfHubHTTPError:
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://huggingface.co/api/models/x")
    )
    return HfHubHTTPError("http error", response=response)


# -- DownloadManager unit tests ----------------------------------------------


async def test_progress_streams_and_terminates_ok(monkeypatch: pytest.MonkeyPatch):
    """Events stream progress and end with a terminal success (FR-04.1)."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=100, delay=0.0),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit", token="test-token"))
    handle = manager.start("org/repo", include=["*.safetensors"])
    events = [event async for event in manager.events(handle)]

    assert events
    assert events[-1].terminal is True
    assert events[-1].error == ""
    assert events[-1].progress == 1.0
    # Preflight summed only the matched files (config.json excluded).
    assert events[-1].bytes_total == 1000
    assert handle.done and handle.ok
    assert manager.list() == [handle]  # completed: moved to history

    # The worker forwarded the request's filters, cache location and token.
    fake = cast(FakeSnapshotDownload, downloads_mod.snapshot_download)
    assert fake.calls[0]["allow_patterns"] == ["*.safetensors"]
    assert str(fake.calls[0]["cache_dir"]) == "/tmp/vibe-dl-unit/hub"
    assert fake.calls[0]["token"] == "test-token"


async def test_attach_to_inflight_returns_same_handle(monkeypatch: pytest.MonkeyPatch):
    """A duplicate start for a running repo attaches (FR-04.1)."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=10, delay=0.02),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    first = manager.start("Org/Repo")
    second = manager.start("org/repo")  # normalized key: case-insensitive
    assert second is first

    assert manager.cancel(first.id) is True
    async for _ in manager.events(first):
        pass
    assert manager.cancel(first.id) is False  # already done

    third = manager.start("org/repo")  # completed downloads restart fresh
    assert third is not first
    assert manager.cancel(third.id) is True


async def test_cancel_terminates_with_cancelled(monkeypatch: pytest.MonkeyPatch):
    """Cancel stops the worker cooperatively with error="cancelled" (FR-04.2)."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=10, delay=0.01),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    handle = manager.start("org/repo")
    await asyncio.sleep(0.05)  # let it make progress first
    assert manager.cancel(handle.id) is True
    events = [event async for event in manager.events(handle)]
    assert events[-1].terminal is True
    assert events[-1].error == "cancelled"
    assert handle.done and not handle.ok


async def test_history_bounded_to_20(monkeypatch: pytest.MonkeyPatch):
    """Only the last 20 completed downloads are kept, newest first (FR-04.2)."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=100, delay=0.0),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    for index in range(25):
        handle = manager.start(f"org/repo{index}")
        async for _ in manager.events(handle):
            pass  # wait for each terminal so completion order is deterministic

    snapshot = manager.list()
    assert len(snapshot) == 20
    assert [h.hf_repo for h in snapshot] == [
        f"org/repo{index}" for index in range(24, 4, -1)
    ]


async def test_gated_preflight_maps_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """Gated repos surface the license/HF_TOKEN hint (spec risks)."""
    api = FakeHfApi()
    api.error = _gated_error()
    monkeypatch.setattr(downloads_mod, "HfApi", lambda token=None: api)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=100, delay=0.0),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    handle = manager.start("org/gated")
    events = [event async for event in manager.events(handle)]
    assert events[-1].terminal is True
    assert "license" in events[-1].error
    assert "HF_TOKEN" in events[-1].error
    assert handle.done and not handle.ok


async def test_download_401_maps_to_friendly_error(monkeypatch: pytest.MonkeyPatch):
    """A 403 from snapshot_download becomes the access-denied message."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)

    def denied(repo_id: str, **kwargs: Any) -> str:
        raise _http_error(403)

    monkeypatch.setattr(downloads_mod, "snapshot_download", denied)
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    handle = manager.start("org/denied")
    events = [event async for event in manager.events(handle)]
    assert events[-1].terminal is True
    assert "HF_TOKEN" in events[-1].error
    assert handle.done and not handle.ok


async def test_preflight_failure_degrades_to_unknown_total(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failed preflight yields bytes_total=0 but the download proceeds."""
    api = FakeHfApi()
    api.error = OSError("network hiccup")
    monkeypatch.setattr(downloads_mod, "HfApi", lambda token=None: api)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=100, delay=0.0),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    handle = manager.start("org/repo")
    events = [event async for event in manager.events(handle)]
    assert handle.bytes_total == 0
    assert handle.done and handle.ok
    assert events[-1].progress == 1.0


async def test_empty_repo_raises_value_error():
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    with pytest.raises(ValueError):
        manager.start("  ")


async def test_attach_with_different_include_raises(monkeypatch: pytest.MonkeyPatch):
    """IMP-02.13: a different include/revision for an in-flight repo must not
    silently attach — the RPC maps this ValueError to INVALID_ARGUMENT."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=10, delay=0.02),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    first = manager.start("org/repo", include=["*.safetensors"])
    with pytest.raises(ValueError, match="already in flight"):
        manager.start("org/repo", include=["config.json"])
    with pytest.raises(ValueError, match="already in flight"):
        manager.start("org/repo", revision="main")
    # The same (repo, revision, sorted include) still attaches.
    assert manager.start("org/repo", include=["*.safetensors"]) is first
    assert manager.cancel(first.id) is True


async def test_terminal_progress_ok_semantics(monkeypatch: pytest.MonkeyPatch):
    """IMP-01.4/02.12: the terminal event carries an explicit `ok` flag —
    True exactly once on success, False with a non-empty error otherwise."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=100, delay=0.0),
    )
    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    events = [event async for event in manager.events(manager.start("org/ok"))]
    ok_events = [event for event in events if event.ok]
    assert len(ok_events) == 1
    assert ok_events[0].terminal is True
    assert ok_events[0].error == ""


@pytest.mark.parametrize(
    "scenario",
    [
        "cancel",
        "gated",
        "http_403",
        "generic",
    ],
)
async def test_terminal_progress_failures_have_error(
    monkeypatch: pytest.MonkeyPatch, scenario: str
):
    """IMP-02.12 stress: every terminal event with ok=False has a non-empty
    error, so a failure can never look like a success."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    if scenario == "gated":
        api = FakeHfApi()
        api.error = _gated_error()
        monkeypatch.setattr(downloads_mod, "HfApi", lambda token=None: api)
        monkeypatch.setattr(
            downloads_mod,
            "snapshot_download",
            FakeSnapshotDownload(FAKE_FILES, chunk=100, delay=0.0),
        )
    elif scenario == "http_403":
        def denied(repo_id: str, **kwargs: Any) -> str:
            raise _http_error(403)

        monkeypatch.setattr(downloads_mod, "snapshot_download", denied)
    elif scenario == "generic":
        def boom(repo_id: str, **kwargs: Any) -> str:
            raise OSError("disk exploded")

        monkeypatch.setattr(downloads_mod, "snapshot_download", boom)
    else:  # cancel
        monkeypatch.setattr(
            downloads_mod,
            "snapshot_download",
            FakeSnapshotDownload(FAKE_FILES, chunk=10, delay=0.01),
        )

    manager = DownloadManager(_settings("/tmp/vibe-dl-unit"))
    handle = manager.start(f"org/{scenario}")
    if scenario == "cancel":
        await asyncio.sleep(0.05)  # let it make progress first
        assert manager.cancel(handle.id) is True
    events = [event async for event in manager.events(handle)]
    terminal = events[-1]
    assert terminal.terminal is True
    assert terminal.ok is False
    assert terminal.error


# -- in-process RPC tests ----------------------------------------------------


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
    settings: Settings, registry: Registry, docker: Any, gpu_fn: Callable[[], GpuStats]
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


def _settings(data_dir: str, token: str = "") -> Settings:
    return Settings(vllm_image_tag="v0.11.2", vibe_data_dir=data_dir, hf_token=token)


def _hub_layout(hf_home: Path, repo_id: str, files: dict[str, bytes]) -> Path:
    """A minimal HF cache layout one repo (parses with real scan_cache_dir)."""
    hub = hf_home / "hub"
    repo_dir = hub / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / REV
    snap.mkdir(parents=True)
    refs = repo_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text(REV)
    for name, content in files.items():
        (snap / name).write_bytes(content)
    return repo_dir


async def test_download_model_rpc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry: Registry,
    fake_docker: Any,
):
    """DownloadModel streams progress over gRPC and lands in ListDownloads."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=200, delay=0.001),
    )
    ns = await _start(_settings(str(tmp_path)), registry, fake_docker, fake_gpu)
    try:
        events = await _drain(
            ns.client.download("org/repo", include=["*.safetensors"])
        )
        assert events
        assert events[-1].terminal is True
        assert events[-1].error == ""
        assert events[-1].bytes_total == 1000
        infos = await ns.client.list_downloads()
        assert any(
            info.hf_repo == "org/repo" and info.done and info.ok
            for info in infos
        )
    finally:
        await _stop(ns)


async def test_cancel_download_rpc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry: Registry,
    fake_docker: Any,
):
    """CancelDownload stops an in-flight stream with a cancelled terminal."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=10, delay=0.05),
    )
    ns = await _start(_settings(str(tmp_path)), registry, fake_docker, fake_gpu)
    try:
        task = asyncio.create_task(_drain(ns.client.download("org/repo")))
        download_id = ""
        for _ in range(200):
            infos = await ns.client.list_downloads()
            if infos:
                download_id = infos[0].id
                break
            await asyncio.sleep(0.01)
        assert download_id
        assert await ns.client.cancel_download(download_id) is True
        events = await task
        assert events[-1].terminal is True
        assert events[-1].error == "cancelled"
    finally:
        await _stop(ns)


async def test_list_cache_rpc(
    tmp_path: Path, registry: Registry, fake_docker: Any
):
    """ListCache reports the tmp HF cache and the disk usage (FR-04.4)."""
    _hub_layout(tmp_path, "org/repo", {"model.safetensors": b"x" * 100})
    ns = await _start(_settings(str(tmp_path)), registry, fake_docker, fake_gpu)
    try:
        response = await ns.client.list_cache()
        assert response.disk_total_bytes > 0
        entries = list(response.entries)
        assert [entry.repo_id for entry in entries] == ["org/repo"]
        assert entries[0].size_bytes == 100
        assert entries[0].last_modified
    finally:
        await _stop(ns)


async def test_remove_cached_rpc(
    tmp_path: Path, registry: Registry, fake_docker: Any
):
    """RemoveCached removes the repo dir and is idempotent (FR-04.4)."""
    _hub_layout(tmp_path, "org/repo", {"model.safetensors": b"x" * 100})
    ns = await _start(_settings(str(tmp_path)), registry, fake_docker, fake_gpu)
    try:
        removed, error = await ns.client.remove_cached("org/repo")
        assert removed is True
        assert error == ""
        response = await ns.client.list_cache()
        assert list(response.entries) == []
        removed, error = await ns.client.remove_cached("org/repo")
        assert removed is True  # idempotent
    finally:
        await _stop(ns)


async def test_remove_cached_refused_while_download_inflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry: Registry,
    fake_docker: Any,
):
    """IMP-02.4: RemoveCached refuses while the repo's download runs."""
    monkeypatch.setattr(downloads_mod, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        downloads_mod,
        "snapshot_download",
        FakeSnapshotDownload(FAKE_FILES, chunk=10, delay=0.05),
    )
    ns = await _start(_settings(str(tmp_path)), registry, fake_docker, fake_gpu)
    try:
        task = asyncio.create_task(_drain(ns.client.download("org/repo")))
        infos: list[control_pb2.DownloadInfo] = []
        for _ in range(200):
            infos = await ns.client.list_downloads()
            if infos:
                break
            await asyncio.sleep(0.01)
        assert infos
        removed, error = await ns.client.remove_cached("org/repo")
        assert removed is False
        assert error == "download in flight"
        await ns.client.cancel_download(infos[0].id)
        await task
    finally:
        await _stop(ns)
