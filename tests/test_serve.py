"""`vibe serve` composition-root tests (IMP-03.1–03.3, 03.17).

The composite path is exercised offline: `build_server`/`build_app`/
`build_admin_app`/`ControllerClient`/uvicorn are fakes, and the installed
SIGTERM handler is driven directly through the loop's signal-handler table.
"""

from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace
from typing import Any

import pytest

import vibe.serve as sv
from vibe.errors import EXIT_OK, EXIT_RUNTIME
from vibe.serve import serve


class FakeUvicorn:
    """`uvicorn.Server` stand-in: drains when `should_exit` is set."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.should_exit = False
        self.crash: Exception | None = None

    async def serve(self) -> None:
        while True:
            if self.crash is not None:
                raise self.crash
            if self.should_exit:
                break
            await asyncio.sleep(0.005)


class FakeGrpcServer:
    """`grpc.aio.Server` stand-in with `start`/`wait_for_termination`/`stop`."""

    def __init__(self) -> None:
        self.started = False
        self.stops: list[int] = []
        self._terminated = asyncio.Event()

    async def start(self) -> None:
        self.started = True

    async def wait_for_termination(self) -> None:
        await self._terminated.wait()

    async def stop(self, grace: int = 0) -> None:
        self.stops.append(grace)
        self._terminated.set()


@pytest.fixture
def serve_wiring(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Wire `vibe.serve` with fakes; `created` lists the two HTTP servers."""
    grpc = FakeGrpcServer()
    created: list[FakeUvicorn] = []

    async def build_server(settings: Any, registry: Any, **kwargs: object) -> FakeGrpcServer:
        return grpc

    def build_app(settings: Any, client: Any) -> object:
        return object()

    def build_admin_app(settings: Any, client: Any) -> object:
        return object()

    class FakeConfig:
        def __init__(
            self, app: Any, host: str, port: int, log_level: str
        ) -> None:
            self.app, self.host, self.port, self.log_level = (
                app,
                host,
                port,
                log_level,
            )

    class FakeServer(FakeUvicorn):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            created.append(self)

    monkeypatch.setenv("VLLM_IMAGE_TAG", "v0.11.2")
    monkeypatch.setattr(sv, "build_server", build_server)
    monkeypatch.setattr(sv, "build_app", build_app)
    monkeypatch.setattr(sv, "build_admin_app", build_admin_app)
    monkeypatch.setattr(sv, "ControllerClient", lambda settings: None)
    monkeypatch.setattr(sv, "uvicorn", SimpleNamespace(Config=FakeConfig))
    monkeypatch.setattr(sv, "_SignalOwnedServer", FakeServer)
    return SimpleNamespace(grpc=grpc, created=created)


async def test_serve_component_crash_exits_runtime(serve_wiring: SimpleNamespace) -> None:
    """IMP-03.2: a crashed HTTP component takes the process down non-zero."""
    task = asyncio.create_task(serve(None))
    for _ in range(200):
        if len(serve_wiring.created) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(serve_wiring.created) == 2, "serve() never built the HTTP servers"
    serve_wiring.created[0].crash = RuntimeError("bind error: 8080 already in use")
    code = await asyncio.wait_for(task, timeout=10)
    assert code == EXIT_RUNTIME
    # The others were shut down, the controller drained with grace=2.
    assert serve_wiring.grpc.stops == [2]
    assert serve_wiring.created[1].should_exit is True


async def test_serve_signal_stops_all_components(serve_wiring: SimpleNamespace) -> None:
    """IMP-03.1: one SIGTERM handler stops both HTTP servers + the controller,
    and the process exits 0."""
    task = asyncio.create_task(serve(None))
    loop = asyncio.get_running_loop()
    # Wait until the composite handler is installed, then drive it directly
    # (sending a real signal would kill the pytest process).
    handler: Any = None
    for _ in range(200):
        handler = getattr(loop, "_signal_handlers", {}).get(signal.SIGTERM)
        if handler is not None:
            break
        await asyncio.sleep(0.01)
    assert handler is not None, "composite SIGTERM handler was not installed"
    getattr(handler, "_run")()  # drive the installed signal handler
    code = await asyncio.wait_for(task, timeout=10)
    assert code == EXIT_OK
    assert serve_wiring.grpc.stops == [2]
    assert serve_wiring.created[0].should_exit is True
    assert serve_wiring.created[1].should_exit is True


async def test_serve_unknown_component_exits_runtime() -> None:
    """IMP-03.17 (belt-and-braces): an unknown `--only` value fails cleanly."""
    assert await serve("bogus") == EXIT_RUNTIME


async def test_serve_only_controller_keeps_own_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--only controller` delegates to `run()` with default signal handling."""
    calls: list[tuple[object, bool]] = []

    async def fake_run(server: object, install_signals: bool = True) -> None:
        calls.append((server, install_signals))

    monkeypatch.setenv("VLLM_IMAGE_TAG", "v0.11.2")
    monkeypatch.setattr(sv, "run", fake_run)
    code = await serve("controller")
    assert code == EXIT_OK
    assert calls[0][1] is True  # the controller owns signals in --only mode
