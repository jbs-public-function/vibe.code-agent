"""ControllerClient tests against an in-process controller (FR-02.3, FR-02.10).

`build_server` is exercised over a real gRPC channel on a free loopback port,
with a fake docker/gpu so no real Docker daemon or NVML is touched. The
servicer is driven by the real `JobRunner` from Phase 0, so load/unload event
semantics match production.

NOTE: this file imports the controller package via the Phase 1 contract
(`vibe.controller.build_server`, `vibe.controller.RingBuffer`). If the
controller package has not landed yet, collection fails with an ImportError —
expected while the two agents work in parallel.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vibe.client import ControllerClient, ControllerUnavailable
from vibe.config import Settings
from vibe.controller import RingBuffer, build_server
from vibe.gen.vibe.v1 import control_pb2
from vibe.gpu import GpuProcess, GpuStats
from vibe.jobs import JobEvent, JobInProgress, JobRunner
from vibe.registry import Registry


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
    """Keep gRPC traffic to the in-process controller off any HTTP proxy.

    Dev shells often export `http_proxy`/`all_proxy`; gRPC's C-core honors
    them even for loopback targets unless `no_proxy` matches.
    """
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("grpc_proxy", raising=False)


async def _start(
    settings: Settings,
    registry: Registry,
    docker: Any,  # conftest.FakeDocker duck-types DockerCtl (see test_jobs.py)
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
        server=server,
        client=client,
        runner=runner,
        buffer=buffer,
        port=port,
    )


async def _stop(ns: SimpleNamespace) -> None:
    """Tear down an in-process controller started by `_start`."""
    await ns.client.close()
    await ns.server.stop(0)


@pytest.fixture
async def controller(
    settings: Settings, registry: Registry, fake_docker: Any
) -> AsyncIterator[SimpleNamespace]:
    """A started in-process controller with an idle (empty) fake docker."""
    ns = await _start(settings, registry, fake_docker, fake_gpu)
    try:
        yield ns
    finally:
        await _stop(ns)


@pytest.fixture
async def slow_controller(
    settings: Settings, registry: Registry, fake_docker: Any
) -> AsyncIterator[SimpleNamespace]:
    """Like `controller`, but `wait_ready` blocks until `ns.gate.set()`.

    Gates the first load so a second, concurrent load can be issued while the
    first job is still in flight.
    """
    gate = asyncio.Event()

    async def gated_wait_ready(spec: object, timeout_s: int) -> None:
        await gate.wait()

    fake_docker.wait_ready = gated_wait_ready
    ns = await _start(settings, registry, fake_docker, fake_gpu)
    ns.gate = gate
    try:
        yield ns
    finally:
        await _stop(ns)


# -- tests ------------------------------------------------------------------


async def test_list_models(controller: SimpleNamespace, registry: Registry) -> None:
    """ListModels returns every registry entry with its state."""
    models = await controller.client.list_models()
    assert sorted(m.alias for m in models) == sorted(registry.aliases)
    qwen = next(m for m in models if m.alias == "qwen2.5-coder-7b")
    assert qwen.display_name == "Qwen2.5 Coder 7B Instruct (AWQ)"
    assert qwen.host_port == 8030
    # No container is running, so every model is AVAILABLE.
    assert all(m.state == control_pb2.AVAILABLE for m in models)


async def test_status_idle(controller: SimpleNamespace) -> None:
    """GetStatus reports no active model when nothing is loaded."""
    status = await controller.client.status()
    assert isinstance(status, control_pb2.Status)
    assert status.active_alias == ""
    assert status.host_port == 0
    assert status.state == control_pb2.MODEL_STATE_UNSPECIFIED


async def test_load_happy_path(controller: SimpleNamespace) -> None:
    """LoadModel streams events ending in READY (FR-02.3)."""
    events = await _drain(controller.client.load("qwen2.5-coder-7b"))
    assert events
    states = {event.state for event in events if event.state}
    assert control_pb2.READY in states
    assert any(
        event.state == control_pb2.READY and "READY" in event.message
        for event in events
    )


async def test_load_unknown_alias(controller: SimpleNamespace) -> None:
    """An unknown alias surfaces as KeyError (NOT_FOUND → KeyError)."""
    with pytest.raises(KeyError) as excinfo:
        await _drain(controller.client.load("does-not-exist"))
    assert excinfo.value.args[0] == "does-not-exist"


async def test_concurrent_load_rejected(slow_controller: SimpleNamespace) -> None:
    """A second load while one is running raises JobInProgress (FR-02.3)."""
    first = asyncio.create_task(_drain(slow_controller.client.load("qwen2.5-coder-7b")))
    try:
        # Wait until the first job has registered itself with the runner.
        for _ in range(200):
            if slow_controller.runner.current_job() is not None:
                break
            await asyncio.sleep(0.01)
        assert slow_controller.runner.current_job() is not None

        with pytest.raises(JobInProgress):
            await _drain(slow_controller.client.load("llama-3.1-8b"))
    finally:
        slow_controller.gate.set()
    events = await first
    assert any(event.state == control_pb2.READY for event in events)


async def test_logs_replay_from_buffer(controller: SimpleNamespace) -> None:
    """StreamLogs replays the ring buffer when no container is running (FR-02.5)."""
    controller.buffer.append(
        JobEvent(datetime.now(UTC), "info", "first buffered line")
    )
    controller.buffer.append(
        JobEvent(datetime.now(UTC), "info", "second buffered line")
    )
    lines = await _drain(controller.client.logs(tail=5))
    text = "\n".join(line.line for line in lines)
    assert "first buffered line" in text
    assert "second buffered line" in text


async def test_gpu_mapping(controller: SimpleNamespace) -> None:
    """GetGpuStats maps the fake GPU snapshot 1:1 onto the proto (FR-02.7)."""
    gpu = await controller.client.gpu()
    assert isinstance(gpu, control_pb2.GpuStats)
    assert gpu.used_mib == 1234
    assert gpu.total_mib == 24576
    assert gpu.utilization_pct == 42
    assert len(gpu.processes) == 1
    assert gpu.processes[0].pid == 4321
    assert gpu.processes[0].name == "vllm"
    assert gpu.processes[0].used_mib == 1000


async def test_unload_idempotent(controller: SimpleNamespace) -> None:
    """UnloadModel with nothing loaded streams a no-op event (FR-02.4)."""
    events = await _drain(controller.client.unload())
    assert events
    assert any(event.state == control_pb2.UNLOADING for event in events)
    assert any("nothing to unload" in event.message for event in events)


async def test_controller_unavailable() -> None:
    """A dead port yields ControllerUnavailable with the how-to-start message."""
    target = f"127.0.0.1:{free_port()}"
    client = ControllerClient(target=target)
    try:
        with pytest.raises(ControllerUnavailable) as excinfo:
            await client.list_models()
        message = str(excinfo.value)
        assert "controller unreachable" in message
        assert target in message
        assert "vibe serve --only controller" in message
    finally:
        await client.close()


async def test_default_target_from_settings() -> None:
    """Without an explicit target, the client uses `settings.controller_target`."""
    settings = Settings(vllm_image_tag="v0.11.2", vibe_bind="127.0.0.1")
    client = ControllerClient(settings)
    try:
        assert client.target == "127.0.0.1:50051"
    finally:
        await client.close()
