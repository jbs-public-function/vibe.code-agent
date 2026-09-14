"""In-process grpc.aio tests for the ModelController servicer.

The server is built with duck-typed fakes (`FakeJobRunner`, `FakeDocker`)
bound to an ephemeral port — no docker or pynvml imports anywhere.
"""

from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime
from typing import Any

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from vibe.config import Settings
from vibe.controller import RingBuffer, build_server
from vibe.docker_ctl import DockerUnavailableError
from vibe.gen.vibe.v1 import control_pb2, control_pb2_grpc
from vibe.gpu import GpuProcess, GpuStats
from vibe.jobs import Job, JobEvent, JobInProgress, JobState

FAKE_GPU_STATS = GpuStats(
    used_mib=2048,
    total_mib=24576,
    utilization_pct=37,
    processes=(GpuProcess(pid=123, name="vllm", used_mib=2048),),
)


# -- helpers -----------------------------------------------------------------


def free_port() -> int:
    """Pick a free loopback port (bind-close race accepted for tests)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# The sandbox sets http_proxy env vars; without this option grpc would try to
# route even 127.0.0.1 connections through the proxy. Tests never need it.
_NO_PROXY_CHANNEL_OPTIONS = (("grpc.enable_http_proxy", 0),)


async def controller_stub(port: int):
    """Create a channel + `ModelControllerStub` for a bound controller."""
    channel = grpc.aio.insecure_channel(
        f"127.0.0.1:{port}", options=_NO_PROXY_CHANNEL_OPTIONS
    )
    return channel, control_pb2_grpc.ModelControllerStub(channel)


async def start_server(
    settings: Settings,
    registry,
    runner,
    docker,
    buffer,
    gpu_fn=None,
):
    """Build and start a controller on an ephemeral port."""
    port = free_port()
    server = await build_server(
        settings,
        registry,
        runner=runner,
        docker=docker,
        gpu_fn=gpu_fn if gpu_fn is not None else (lambda: FAKE_GPU_STATS),
        buffer=buffer,
        port=port,
    )
    await server.start()
    return server, port


# -- fakes --------------------------------------------------------------------


class FakeJobRunner:
    """Duck-typed `JobRunner` with scriptable load/unload streams.

    `hold`/`hold_at` block the load generator mid-stream (before yielding
    `load_events[hold_at]`) so tests can observe an in-flight job.
    """

    def __init__(
        self,
        load_events: list | None = None,
        load_exc: Exception | None = None,
        unload_exc: Exception | None = None,
        hold: asyncio.Event | None = None,
        hold_at: int = 0,
    ) -> None:
        self.load_events = list(load_events or [])
        self.load_exc = load_exc
        self.unload_exc = unload_exc
        self.hold = hold
        self.hold_at = hold_at
        self.load_calls: list = []
        self.unload_calls = 0
        self.load_consumed = False
        self.unload_consumed = False
        self._job: Job | None = None

    def current_job(self) -> Job | None:
        return self._job

    def _make_job(self, kind: str, alias: str | None) -> Job:
        return Job(
            id="job-1" if kind == "load" else "job-2",
            kind=kind,
            alias=alias,
            state=JobState.LOADING if kind == "load" else JobState.UNLOADING,
            started_at=datetime.now(UTC),
        )

    async def load(self, alias, force=False, keep_failed=False, timeout_s=None):
        # Body runs on first anext, mirroring the real runner.
        self.load_calls.append((alias, force, keep_failed, timeout_s))
        if self._job is not None:
            raise JobInProgress(f"job {self._job.id} is already running")
        self._job = self._make_job("load", alias)
        try:
            for index, event in enumerate(self.load_events):
                if self.hold is not None and index == self.hold_at:
                    await self.hold.wait()
                yield event
            if self.load_exc is not None:
                raise self.load_exc
        finally:
            self.load_consumed = True
            self._job = None

    async def unload(self):
        self.unload_calls += 1
        if self._job is not None:
            raise JobInProgress(f"job {self._job.id} is already running")
        self._job = self._make_job("unload", None)
        try:
            if self.unload_exc is not None:
                raise self.unload_exc
            yield JobEvent(
                datetime.now(UTC),
                "info",
                "all models unloaded",
                state=JobState.AVAILABLE,
            )
        finally:
            self.unload_consumed = True
            self._job = None

    async def status(self) -> dict:
        return {
            "active_alias": None,
            "state": None,
            "container": None,
            "started_at": None,
            "host_port": None,
            "image": None,
            "gpu": None,
            "gpu_error": None,
        }


class FakeDocker:
    """Duck-typed `DockerCtl` (mirrors the shape in tests/conftest.py)."""

    def __init__(self, pingable: bool = True) -> None:
        self.managed: list = []
        self.info_by_name: dict = {}
        self.log_lines: list = []
        self.pingable = pingable

    @property
    def client(self):
        owner = self

        class _Client:
            def ping(self) -> None:
                if not owner.pingable:
                    raise DockerUnavailableError(
                        "cannot reach Docker daemon — is docker running?"
                    )

        return _Client()

    def find_managed(self) -> list:
        return list(self.managed)

    def container_info(self, container) -> dict:
        default = {
            "name": container.name,
            "id": getattr(container, "id", "c" * 12),
            "status": "running",
            "started_at": "2026-01-01T00:00:00Z",
            "host_port": 8000,
            "image": "vllm/vllm-openai:v0.11.2",
            "alias": None,
        }
        return dict(self.info_by_name.get(container.name, default))

    def logs(self, container, tail=None, follow=False):
        return iter(self.log_lines)


def job_events(*states) -> list[JobEvent]:
    """Build one info JobEvent per JobState."""
    return [
        JobEvent(datetime.now(UTC), "info", f"step {state.value}", state=state)
        for state in states
    ]


# -- tests ---------------------------------------------------------------------


async def test_list_models(registry, tmp_path):
    settings = Settings(vllm_image_tag="v0.11.2", vibe_data_dir=str(tmp_path))
    server, port = await start_server(
        settings, registry, FakeJobRunner(), FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        response = await stub.ListModels(control_pb2.ListModelsRequest())
        assert [model.alias for model in response.models] == [
            "qwen2.5-coder-7b",
            "llama-3.1-8b",
            "deepseek-r1-14b",
        ]
        qwen = response.models[0]
        assert qwen.display_name == "Qwen2.5 Coder 7B Instruct (AWQ)"
        assert qwen.hf_repo == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
        assert qwen.quantization == "awq"
        assert qwen.max_model_len == 32768
        assert qwen.host_port == 8030
        assert "code" in qwen.tags
        # No container and no job: everything is AVAILABLE, nothing cached.
        assert all(
            model.state == control_pb2.AVAILABLE for model in response.models
        )
        assert all(model.cached_locally is False for model in response.models)
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_get_status_empty(settings, registry):
    server, port = await start_server(
        settings, registry, FakeJobRunner(), FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        status = await stub.GetStatus(control_pb2.GetStatusRequest())
        assert status.active_alias == ""
        assert status.state == control_pb2.MODEL_STATE_UNSPECIFIED
        assert status.container_id == ""
        assert status.host_port == 0
        assert status.started_at_unix == 0
        assert not status.HasField("current_job")
        assert status.last_error == ""
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_load_happy_path(settings, registry):
    events = job_events(JobState.UNLOADING, JobState.LOADING, JobState.READY)
    runner = FakeJobRunner(load_events=events)
    buffer = RingBuffer()
    server, port = await start_server(
        settings, registry, runner, FakeDocker(), buffer
    )
    channel, stub = await controller_stub(port)
    try:
        received = [
            event
            async for event in stub.LoadModel(
                control_pb2.LoadModelRequest(
                    alias="qwen2.5-coder-7b",
                    force=True,
                    keep_failed=True,
                    timeout_s=30,
                )
            )
        ]
        assert [event.state for event in received] == [
            control_pb2.UNLOADING,
            control_pb2.LOADING,
            control_pb2.READY,
        ]
        assert all(event.job_id == "job-1" for event in received)
        assert received[-1].terminal is True
        assert received[-1].state == control_pb2.READY
        assert received[-1].ts_unix_ms == int(
            events[-1].ts.timestamp() * 1000
        )
        # Request options were forwarded to the runner verbatim.
        assert runner.load_calls == [
            ("qwen2.5-coder-7b", True, True, 30)
        ]
        # Every event also landed in the ring buffer.
        assert len(buffer) == 3
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_load_unknown_alias_not_found(settings, registry):
    server, port = await start_server(
        settings, registry, FakeJobRunner(), FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        call = stub.LoadModel(control_pb2.LoadModelRequest(alias="nope"))
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await call.read()
        assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND
        assert "unknown alias 'nope'" in (excinfo.value.details() or "")
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_concurrent_load_fails_precondition(settings, registry):
    hold = asyncio.Event()
    events = job_events(JobState.UNLOADING, JobState.LOADING, JobState.READY)
    runner = FakeJobRunner(load_events=events, hold=hold, hold_at=1)
    server, port = await start_server(
        settings, registry, runner, FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        call1 = stub.LoadModel(
            control_pb2.LoadModelRequest(alias="qwen2.5-coder-7b")
        )
        # grpc.aio forbids mixing read() and async-for on one call: keep a
        # single iterator for both the first read and the later drain.
        stream1 = call1.__aiter__()
        first = await anext(stream1)
        assert first.state == control_pb2.UNLOADING
        assert runner.current_job() is not None

        call2 = stub.LoadModel(
            control_pb2.LoadModelRequest(alias="llama-3.1-8b")
        )
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await call2.read()
        assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "job-1" in (excinfo.value.details() or "")

        # Release the first load and let it finish cleanly.
        hold.set()
        remaining = [event async for event in stream1]
        assert remaining[-1].state == control_pb2.READY
        assert runner.current_job() is None
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_client_cancel_does_not_abort_job(settings, registry):
    """FR-02.11: cancelling LoadModel mid-stream keeps the job running."""
    hold = asyncio.Event()
    events = job_events(JobState.UNLOADING, JobState.LOADING, JobState.READY)
    runner = FakeJobRunner(load_events=events, hold=hold, hold_at=1)
    buffer = RingBuffer()
    server, port = await start_server(
        settings, registry, runner, FakeDocker(), buffer
    )
    channel, stub = await controller_stub(port)
    try:
        call = stub.LoadModel(
            control_pb2.LoadModelRequest(alias="qwen2.5-coder-7b")
        )
        first = await call.read()
        assert first.state == control_pb2.UNLOADING

        call.cancel()
        await asyncio.sleep(0.1)  # let the server process the cancellation

        # The job is still in flight after the client went away.
        assert runner.current_job() is not None
        assert not runner.load_consumed

        hold.set()
        for _ in range(200):
            if runner.load_consumed:
                break
            await asyncio.sleep(0.01)
        assert runner.load_consumed
        assert runner.current_job() is None

        # The terminal event landed in the ring buffer despite the cancel.
        buffered = buffer.recent(20)
        assert any(
            event.state is JobState.READY for event in buffered
        )
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_unload_idempotent_single_terminal_event(settings, registry):
    runner = FakeJobRunner()
    server, port = await start_server(
        settings, registry, runner, FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        received = [
            event
            async for event in stub.UnloadModel(control_pb2.UnloadModelRequest())
        ]
        assert len(received) == 1
        assert received[0].terminal is True
        assert received[0].state == control_pb2.AVAILABLE
        assert received[0].job_id == "job-2"
        assert runner.unload_consumed
        assert runner.current_job() is None
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_get_gpu_stats(settings, registry):
    server, port = await start_server(
        settings, registry, FakeJobRunner(), FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        stats = await stub.GetGpuStats(control_pb2.GetGpuStatsRequest())
        assert stats.used_mib == 2048
        assert stats.total_mib == 24576
        assert stats.utilization_pct == 37
        assert len(stats.processes) == 1
        assert stats.processes[0].pid == 123
        assert stats.processes[0].name == "vllm"
        assert stats.processes[0].used_mib == 2048
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_health_serving_and_not_serving(settings, registry):
    docker = FakeDocker(pingable=True)
    server, port = await start_server(
        settings, registry, FakeJobRunner(), docker, RingBuffer()
    )
    channel = grpc.aio.insecure_channel(
        f"127.0.0.1:{port}", options=_NO_PROXY_CHANNEL_OPTIONS
    )
    # grpcio-health-checking ships no type stubs, so pyright can't see the
    # generated `Check` method on the stub class.
    health_stub: Any = health_pb2_grpc.HealthStub(channel)
    try:
        response = await health_stub.Check(
            health_pb2.HealthCheckRequest(service="vibe.v1.ModelController")
        )
        assert response.status == health_pb2.HealthCheckResponse.SERVING

        overall = await health_stub.Check(
            health_pb2.HealthCheckRequest(service="")
        )
        assert overall.status == health_pb2.HealthCheckResponse.SERVING

        # Docker daemon unreachable -> NOT_SERVING (FR-02.8).
        docker.pingable = False
        response = await health_stub.Check(
            health_pb2.HealthCheckRequest(service="vibe.v1.ModelController")
        )
        assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_stream_logs_replays_buffer(settings, registry):
    """FR-02.5: no container -> ring-buffered job events, oldest first."""
    buffer = RingBuffer()
    first = JobEvent(
        datetime(2026, 1, 1, tzinfo=UTC),
        "info",
        "first message",
        state=JobState.UNLOADING,
    )
    second = JobEvent(
        datetime(2026, 1, 2, tzinfo=UTC),
        "info",
        "second message",
        state=JobState.READY,
    )
    buffer.append(first)
    buffer.append(second)
    server, port = await start_server(
        settings, registry, FakeJobRunner(), FakeDocker(), buffer
    )
    channel, stub = await controller_stub(port)
    try:
        lines = [
            line
            async for line in stub.StreamLogs(
                control_pb2.StreamLogsRequest(tail=0, follow=True)
            )
        ]
        assert [line.line for line in lines] == [
            "[info] first message",
            "[info] second message",
        ]
        assert lines[0].ts_unix_ms == int(first.ts.timestamp() * 1000)
        assert lines[1].ts_unix_ms == int(second.ts.timestamp() * 1000)
    finally:
        await channel.close()
        await server.stop(grace=0)
