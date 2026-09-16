"""In-process grpc.aio tests for the ModelController servicer.

The server is built with duck-typed fakes (`FakeJobRunner`, `FakeDocker`)
bound to an ephemeral port — no docker or pynvml imports anywhere.
"""

from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from vibe.config import Settings
from vibe.controller import RingBuffer, build_server
from vibe.controller import servicer as servicer_mod
from vibe.controller.server import run as controller_run
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
    """FR-02.11: cancelling LoadModel mid-stream keeps the job running,
    and GetStatus reports LOADING meanwhile (IMP-02.1/02.2)."""
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

        # GetStatus reflects the in-flight job (IMP-02.2).
        status = await stub.GetStatus(control_pb2.GetStatusRequest())
        assert status.state == control_pb2.LOADING
        assert status.HasField("current_job")
        assert status.current_job.kind == control_pb2.LOAD
        assert status.current_job.alias == "qwen2.5-coder-7b"
        assert status.current_job.terminal is False

        call.cancel()
        await asyncio.sleep(0.1)  # let the server process the cancellation

        # The job is still in flight after the client went away, and
        # GetStatus still shows LOADING.
        assert runner.current_job() is not None
        assert not runner.load_consumed
        status = await stub.GetStatus(control_pb2.GetStatusRequest())
        assert status.state == control_pb2.LOADING

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


async def test_keyerror_after_first_event_is_terminal_error(settings, registry):
    """IMP-02.5: a KeyError after events started flowing is an escaped
    exception (terminal ERROR event), not a NOT_FOUND startup error."""
    events = job_events(JobState.UNLOADING, JobState.LOADING)
    runner = FakeJobRunner(load_events=events, load_exc=KeyError("vanished"))
    server, port = await start_server(
        settings, registry, runner, FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        received = [
            event
            async for event in stub.LoadModel(
                control_pb2.LoadModelRequest(alias="qwen2.5-coder-7b")
            )
        ]
        assert [event.state for event in received] == [
            control_pb2.UNLOADING,
            control_pb2.LOADING,
            control_pb2.ERROR,
        ]
        assert received[-1].terminal is True
        assert "unexpected error" in received[-1].message
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_heartbeat_debug_events_during_hold(settings, registry, monkeypatch):
    """NFR-02.5: a held job stream emits debug heartbeats every
    HEARTBEAT_SECONDS."""
    monkeypatch.setattr(servicer_mod, "HEARTBEAT_SECONDS", 0.05)
    hold = asyncio.Event()
    events = job_events(JobState.UNLOADING, JobState.LOADING, JobState.READY)
    runner = FakeJobRunner(load_events=events, hold=hold, hold_at=1)
    server, port = await start_server(
        settings, registry, runner, FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        stream = stub.LoadModel(
            control_pb2.LoadModelRequest(alias="qwen2.5-coder-7b")
        ).__aiter__()
        first = await anext(stream)
        assert first.state == control_pb2.UNLOADING

        heartbeats: list[control_pb2.JobEvent] = []

        async def collect() -> None:
            async for event in stream:
                heartbeats.append(event)

        collector = asyncio.create_task(collect())
        for _ in range(200):
            if any(
                event.level == "debug"
                and event.message == servicer_mod.HEARTBEAT_MESSAGE
                for event in heartbeats
            ):
                break
            await asyncio.sleep(0.01)
        assert any(
            event.level == "debug"
            and event.message == servicer_mod.HEARTBEAT_MESSAGE
            for event in heartbeats
        )

        hold.set()
        await collector
        assert heartbeats[-1].state == control_pb2.READY
    finally:
        await channel.close()
        await server.stop(grace=0)


# -- Chat proxy tests (FR-02.6) ----------------------------------------------


class FakeChatResponse:
    """Scriptable httpx response for the Chat proxy's SSE loop."""

    def __init__(
        self,
        status_code: int = 200,
        lines: list[str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._lines = list(lines or [])
        self._body = body
        self.aclose_calls = 0

    async def __aenter__(self) -> "FakeChatResponse":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        pass

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        self.aclose_calls += 1

    def aiter_lines(self):
        async def _lines():
            for line in self._lines:
                yield line

        return _lines()


class FakeChatClient:
    """Scriptable `httpx.AsyncClient`: one scripted `stream` response."""

    def __init__(self, response: FakeChatResponse) -> None:
        self._response = response

    async def __aenter__(self) -> "FakeChatClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        pass

    def stream(self, method: str, url: str, json=None) -> FakeChatResponse:
        return self._response


def _chat_docker() -> FakeDocker:
    """FakeDocker with one running container for alias qwen2.5-coder-7b."""
    docker = FakeDocker()
    container = SimpleNamespace(name="vibe-qwen2.5-coder-7b", id="c" * 12)
    docker.managed = [container]
    docker.info_by_name[container.name] = {
        "name": container.name,
        "id": "c" * 12,
        "status": "running",
        "started_at": "2026-01-01T00:00:00Z",
        "host_port": 8030,
        "image": "vllm/vllm-openai:v0.11.2",
        "alias": "qwen2.5-coder-7b",
    }
    return docker


def _chat_request() -> control_pb2.ChatRequest:
    return control_pb2.ChatRequest(
        alias="qwen2.5-coder-7b",
        messages=[control_pb2.ChatMessage(role="user", content="hi")],
    )


async def test_chat_sse_parsing(settings, registry, monkeypatch):
    """Chat parses `data:` SSE lines into ChatChunks (FR-02.6)."""
    response = FakeChatResponse(
        lines=[
            ": keep-alive",
            'data: {"choices":[{"delta":{"content":"Hi"},'
            '"finish_reason":null}],"usage":{}}',
            "data: not-json",
            'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":7,"completion_tokens":2}}',
            "data: [DONE]",
        ]
    )
    monkeypatch.setattr(
        servicer_mod.httpx,
        "AsyncClient",
        lambda **kwargs: FakeChatClient(response),
    )
    server, port = await start_server(
        settings, registry, FakeJobRunner(), _chat_docker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        chunks = [chunk async for chunk in stub.Chat(_chat_request())]
        assert [chunk.delta for chunk in chunks] == ["Hi", "!"]
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].prompt_tokens == 7
        assert chunks[-1].completion_tokens == 2
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_chat_no_container_fails_precondition(settings, registry):
    """Chat with no running container aborts FAILED_PRECONDITION (FR-02.6)."""
    server, port = await start_server(
        settings, registry, FakeJobRunner(), FakeDocker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        call = stub.Chat(_chat_request())
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await call.read()
        assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "no model loaded" in (excinfo.value.details() or "")
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_chat_upstream_500_becomes_unavailable(
    settings, registry, monkeypatch
):
    """An upstream vLLM error surfaces as UNAVAILABLE with the status line."""
    response = FakeChatResponse(status_code=500, body=b'{"error": "oom"}')
    monkeypatch.setattr(
        servicer_mod.httpx,
        "AsyncClient",
        lambda **kwargs: FakeChatClient(response),
    )
    server, port = await start_server(
        settings, registry, FakeJobRunner(), _chat_docker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        call = stub.Chat(_chat_request())
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await call.read()
        assert excinfo.value.code() == grpc.StatusCode.UNAVAILABLE
        assert "model server returned 500" in (excinfo.value.details() or "")
    finally:
        await channel.close()
        await server.stop(grace=0)


async def test_chat_cancel_acloses_upstream(settings, registry, monkeypatch):
    """Cancelling the Chat RPC closes the upstream response (FR-02.11)."""
    response = FakeChatResponse(
        lines=[
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{"content":"there"}}]}',
            'data: {"choices":[{"delta":{"content":"more"}}]}',
        ]
    )
    monkeypatch.setattr(
        servicer_mod.httpx,
        "AsyncClient",
        lambda **kwargs: FakeChatClient(response),
    )
    server, port = await start_server(
        settings, registry, FakeJobRunner(), _chat_docker(), RingBuffer()
    )
    channel, stub = await controller_stub(port)
    try:
        call = stub.Chat(_chat_request())
        first = await call.read()
        assert first.delta == "Hi"
        call.cancel()
        for _ in range(200):
            if response.aclose_calls:
                break
            await asyncio.sleep(0.01)
        assert response.aclose_calls == 1
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


async def test_run_returns_on_stop_and_closes_downloads(settings, registry):
    """IMP-02.19/02.14: `run(server, install_signals=False)` blocks until
    the server stops, then closes the download executor."""
    port = free_port()
    runner: Any = FakeJobRunner()
    docker: Any = FakeDocker()
    server = await build_server(
        settings,
        registry,
        runner=runner,
        docker=docker,
        gpu_fn=lambda: FAKE_GPU_STATS,
        buffer=RingBuffer(),
        port=port,
    )
    task = asyncio.create_task(controller_run(server, install_signals=False))
    for _ in range(200):
        if getattr(server, "_state", None) is not None:
            break
        await asyncio.sleep(0.01)
    await server.stop(grace=0)
    await asyncio.wait_for(task, timeout=5)
