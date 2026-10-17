"""GetJobEvents replay tests against an in-process controller (FR-01.4).

The admin job page reads events for jobs it did not start through
`ControllerClient.job_events`, which replays the controller's bounded
per-job history oldest-first (NFR-01.5). `build_server` is exercised over
a real gRPC channel on a free loopback port with a fake docker/gpu, so no
real Docker daemon or NVML is touched (mirrors `test_client.py`).
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vibe.client import ControllerClient
from vibe.config import Settings
from vibe.controller import RingBuffer, build_server
from vibe.controller.servicer import ModelControllerServicer
from vibe.docker_ctl import DockerError
from vibe.gen.vibe.v1 import control_pb2
from vibe.gpu import GpuProcess, GpuStats
from vibe.jobs import JobEvent, JobRunner
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


def _summaries(events: list[control_pb2.JobEvent]) -> list[tuple]:
    """The comparable fields of a JobEvent sequence, in stream order."""
    return [
        (event.job_id, event.level, event.message, event.state, event.terminal)
        for event in events
    ]


async def _replay(ns: SimpleNamespace, job_id: str) -> list[control_pb2.JobEvent]:
    """Replay one job's history through a second, independent client."""
    second = ControllerClient(target=f"127.0.0.1:{ns.port}")
    try:
        return await _drain(second.job_events(job_id))
    finally:
        await second.close()


# -- tests ------------------------------------------------------------------


async def test_job_events_replay_load(controller: SimpleNamespace) -> None:
    """A second client replays a load job's events, oldest first (FR-01.4)."""
    collected = await _drain(controller.client.load("qwen2.5-coder-7b"))
    assert collected
    job_id = collected[0].job_id
    assert job_id
    assert collected[-1].terminal

    replayed = await _replay(controller, job_id)
    assert replayed
    assert _summaries(replayed) == _summaries(collected)
    assert replayed[-1].terminal


async def test_job_events_replay_failed_load(
    controller: SimpleNamespace, fake_docker: Any
) -> None:
    """A load that dies with DockerError still replays its ERROR history."""
    def boom(spec: object) -> None:
        raise DockerError("docker daemon unreachable")

    fake_docker.start = boom
    collected = await _drain(controller.client.load("qwen2.5-coder-7b"))
    assert collected
    job_id = collected[0].job_id
    assert job_id
    assert collected[-1].state == control_pb2.ERROR
    assert collected[-1].terminal

    replayed = await _replay(controller, job_id)
    assert _summaries(replayed) == _summaries(collected)
    assert replayed[-1].terminal


async def test_job_events_unknown_job(controller: SimpleNamespace) -> None:
    """An unknown job id surfaces as KeyError (NOT_FOUND → KeyError)."""
    with pytest.raises(KeyError) as excinfo:
        await _drain(controller.client.job_events("no-such-job"))
    assert "unknown job 'no-such-job'" in str(excinfo.value)


async def test_job_events_replay_unload(controller: SimpleNamespace) -> None:
    """Unload events replay too, including the idempotent no-op path."""
    collected = await _drain(controller.client.unload())
    assert collected
    job_id = collected[0].job_id
    assert job_id

    replayed = await _replay(controller, job_id)
    assert replayed
    assert _summaries(replayed) == _summaries(collected)


async def test_job_event_history_is_bounded(
    settings: Settings, registry: Registry, fake_docker: Any
) -> None:
    """Per-job history evicts the oldest job past JOB_HISTORY_JOBS (NFR-01.5)."""
    servicer = ModelControllerServicer(
        settings,
        registry,
        JobRunner(settings, registry, docker=fake_docker, get_gpu_stats=fake_gpu),
        fake_docker,
        fake_gpu,
        RingBuffer(),
    )

    def event(message: str) -> JobEvent:
        return JobEvent(datetime.now(UTC), "info", message)

    servicer._record_job_event("", event("no job id"))  # dropped
    for index in range(9):
        servicer._record_job_event(f"job-{index}", event(f"message-{index}"))

    assert "job-0" not in servicer._job_events
    assert "job-8" in servicer._job_events
    assert len(servicer._job_events) == 8


async def test_replayed_events_carry_kind_and_alias(controller: SimpleNamespace) -> None:
    """IMP-01.1: every streamed and replayed event identifies its job."""
    collected = await _drain(controller.client.load("qwen2.5-coder-7b"))
    job_id = collected[0].job_id
    assert all(event.kind == control_pb2.LOAD for event in collected)
    assert all(event.alias == "qwen2.5-coder-7b" for event in collected)

    replayed = await _replay(controller, job_id)
    assert replayed
    assert all(event.kind == control_pb2.LOAD for event in replayed)
    assert all(event.alias == "qwen2.5-coder-7b" for event in replayed)

    unload = await _drain(controller.client.unload())
    unload_replayed = await _replay(controller, unload[0].job_id)
    assert unload_replayed
    assert all(event.kind == control_pb2.UNLOAD for event in unload_replayed)
    assert all(event.alias == "" for event in unload_replayed)


async def test_get_job_not_found_after_eviction(
    settings: Settings, registry: Registry, fake_docker: Any
) -> None:
    """IMP-01.2: GetJob serves the bounded history; evicted jobs are
    NOT_FOUND (KeyError)."""
    ns = await _start(settings, registry, fake_docker, fake_gpu)
    try:
        job_ids: list[str] = []
        for _ in range(9):  # JOB_HISTORY_JOBS = 8
            events = await _drain(ns.client.load("qwen2.5-coder-7b"))
            job_ids.append(events[0].job_id)
        assert len(set(job_ids)) == 9

        with pytest.raises(KeyError):
            await ns.client.get_job(job_ids[0])
        latest = await ns.client.get_job(job_ids[-1])
        assert latest.terminal is True
        assert latest.ok is True
    finally:
        await _stop(ns)


async def test_job_events_follow_tails_live_events(
    settings: Settings, registry: Registry, fake_docker: Any
) -> None:
    """IMP-01.7: follow=true replays and then tails until the terminal
    event — no polling required."""
    gate = asyncio.Event()

    async def gated_wait_ready(spec: object, timeout_s: int) -> None:
        await gate.wait()

    fake_docker.wait_ready = gated_wait_ready
    ns = await _start(settings, registry, fake_docker, fake_gpu)
    try:
        load_task = asyncio.create_task(
            _drain(ns.client.load("qwen2.5-coder-7b"))
        )
        current = None
        for _ in range(200):
            current = ns.runner.current_job()
            if current is not None:
                break
            await asyncio.sleep(0.01)
        assert current is not None

        follower = ControllerClient(target=f"127.0.0.1:{ns.port}")
        try:
            stream = follower.job_events(current.id, follow=True)
            first = await anext(stream)
            assert first.job_id == current.id
            assert first.kind == control_pb2.LOAD
            assert first.alias == "qwen2.5-coder-7b"

            gate.set()
            remaining = [event async for event in stream]
            assert remaining
            assert remaining[-1].terminal is True
            assert remaining[-1].state == control_pb2.READY
        finally:
            await follower.close()
        collected = await load_task
        assert collected[-1].state == control_pb2.READY
    finally:
        await _stop(ns)
