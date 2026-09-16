"""JobRunner unit tests using the in-memory FakeDocker (no docker/pynvml imports)."""

from __future__ import annotations

import types

import pytest

from vibe.docker_ctl import ContainerExited
from vibe.gpu import GpuProcess, GpuStats
from vibe.jobs import JobInProgress, JobRunner, JobState


async def drain(agen):
    """Collect every event of an async generator into a list."""
    return [event async for event in agen]


def make_runner(settings, registry, docker, get_gpu_stats=None):
    if get_gpu_stats is None:
        return JobRunner(settings, registry, docker=docker)
    return JobRunner(settings, registry, docker=docker, get_gpu_stats=get_gpu_stats)


# -- load --------------------------------------------------------------------


async def test_load_happy_path(settings, registry, fake_docker):
    runner = make_runner(settings, registry, fake_docker)
    events = await drain(runner.load("qwen2.5-coder-7b"))

    states = [event.state for event in events if event.state is not None]
    assert JobState.UNLOADING in states
    assert JobState.LOADING in states
    assert JobState.READY in states
    assert (
        states.index(JobState.UNLOADING)
        < states.index(JobState.LOADING)
        < states.index(JobState.READY)
    )

    final = events[-1]
    assert final.state == JobState.READY
    assert "qwen2.5-coder-7b READY at http://127.0.0.1:8030/v1" in final.message

    # purge ran (nothing found), start ran, readiness was awaited
    assert any("no containers to purge" in event.message for event in events)
    assert len(fake_docker.started) == 1
    spec, container = fake_docker.started[0]
    assert spec.alias == "qwen2.5-coder-7b"
    assert container.name == "vibe-qwen2.5-coder-7b"
    assert len(container.id) >= 12
    assert len(fake_docker.wait_ready_calls) == 1
    wait_spec, wait_timeout = fake_docker.wait_ready_calls[0]
    assert wait_spec.alias == "qwen2.5-coder-7b"
    assert wait_timeout == settings.vibe_load_timeout

    # the job slot is released once the stream finishes
    assert runner.current_job() is None


async def test_load_unknown_alias_raises_keyerror(settings, registry, fake_docker):
    runner = make_runner(settings, registry, fake_docker)
    with pytest.raises(KeyError):
        await anext(runner.load("no-such-alias"))
    assert fake_docker.started == []
    assert runner.current_job() is None


# -- single in-flight guard -----------------------------------------------------


async def test_single_inflight_guard(settings, registry, fake_docker):
    runner = make_runner(settings, registry, fake_docker)
    load_gen = runner.load("qwen2.5-coder-7b")

    first = await anext(load_gen)
    assert first.state == JobState.UNLOADING
    assert runner.current_job() is not None

    # a second job is rejected while the load is still mid-stream
    with pytest.raises(JobInProgress):
        await anext(runner.unload())

    # draining the load releases the slot
    await drain(load_gen)
    assert runner.current_job() is None


# -- unload ----------------------------------------------------------------------


async def test_unload_nothing_to_unload(settings, registry, fake_docker):
    fake_docker.purge_names = []
    runner = make_runner(settings, registry, fake_docker)
    events = await drain(runner.unload())

    assert events[0].state == JobState.UNLOADING
    assert any("nothing to unload" in event.message for event in events)
    # IMP-02.7: the no-op unload ends with a terminal AVAILABLE event so
    # clients can tell "done" from "disconnected" (FR-02.4).
    assert events[-1].state == JobState.AVAILABLE
    assert runner.current_job() is None


async def test_unload_removes_containers(settings, registry, fake_docker):
    fake_docker.purge_names = ["vibe-x"]
    runner = make_runner(settings, registry, fake_docker)
    events = await drain(runner.unload())

    assert events[0].state == JobState.UNLOADING
    assert any(
        "removed 1 container(s): vibe-x" in event.message for event in events
    )
    assert events[-1].state == JobState.AVAILABLE
    assert runner.current_job() is None


async def test_load_container_exited_reports_last_logs(settings, registry, fake_docker):
    """ContainerExited surfaces the tail logs and cleans up (FR-00.11,
    IMP-02.8). The generator re-raises after the ERROR event by design."""
    fake_docker.wait_ready_error = ContainerExited(
        "container vibe-x exited before becoming ready",
        last_logs="line-1\nline-2",
    )
    runner = make_runner(settings, registry, fake_docker)
    events = []
    with pytest.raises(ContainerExited):
        async for event in runner.load("qwen2.5-coder-7b"):
            events.append(event)

    assert events[-1].state == JobState.ERROR
    assert "line-1" in events[-1].message
    # Best-effort cleanup ran (keep_failed defaults to False).
    assert len(fake_docker.stopped) == 1
    assert runner.current_job() is None


async def test_load_container_exited_cleanup_failure_keeps_cause(
    settings, registry, fake_docker
):
    """IMP-02.8: a cleanup failure never masks the readiness error."""
    fake_docker.wait_ready_error = ContainerExited(
        "container vibe-x exited before becoming ready"
    )

    def boom(container) -> None:
        raise RuntimeError("docker SDK APIError during stop")

    fake_docker.stop = boom
    runner = make_runner(settings, registry, fake_docker)
    events = []
    with pytest.raises(ContainerExited):
        async for event in runner.load("qwen2.5-coder-7b"):
            events.append(event)

    assert events[-1].state == JobState.ERROR
    assert "vibe-x exited" in events[-1].message


# -- status ------------------------------------------------------------------------


async def test_status_with_managed_container(settings, registry, fake_docker):
    container = types.SimpleNamespace(id="d" * 12, name="vibe-qwen2.5-coder-7b")
    fake_docker.managed = [container]
    fake_docker.info_by_name[container.name] = {
        "name": container.name,
        "id": "d" * 12,
        "status": "running",
        "started_at": "2026-01-01T00:00:00Z",
        "host_port": 8030,
        "image": "vllm/vllm-openai:v0.11.2",
        "alias": "qwen2.5-coder-7b",
    }

    def fake_gpu_stats() -> GpuStats:
        return GpuStats(
            used_mib=2048,
            total_mib=24576,
            utilization_pct=37,
            processes=(GpuProcess(pid=123, name="vllm", used_mib=2048),),
        )

    runner = make_runner(settings, registry, fake_docker, get_gpu_stats=fake_gpu_stats)
    status = await runner.status()

    assert status["active_alias"] == "qwen2.5-coder-7b"
    assert status["state"] == "running"
    assert status["container"] == "vibe-qwen2.5-coder-7b"
    assert status["host_port"] == 8030
    assert status["image"] == "vllm/vllm-openai:v0.11.2"
    assert status["gpu_error"] is None
    assert status["gpu"] == {
        "used_mib": 2048,
        "total_mib": 24576,
        "utilization_pct": 37,
        "processes": [{"pid": 123, "name": "vllm", "used_mib": 2048}],
    }


async def test_status_without_managed_container(settings, registry, fake_docker):
    def fake_gpu_stats() -> GpuStats:
        return GpuStats(used_mib=0, total_mib=24576, utilization_pct=0, processes=())

    runner = make_runner(settings, registry, fake_docker, get_gpu_stats=fake_gpu_stats)
    status = await runner.status()

    assert status["active_alias"] is None
    assert status["state"] is None
    assert status["host_port"] is None
    assert status["gpu"]["used_mib"] == 0
