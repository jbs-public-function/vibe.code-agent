"""gRPC server composition for the vibe controller (Phase 1).

`build_server` wires the servicer to the real Phase-0 components (or to test
fakes), binds `${VIBE_BIND}:50051` insecure (NFR-02.4) and registers the
standard gRPC health service (FR-02.8). `run`/`main` are the composition
root used by `vibe serve` (FR-02.9).
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from vibe.config import CONTROLLER_PORT, Settings
from vibe.controller.jev import JevState
from vibe.controller.logbuffer import RingBuffer
from vibe.controller.servicer import ModelControllerServicer
from vibe.docker_ctl import DockerCtl, DockerError
from vibe.gen.vibe.v1 import control_pb2_grpc
from vibe.gpu import GpuStats, get_gpu_stats
from vibe.jobs import JobRunner
from vibe.logging import get_logger
from vibe.registry import Registry

SERVICE_NAME = "vibe.v1.ModelController"

# JEV-02.8: explicit gRPC message-size bound — schemas are capped at 64 KiB
# by validation, and 256 KiB leaves room for framing/overhead.
MAX_MESSAGE_BYTES = 256 * 1024

_log = get_logger("vibe.controller")


class DockerHealthServicer(health.HealthServicer):
    """Standard health service: SERVING iff Docker is reachable (FR-02.8)."""

    def __init__(self, docker: DockerCtl) -> None:
        super().__init__()
        self._docker = docker

    async def Check(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, request, context
    ):
        """Override: health derives from Docker reachability, per request.

        The base `HealthServicer.Check` is synchronous; grpc.aio dispatches
        coroutine handlers only, so this async override is required.
        """
        if request.service not in ("", SERVICE_NAME):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"unknown service: {request.service}")
            return health_pb2.HealthCheckResponse()
        reachable = await asyncio.to_thread(self._docker_reachable)
        status = (
            health_pb2.HealthCheckResponse.SERVING
            if reachable
            else health_pb2.HealthCheckResponse.NOT_SERVING
        )
        return health_pb2.HealthCheckResponse(status=status)

    def _docker_reachable(self) -> bool:
        try:
            self._docker.client.ping()
        except DockerError:
            # `from_env` failed (DockerUnavailableError) — daemon unreachable.
            return False
        except Exception:  # noqa: BLE001
            # The docker SDK raises its own APIError for a dead daemon, and
            # XR-5 forbids importing the SDK here — treat any failure as
            # unreachable rather than crashing the health check.
            return False
        return True


async def build_server(
    settings: Settings,
    registry: Registry,
    *,
    runner: JobRunner | None = None,
    docker: DockerCtl | None = None,
    gpu_fn: Callable[[], GpuStats] | None = None,
    buffer: RingBuffer | None = None,
    port: int | None = None,
) -> grpc.aio.Server:
    """Compose the controller: servicer + health on `vibe_bind:port`.

    Real components are wired when the corresponding argument is None; tests
    inject duck-typed fakes and an ephemeral port.
    """
    if runner is None:
        runner = JobRunner(settings, registry)
    if docker is None:
        docker = DockerCtl(settings)
    if gpu_fn is None:
        gpu_fn = get_gpu_stats
    if buffer is None:
        buffer = RingBuffer()

    servicer = ModelControllerServicer(
        settings, registry, runner, docker, gpu_fn, buffer
    )
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ]
    )
    # `run` reaches the servicer through this attribute to close the
    # download executor on shutdown (IMP-02.14); like `bound_address` below
    # it is an implementation detail of the composition root.
    server.servicer = servicer  # pyright: ignore[reportAttributeAccessIssue]
    control_pb2_grpc.add_ModelControllerServicer_to_server(servicer, server)
    health_pb2_grpc.add_HealthServicer_to_server(
        DockerHealthServicer(docker), server
    )

    bind_port = CONTROLLER_PORT if port is None else port
    address = f"{settings.vibe_bind}:{bind_port}"
    # add_insecure_port returns the actual bound port (differs when port=0).
    bound = server.add_insecure_port(address)
    server.bound_address = (  # pyright: ignore[reportAttributeAccessIssue]
        f"{settings.vibe_bind}:{bound}"
    )
    return server


async def run(server: grpc.aio.Server, install_signals: bool = True) -> None:
    """Start the server and block until termination (Ctrl+C/SIGTERM).

    With `install_signals` (the default) SIGINT/SIGTERM are handled inside
    the loop: they trigger `server.stop(grace)` so pump tasks and downloads
    drain instead of being hard-cancelled by `asyncio.run`'s
    KeyboardInterrupt (IMP-02.19). The all-in-one composition root passes
    `install_signals=False` and owns the composite signal handler itself
    (IMP-03.1).
    """
    await server.start()
    address = getattr(server, "bound_address", None) or "(unknown address)"
    _log.info("listening", extra={"address": address})
    loop = asyncio.get_running_loop()

    # JEV-02.11: on start, drop Jev state whose container is no longer
    # running (the controller restarted but Docker kept or lost containers).
    await _reconcile_jev(server)

    stop_task: asyncio.Task | None = None

    def _request_stop() -> None:
        """`loop.add_signal_handler` callback: stop gracefully, not hard."""
        nonlocal stop_task
        if stop_task is None:
            stop_task = asyncio.create_task(server.stop(grace=2))

    if install_signals:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):
                # Non-Unix, or not the main thread: fall back to the
                # KeyboardInterrupt path below.
                pass
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        # Only reachable when the signal handlers could not be installed:
        # still stop gracefully rather than hard-cancelling pump tasks.
        await server.stop(grace=2)
    if stop_task is not None:
        await stop_task
    servicer = getattr(server, "servicer", None)
    if servicer is not None:
        servicer.downloads.close()  # IMP-02.14: executor shutdown
    _log.info("stopped")


async def _reconcile_jev(server: grpc.aio.Server) -> None:
    """Keep Jev state only while a managed container for its alias runs.

    Best-effort: an unreachable Docker daemon leaves the state untouched
    (GetStatus reconciles lazily once Docker answers).
    """
    servicer = getattr(server, "servicer", None)
    if servicer is None:
        return
    state: JevState | None = getattr(servicer.runner, "jev_state", None)
    if state is None or state.current() is None:
        return
    docker = getattr(servicer.runner, "docker", None)
    if docker is None:
        return
    try:
        managed = await asyncio.to_thread(docker.find_managed)
    except DockerError:
        return
    running: set[str] = set()
    for container in managed:
        try:
            info = await asyncio.to_thread(docker.container_info, container)
        except DockerError:
            continue
        alias = info.get("alias")
        if alias and "running" in str(info.get("status", "")).lower():
            running.add(str(alias))
    if state.reconcile(running):
        _log.info(
            "dropped stale Jev state",
            extra={"reason": "no running container for its alias"},
        )


def main(settings: Settings, registry: Registry) -> int:
    """Composition root for `vibe serve --only controller` (FR-02.9)."""

    async def _serve() -> None:
        server = await build_server(settings, registry)
        await run(server)

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        return 0
    return 0
