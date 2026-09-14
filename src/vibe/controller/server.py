"""gRPC server composition for the vibe controller (Phase 1).

`build_server` wires the servicer to the real Phase-0 components (or to test
fakes), binds `${VIBE_BIND}:50051` insecure (NFR-02.4) and registers the
standard gRPC health service (FR-02.8). `run`/`main` are the composition
root used by `vibe serve` (FR-02.9).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from vibe.config import CONTROLLER_PORT, Settings
from vibe.controller.logbuffer import RingBuffer
from vibe.controller.servicer import ModelControllerServicer
from vibe.docker_ctl import DockerCtl, DockerError
from vibe.gen.vibe.v1 import control_pb2_grpc
from vibe.gpu import GpuStats, get_gpu_stats
from vibe.jobs import JobRunner
from vibe.registry import Registry

SERVICE_NAME = "vibe.v1.ModelController"


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
    server = grpc.aio.server()
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


async def run(server: grpc.aio.Server) -> None:
    """Start the server and block until termination (Ctrl+C)."""
    await server.start()
    address = getattr(server, "bound_address", None) or "(unknown address)"
    print(f"vibe controller listening on {address}")
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(grace=2)
        print("controller stopped")


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
