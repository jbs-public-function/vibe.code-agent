"""`vibe serve` composition root (FR-02.9).

Composes the gRPC controller, the OpenAI-compatible gateway, and the
Phase-3 admin UI into one asyncio loop; `--only` runs a single component.
To keep imports one-way (the controller package must not import the CLI),
the composition is done here.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections.abc import Iterator

import uvicorn

from vibe import logging as vlog
from vibe.admin.app import build_app as build_admin_app
from vibe.client import ControllerClient
from vibe.config import (
    ADMIN_PORT,
    GATEWAY_PORT,
    ConfigError,
    Settings,
    load_config,
)
from vibe.controller.server import build_server, run
from vibe.errors import EXIT_CONFIG, EXIT_OK, EXIT_RUNTIME
from vibe.gateway.app import build_app
from vibe.registry import Registry

logger = vlog.get_logger("vibe.serve")

# How long the composite shutdown waits for each HTTP server to drain its
# in-flight requests before cancelling (IMP-03.3).
_SHUTDOWN_GRACE = 5.0

# The gRPC server is stopped with this grace period, mirroring the
# controller's own signal path in `controller/server.py run()`.
_GRPC_GRACE = 2


class _SignalOwnedServer(uvicorn.Server):
    """`uvicorn.Server` with signal capture disabled (IMP-03.1).

    `Server.serve()` = `capture_signals()` + `_serve()`. The default
    `capture_signals` installs its own SIGINT/SIGTERM handlers and, on exit,
    re-raises the captured signal with the original (default) handler —
    killing the composite process before the gRPC server has been stopped.
    One handler installed by `serve()` below owns the signals instead.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _load_config() -> tuple[Settings, Registry]:
    """Load settings and the registry, exiting with code 2 on any error.

    Mirrors the `_load_config` pattern in `cli.py`, but raises `SystemExit`
    instead of `typer.Exit` so this module stays free of Typer imports.
    """
    try:
        return load_config()
    except ConfigError as exc:
        print("config error:", file=sys.stderr)
        for line in exc.errors:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(EXIT_CONFIG) from exc


async def serve(only: str | None) -> int:
    """Start the control plane: controller, gateway, and admin (FR-02.9)."""
    if only not in (None, "controller", "gateway", "admin"):
        logger.error(
            "unknown component '%s' (expected: controller, gateway, or admin)", only
        )
        return EXIT_RUNTIME

    settings, registry = _load_config()
    # IMP-03.8: the shared structured logger is the first thing configured;
    # `vibe serve` output (including this module's prints) lands on stdout.
    vlog.configure(settings.vibe_log_level)

    if only == "controller":
        logger.info("starting controller on %s", settings.controller_target)
        server = await build_server(settings, registry)
        await run(server)
        return EXIT_OK

    if only == "admin":
        admin_client = ControllerClient(settings)
        admin_app = build_admin_app(settings, admin_client)
        logger.info(
            "starting admin UI on http://%s:%s", settings.vibe_bind, ADMIN_PORT
        )
        uvc = uvicorn.Server(
            uvicorn.Config(
                admin_app, host=settings.vibe_bind, port=ADMIN_PORT, log_level="info"
            )
        )
        await uvc.serve()
        return EXIT_OK

    client = ControllerClient(settings)
    app = build_app(settings, client)

    if only == "gateway":
        logger.info(
            "starting gateway on http://%s:%s", settings.vibe_bind, GATEWAY_PORT
        )
        uvc = uvicorn.Server(
            uvicorn.Config(
                app, host=settings.vibe_bind, port=GATEWAY_PORT, log_level="info"
            )
        )
        await uvc.serve()
        return EXIT_OK

    # Default (only is None): run controller, gateway, and admin together in
    # one loop. IMP-03.1: exactly one SIGINT/SIGTERM handler (installed
    # below) owns the composite process — it stops both HTTP servers via
    # `should_exit` and drains the gRPC server via `stop(grace)`.
    admin_client = ControllerClient(settings)
    admin_app = build_admin_app(settings, admin_client)
    grpc_srv = await build_server(settings, registry)
    await grpc_srv.start()
    logger.info("vibe controller listening on %s", settings.controller_target)
    logger.info("starting gateway on http://%s:%s", settings.vibe_bind, GATEWAY_PORT)
    logger.info("starting admin UI on http://%s:%s", settings.vibe_bind, ADMIN_PORT)
    uvc = _SignalOwnedServer(
        uvicorn.Config(
            app, host=settings.vibe_bind, port=GATEWAY_PORT, log_level="info"
        )
    )
    admin_uvc = _SignalOwnedServer(
        uvicorn.Config(
            admin_app, host=settings.vibe_bind, port=ADMIN_PORT, log_level="info"
        )
    )

    loop = asyncio.get_running_loop()
    stop_task: asyncio.Task[None] | None = None

    def _request_stop() -> None:
        """Signal handler: drain the HTTP servers and stop the controller."""
        nonlocal stop_task
        uvc.should_exit = True
        admin_uvc.should_exit = True
        if stop_task is None:
            stop_task = asyncio.create_task(grpc_srv.stop(grace=_GRPC_GRACE))

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            # Non-Unix, or not the main thread: asyncio.run's
            # KeyboardInterrupt path in `run_serve` still applies.
            pass

    uvc_task = asyncio.create_task(uvc.serve())
    admin_task = asyncio.create_task(admin_uvc.serve())
    term_task = asyncio.create_task(grpc_srv.wait_for_termination())

    # IMP-03.2: the first completed task decides the outcome. A finished
    # HTTP task carrying an exception (bind error, crash) takes the process
    # down with a non-zero exit; a clean finish means a signal arrived.
    done, _pending = await asyncio.wait(
        {uvc_task, admin_task, term_task}, return_when=asyncio.FIRST_COMPLETED
    )

    crashed: BaseException | None = None
    for task in (uvc_task, admin_task):
        if task in done and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                crashed = exc

    if crashed is not None:
        logger.error(
            "component crashed", extra={"reason": str(crashed)}, exc_info=crashed
        )

    # IMP-03.3: drain instead of cancel — set `should_exit`, give each
    # server the grace period to finish in-flight SSE responses, and cancel
    # only on timeout.
    for server, task in ((uvc, uvc_task), (admin_uvc, admin_task)):
        if task.done():
            continue
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=_SHUTDOWN_GRACE)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        except (asyncio.CancelledError, Exception):
            pass

    if stop_task is None:
        stop_task = asyncio.create_task(grpc_srv.stop(grace=_GRPC_GRACE))
    try:
        await stop_task
    except (asyncio.CancelledError, Exception):
        pass
    if not term_task.done():
        try:
            await asyncio.wait_for(term_task, timeout=_SHUTDOWN_GRACE)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            term_task.cancel()

    logger.info("gateway stopped")
    logger.info("admin stopped")
    logger.info("controller stopped")
    return EXIT_RUNTIME if crashed is not None else EXIT_OK


def run_serve(only: str | None) -> int:
    """Sync entry point for the Typer command; Ctrl-C exits cleanly."""
    try:
        return asyncio.run(serve(only))
    except KeyboardInterrupt:
        return EXIT_OK
