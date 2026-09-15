"""`vibe serve` composition root (FR-02.9).

Composes the gRPC controller, the OpenAI-compatible gateway, and the
Phase-3 admin UI into one asyncio loop; `--only` runs a single component.
To keep imports one-way (the controller package must not import the CLI),
the composition is done here.
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn
from pydantic import ValidationError

from vibe.admin.app import build_app as build_admin_app
from vibe.client import ControllerClient
from vibe.config import ADMIN_PORT, GATEWAY_PORT, Settings
from vibe.controller.server import build_server, run
from vibe.errors import EXIT_CONFIG, EXIT_OK, EXIT_RUNTIME
from vibe.gateway.app import build_app
from vibe.registry import Registry, RegistryError


def _load_config() -> tuple[Settings, Registry]:
    """Load settings and the registry, exiting with code 2 on any error.

    Mirrors the `_load_config` pattern in `cli.py`, but raises `SystemExit`
    instead of `typer.Exit` so this module stays free of Typer imports.
    """
    try:
        settings = Settings()
    except ValidationError as exc:
        print("config error:", file=sys.stderr)
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            print(f"  {loc}: {err['msg']}" if loc else f"  {err['msg']}", file=sys.stderr)
        raise SystemExit(EXIT_CONFIG) from exc
    try:
        registry = Registry.load(Registry.default_path())
    except RegistryError as exc:
        print("config error:", file=sys.stderr)
        for line in exc.errors:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(EXIT_CONFIG) from exc
    return settings, registry


async def serve(only: str | None) -> int:
    """Start the control plane: controller, gateway, and admin (FR-02.9)."""
    if only not in (None, "controller", "gateway", "admin"):
        print(
            f"unknown component '{only}' (expected: controller, gateway, or admin)",
            file=sys.stderr,
        )
        return EXIT_RUNTIME

    settings, registry = _load_config()

    if only == "controller":
        print(f"starting controller on {settings.controller_target}")
        server = await build_server(settings, registry)
        await run(server)
        return EXIT_OK

    if only == "admin":
        admin_client = ControllerClient(settings)
        admin_app = build_admin_app(settings, admin_client)
        print(f"starting admin UI on http://{settings.vibe_bind}:{ADMIN_PORT}")
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
        print(f"starting gateway on http://{settings.vibe_bind}:{GATEWAY_PORT}")
        uvc = uvicorn.Server(
            uvicorn.Config(app, host=settings.vibe_bind, port=GATEWAY_PORT, log_level="info")
        )
        await uvc.serve()
        return EXIT_OK

    # Default (only is None): run controller, gateway, and admin together in
    # one loop. uvicorn installs SIGINT handlers in the main thread, which
    # raise KeyboardInterrupt into this loop and abort wait_for_termination();
    # the finally block stops both HTTP servers before the gRPC server, and
    # run_serve converts any stray KeyboardInterrupt to EXIT_OK.
    admin_client = ControllerClient(settings)
    admin_app = build_admin_app(settings, admin_client)
    grpc_srv = await build_server(settings, registry)
    await grpc_srv.start()
    print(f"vibe controller listening on {settings.controller_target}")
    print(f"starting gateway on http://{settings.vibe_bind}:{GATEWAY_PORT}")
    print(f"starting admin UI on http://{settings.vibe_bind}:{ADMIN_PORT}")
    uvc = uvicorn.Server(
        uvicorn.Config(app, host=settings.vibe_bind, port=GATEWAY_PORT, log_level="info")
    )
    admin_uvc = uvicorn.Server(
        uvicorn.Config(
            admin_app, host=settings.vibe_bind, port=ADMIN_PORT, log_level="info"
        )
    )
    uvc_task = asyncio.create_task(uvc.serve())
    admin_task = asyncio.create_task(admin_uvc.serve())
    try:
        await grpc_srv.wait_for_termination()
    finally:
        for server, task in ((uvc, uvc_task), (admin_uvc, admin_task)):
            server.should_exit = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await grpc_srv.stop(grace=2)
    return EXIT_OK


def run_serve(only: str | None) -> int:
    """Sync entry point for the Typer command; Ctrl-C exits cleanly."""
    try:
        return asyncio.run(serve(only))
    except KeyboardInterrupt:
        return EXIT_OK
