"""Zero-JavaScript admin UI app (FR-01.1–01.11, NFR-01.1–01.7).

`build_app(settings, client)` composes the security middleware, the vendored
stylesheet, the Jinja2 environment, the in-memory session store and the six
feature routers into a FastAPI app (served on port 8081 by `vibe serve`).
See `.zed/docs/fable-5/features/01-admin-ui.md` for the feature spec.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vibe.admin.routes import chat, dashboard, developer, jev, jobs, logs, models
from vibe.admin.routes import fmt_thousands, fmt_ts, fmt_uptime, job_kind_name
from vibe.admin.security import CSP, SecurityMiddleware
from vibe.admin.sessions import SessionStore
from vibe.client import ControllerClient
from vibe.config import GATEWAY_PORT, Settings
from vibe.gateway.models import state_name

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


class _SecurityHeaders:
    """Add the CSP and nosniff headers to every response (NFR-01.1).

    Written as plain ASGI middleware rather than `BaseHTTPMiddleware` so the
    chunked chat stream keeps its progressive flush (NFR-01.6).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if "content-security-policy" not in headers:
                    headers["content-security-policy"] = CSP
                if "x-content-type-options" not in headers:
                    headers["x-content-type-options"] = "nosniff"
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _gateway_health_url(bind: str) -> str:
    """The gateway `/health` probe URL for a bind address (IMP-04.24).

    Wildcard binds cannot be dialed verbatim, so they map to loopback; IPv6
    literals are bracketed for the URL authority component.
    """
    host = bind or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{GATEWAY_PORT}/health"


def _template_context(request: Request) -> dict[str, Any]:
    """Per-render Jinja context: root-relative URL builder (IMP-04.26).

    `request.app.url_path_for` yields paths like `/jobs/<id>` regardless of
    the request host, so links stay correct behind a TLS proxy.
    """
    return {
        "path_for": request.app.url_path_for,
        "gateway_port": GATEWAY_PORT,
    }


def build_app(settings: Settings, client: ControllerClient) -> FastAPI:
    """Compose the admin ASGI app (FR-01.1).

    `client` is injected by the caller (the `vibe serve` composition root) —
    the admin never constructs its own controller connection.
    """
    templates = Jinja2Templates(
        directory=str(TEMPLATES_DIR), context_processors=[_template_context]
    )
    templates.env.filters.update(
        ts=fmt_ts,
        uptime=fmt_uptime,
        thousands=fmt_thousands,
        statename=state_name,
        jobkind=job_kind_name,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            tasks = cast(set[asyncio.Task[None]], app.state.background_tasks)
            if tasks:
                for task in list(tasks):
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            for closer in (
                cast(httpx.AsyncClient, app.state.http).aclose,
                client.close,
            ):
                try:
                    await closer()
                except Exception:
                    pass  # shutdown must not raise

    app = FastAPI(lifespan=lifespan)
    app.state.client = client
    app.state.templates = templates
    app.state.sessions = SessionStore()
    app.state.background_tasks = set()
    app.state.http = httpx.AsyncClient(timeout=0.5)  # gateway probe budget (IMP-04.24)
    app.state.gateway_health_url = _gateway_health_url(settings.vibe_bind)

    allowed_hosts = {"127.0.0.1", "localhost", "::1", settings.vibe_bind}
    app.add_middleware(_SecurityHeaders)
    app.add_middleware(SecurityMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(dashboard.router)
    app.include_router(models.router)
    app.include_router(jev.router)
    app.include_router(jobs.router)
    app.include_router(chat.router)
    app.include_router(logs.router)
    app.include_router(developer.router)
    return app
