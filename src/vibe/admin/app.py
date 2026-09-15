"""Zero-JavaScript admin UI app (FR-01.1–01.11, NFR-01.1–01.7).

`build_app(settings, client)` composes the security-header middleware, the
vendored stylesheet, the Jinja2 environment, the in-memory session store and
the six feature routers into a FastAPI app (served on port 8081 by `vibe
serve`). See `.zed/docs/fable-5/features/01-admin-ui.md` for the feature spec.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vibe.admin.routes import chat, dashboard, developer, jobs, logs, models
from vibe.admin.routes import fmt_mib, fmt_ts, fmt_uptime, job_kind_name
from vibe.admin.sessions import SessionStore
from vibe.client import ControllerClient
from vibe.config import GATEWAY_PORT, Settings
from vibe.gateway.models import state_name

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# NFR-01.1: the exact policy carried by every response; scripts disabled,
# styles allowed only from our own static mount.
CSP = "default-src 'self'; script-src 'none'; style-src 'self'"


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


def build_app(settings: Settings, client: ControllerClient) -> FastAPI:
    """Compose the admin ASGI app (FR-01.1).

    `client` is injected by the caller (the `vibe serve` composition root) —
    the admin never constructs its own controller connection.
    """
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters.update(
        ts=fmt_ts,
        uptime=fmt_uptime,
        mib=fmt_mib,
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
    app.state.known_jobs = {}
    app.state.http = httpx.AsyncClient(timeout=1.0)
    app.state.gateway_health_url = f"http://{settings.vibe_bind}:{GATEWAY_PORT}/health"

    app.add_middleware(_SecurityHeaders)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(dashboard.router)
    app.include_router(models.router)
    app.include_router(jobs.router)
    app.include_router(chat.router)
    app.include_router(logs.router)
    app.include_router(developer.router)
    return app
