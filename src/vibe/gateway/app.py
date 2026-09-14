"""OpenAI-compatible gateway app (FR-03.1–03.10, NFR-03.2/03.4).

`build_app` composes the router decision table, the streaming proxy and the
swap coordinator into a Starlette app:

- `GET  /v1/models`             — every registry alias, OpenAI-style (FR-03.2)
- `POST /v1/chat/completions`   — proxied to the active vLLM, or an
  `POST /v1/completions`          OpenAI-style error / swap-on-request
  `POST /v1/embeddings`
- `GET  /health`                — liveness + controller status (FR-03.10)
- anything else                 — OpenAI-style 404 (FR-03.1)

The gateway holds no model state of its own: routing comes from a
`ListModels`/`GetStatus` snapshot cached for at most `SNAPSHOT_TTL` seconds
and invalidated on upstream 502s (NFR-03.2, FR-03.9).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from vibe.client import ControllerClient
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import build_models_payload, state_name
from vibe.gateway.proxy import UpstreamError, stream_proxy
from vibe.gateway.router import Proxy, Retry, Swap, decide
from vibe.gateway.swap import SwapConflict, SwapCoordinator, SwapFailed, SwapTimeout

# NFR-03.4: request bodies up to 32 MiB. Module-level so tests can shrink it.
BODY_LIMIT = 32 * 1024 * 1024

# NFR-03.2: routing snapshot is never older than this.
SNAPSHOT_TTL = 5.0

# The controller is never waited on longer than this for /health (FR-03.10).
_HEALTH_REFRESH_TIMEOUT = 2.0

_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"]


def _error_response(
    status_code: int, error: dict[str, str | None], retry_after: int | None = None
) -> JSONResponse:
    """OpenAI-style `{"error": {...}}` body with an optional Retry-After."""
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse({"error": error}, status_code=status_code, headers=headers)


def _controller_down() -> JSONResponse:
    return _error_response(
        503,
        {
            "message": "controller unreachable",
            "type": "controller_unavailable",
            "code": "controller_unavailable",
        },
        retry_after=5,
    )


@dataclass
class _Snapshot:
    """A routing snapshot: controller status + registry, with its fetch time."""

    ts: float  # time.monotonic() at fetch
    status: control_pb2.Status
    infos: list[control_pb2.ModelInfo]


class _SnapshotCache:
    """TTL'd routing snapshot behind an async lock (NFR-03.2).

    A failed refresh (controller down / any gRPC error) is reported as `None`
    and drops the previous snapshot, so callers see "controller down" rather
    than stale routing data.
    """

    def __init__(self, client: ControllerClient, ttl: float) -> None:
        self._client = client
        self._ttl = ttl
        self._lock = asyncio.Lock()
        self._snapshot: _Snapshot | None = None

    def invalidate(self) -> None:
        """Drop the snapshot so the next request re-asks the controller."""
        self._snapshot = None

    async def get(self) -> _Snapshot | None:
        """Return a fresh snapshot, or `None` when the controller is down."""
        snapshot = self._snapshot
        if snapshot is not None and time.monotonic() - snapshot.ts < self._ttl:
            return snapshot
        async with self._lock:
            snapshot = self._snapshot
            if snapshot is not None and time.monotonic() - snapshot.ts < self._ttl:
                return snapshot
            try:
                infos = await self._client.list_models()
                status = await self._client.status()
            except Exception:
                # ControllerUnavailable or any gRPC error: controller down.
                self._snapshot = None
                return None
            snapshot = _Snapshot(time.monotonic(), status, infos)
            self._snapshot = snapshot
            return snapshot


def build_app(
    settings: Settings,
    client: ControllerClient | None = None,
    http: httpx.AsyncClient | None = None,
) -> Starlette:
    """Compose the gateway ASGI app (FR-03.1)."""
    ctl = client if client is not None else ControllerClient(settings)
    transport = http if http is not None else httpx.AsyncClient(
        # Streaming generations can pause between chunks (long thinking
        # steps, slow first token), so disable the read timeout; connect
        # stays bounded so dead containers fail fast (FR-03.9).
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
    )
    cache = _SnapshotCache(ctl, SNAPSHOT_TTL)
    coordinator = SwapCoordinator(ctl, settings.vibe_swap_timeout)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for closer in (transport.aclose, ctl.close):
                try:
                    await closer()
                except Exception:
                    pass  # shutdown must not raise

    async def proxy_or_502(request: Request, upstream: str, body: bytes) -> Response:
        try:
            return await stream_proxy(request, upstream, body, transport)
        except UpstreamError as exc:
            cache.invalidate()  # FR-03.9: re-ask the controller next request
            return _error_response(
                502,
                {
                    "message": str(exc),
                    "type": "upstream_error",
                    "code": "upstream_error",
                },
            )

    async def models_endpoint(request: Request) -> JSONResponse:
        snapshot = await cache.get()
        if snapshot is None:
            return _controller_down()
        return JSONResponse(build_models_payload(snapshot.infos, snapshot.status))

    async def health_endpoint(request: Request) -> JSONResponse:
        try:
            snapshot = await asyncio.wait_for(
                cache.get(), timeout=_HEALTH_REFRESH_TIMEOUT
            )
        except asyncio.TimeoutError:
            snapshot = None
        if snapshot is None:
            return JSONResponse(
                {
                    "gateway": "ok",
                    "controller": "down",
                    "active": None,
                    "state": "AVAILABLE",
                }
            )
        status = snapshot.status
        return JSONResponse(
            {
                "gateway": "ok",
                "controller": "ok",
                "active": status.active_alias or None,
                "state": state_name(status.state),
            }
        )

    async def post_endpoint(request: Request) -> Response:
        # NFR-03.4: buffer the body bounded at BODY_LIMIT so the `model`
        # field can be inspected before routing.
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > BODY_LIMIT:
                return _error_response(
                    413,
                    {
                        "message": "request body too large",
                        "type": "invalid_request_error",
                        "code": None,
                    },
                )
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return _error_response(
                400,
                {
                    "message": "request body must be valid JSON",
                    "type": "invalid_request_error",
                    "code": None,
                },
            )
        model = data.get("model") if isinstance(data, dict) else None
        snapshot = await cache.get()
        if snapshot is None:
            return _controller_down()
        aliases = [info.alias for info in snapshot.infos]
        decision = decide(
            model,
            aliases,
            snapshot.status,
            settings.vibe_swap_on_request,
            coordinator.in_flight_alias,
        )
        if isinstance(decision, Proxy):
            return await proxy_or_502(request, decision.upstream, body)
        if isinstance(decision, Retry):
            return _error_response(
                decision.status, decision.error, decision.after or None
            )
        # Swap: load the model first, then proxy.
        try:
            port = await coordinator.ensure_ready(decision.alias)
        except SwapConflict as exc:
            return _error_response(
                503,
                {
                    "message": str(exc),
                    "type": "swap_in_progress",
                    "code": "swap_in_progress",
                },
                retry_after=30,
            )
        except SwapTimeout as exc:
            return _error_response(
                504,
                {
                    "message": str(exc),
                    "type": "swap_timeout",
                    "code": "swap_timeout",
                },
            )
        except SwapFailed as exc:
            return _error_response(
                503,
                {
                    "message": str(exc),
                    "type": "model_load_failed",
                    "code": "model_load_failed",
                },
                retry_after=30,
            )
        # The active model just changed: drop the stale routing snapshot.
        cache.invalidate()
        return await proxy_or_502(
            request, f"http://127.0.0.1:{port}", body
        )

    async def not_found_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": f"Unknown request URL: {request.url.path}.",
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
            status_code=404,
        )

    routes = [
        Route("/health", health_endpoint, methods=["GET"]),
        Route("/v1/models", models_endpoint, methods=["GET"]),
        Route("/v1/chat/completions", post_endpoint, methods=["POST"]),
        Route("/v1/completions", post_endpoint, methods=["POST"]),
        Route("/v1/embeddings", post_endpoint, methods=["POST"]),
        # Catch-all registered last (FR-03.1): unknown /v1/* and everything
        # else becomes an OpenAI-style 404.
        Route("/{path:path}", not_found_endpoint, methods=_METHODS),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
