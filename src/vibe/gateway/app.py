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

Jev mode (improvements-2, doc 03): when `Status.jev` is set for the active
alias, chat-completions requests to it have `response_format` injected from
the `GetActiveJev` schema (cached by sha) and incompatible requests are
rejected with 400 `jev_incompatible_request` before the upstream is touched.

IMP-03.8 (ADR-10): every decision point emits a structured log line via
`logging.getLogger("vibe.gateway")` with `extra={alias, upstream, status,
elapsed_ms, reason}` — decision outcome, 502 + invalidation, controller-down
transitions and 413 rejections (swap lifecycle logs live in `swap.py`).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from vibe import logging as vlog
from vibe.client import ControllerClient
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.jev import apply_jev, check_compatible
from vibe.gateway.models import build_models_payload, state_name
from vibe.gateway.proxy import UpstreamError, stream_proxy
from vibe.gateway.router import Proxy, Retry, Swap, decide
from vibe.gateway.swap import SwapConflict, SwapCoordinator, SwapFailed, SwapTimeout

logger = vlog.get_logger("vibe.gateway")

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


def _too_large() -> JSONResponse:
    """413 for bodies exceeding `BODY_LIMIT` (NFR-03.4, IMP-03.11)."""
    logger.info("request rejected", extra={"status": 413, "reason": "body too large"})
    return _error_response(
        413,
        {
            "message": "request body too large",
            "type": "invalid_request_error",
            "code": None,
        },
    )


def _jev_unavailable(schema_name: str) -> JSONResponse:
    """503 `jev_unavailable` — fail closed (JEV-03.2).

    Proxying without the schema would return free text to a caller that will
    `json.loads` it; a 503 with `Retry-After` follows the gateway's existing
    retry conventions. The `jev_unavailable` log line is emitted where the
    fetch failed (see `_SnapshotCache.get_active_jev_body`).
    """
    return _error_response(
        503,
        {
            "message": (
                f"Jev is active ({schema_name}) but its schema could not be "
                "fetched; refusing to proxy unconstrained. Retry shortly, or "
                "detach with `vibe jev off`."
            ),
            "type": "jev_unavailable",
            "code": "jev_unavailable",
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
        self._controller_ok: bool | None = None  # None = never talked to it
        # JEV-03.2: the active Jev schema body, cached by content sha.
        self._jev_body: tuple[str, dict[str, object]] | None = None

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
            except Exception as exc:
                # ControllerUnavailable or any gRPC error: controller down.
                if self._controller_ok is not False:
                    # IMP-03.8: log controller-down *transitions* only.
                    logger.info(
                        "controller down",
                        extra={"reason": f"{type(exc).__name__}: {exc}"},
                    )
                    self._controller_ok = False
                self._snapshot = None
                return None
            if self._controller_ok is not True:
                logger.info("controller back")
                self._controller_ok = True
            snapshot = _Snapshot(time.monotonic(), status, infos)
            self._snapshot = snapshot
            return snapshot

    async def get_active_jev_body(
        self, info: control_pb2.JevInfo
    ) -> dict[str, object] | None:
        """The active Jev schema body for `info.sha256`, cached by sha (JEV-03.2).

        `GetActiveJev` is fetched once per sha change, under the snapshot
        lock. Returns `None` — fail closed — when the RPC fails, the response
        carries no info, its sha differs from `Status.jev.sha256`, or the
        body is not a JSON object. The body cache is content-addressed, so it
        survives snapshot invalidation and detach/attach cycles unchanged.
        """
        sha = info.sha256
        if self._jev_body is not None and self._jev_body[0] == sha:
            return self._jev_body[1]
        async with self._lock:
            if self._jev_body is not None and self._jev_body[0] == sha:
                return self._jev_body[1]
            try:
                active = await self._client.get_active_jev()
            except Exception as exc:
                # ControllerUnavailable or any gRPC error (JEV-03.8).
                logger.info(
                    "jev unavailable",
                    extra={"reason": f"{type(exc).__name__}: {exc}"},
                )
                return None
            if not active.HasField("info") or active.info.sha256 != sha:
                logger.info(
                    "jev unavailable",
                    extra={"reason": "active schema sha mismatch"},
                )
                return None
            try:
                schema = json.loads(active.schema_json)
            except json.JSONDecodeError:
                logger.info(
                    "jev unavailable",
                    extra={"reason": "active schema body is not JSON"},
                )
                return None
            if not isinstance(schema, dict):
                logger.info(
                    "jev unavailable",
                    extra={"reason": "active schema body is not an object"},
                )
                return None
            self._jev_body = (sha, schema)
            return schema


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
    coordinator = SwapCoordinator(
        ctl, settings.vibe_swap_timeout, settings.vibe_load_timeout
    )

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

    async def proxy_or_502(
        request: Request,
        upstream: str,
        body: bytes,
        alias: str | None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> Response:
        try:
            return await stream_proxy(request, upstream, body, transport, extra_headers)
        except UpstreamError as exc:
            cache.invalidate()  # FR-03.9: re-ask the controller next request
            logger.info(
                "upstream failure",
                extra={
                    "alias": alias or "",
                    "upstream": upstream,
                    "status": 502,
                    "reason": str(exc),
                },
            )
            return _error_response(
                502,
                {
                    "message": str(exc),
                    "type": "upstream_error",
                    "code": "upstream_error",
                },
            )

    def _body_for_head(response: JSONResponse) -> Response:
        """Empty-body twin of `response` for HEAD requests."""
        return Response(status_code=response.status_code, headers=response.headers)

    async def models_endpoint(request: Request) -> Response:
        snapshot = await cache.get()
        if snapshot is None:
            response: JSONResponse = _controller_down()
        else:
            response = JSONResponse(build_models_payload(snapshot.infos, snapshot.status))
        if request.method == "HEAD":
            return _body_for_head(response)
        return response

    async def health_endpoint(request: Request) -> Response:
        try:
            snapshot = await asyncio.wait_for(
                cache.get(), timeout=_HEALTH_REFRESH_TIMEOUT
            )
        except asyncio.TimeoutError:
            snapshot = None
        if snapshot is None:
            # IMP-03.10: the controller's state is unknown while it is down.
            payload: dict[str, object] = {
                "gateway": "ok",
                "controller": "down",
                "active": None,
                "state": None,
                "jev": None,
            }
        else:
            status = snapshot.status
            jev: dict[str, str] | None = None
            if status.HasField("jev"):
                # JEV-03.7: name + sha only; the body is never shipped here.
                jev = {
                    "name": status.jev.schema_name,
                    "sha256": status.jev.sha256,
                }
            payload = {
                "gateway": "ok",
                "controller": "ok",
                "active": status.active_alias or None,
                "state": state_name(status.state),
                "jev": jev,
            }
        response = JSONResponse(payload)
        if request.method == "HEAD":
            return _body_for_head(response)
        return response

    async def post_endpoint(request: Request) -> Response:
        started = time.monotonic()
        # NFR-03.4: buffer the body bounded at BODY_LIMIT so the `model`
        # field can be inspected before routing.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                # IMP-03.11: reject oversized bodies up front, before
                # reading a single chunk.
                if int(content_length) > BODY_LIMIT:
                    return _too_large()
            except ValueError:
                pass  # malformed length: fall through to the streamed check
        parts: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            parts.append(chunk)
            size += len(chunk)
            if size > BODY_LIMIT:
                return _too_large()
        body = b"".join(parts)
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
        # IMP-03.8: structured decision-outcome log.
        if isinstance(decision, Proxy):
            log_extra: dict[str, object] = {
                "alias": model or "",
                "kind": "proxy",
                "upstream": decision.upstream,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
        elif isinstance(decision, Swap):
            log_extra = {
                "alias": decision.alias,
                "kind": "swap",
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
        else:
            log_extra = {
                "alias": model or "",
                "kind": "retry",
                "status": decision.status,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
        logger.info("decision", extra=log_extra)
        if isinstance(decision, Proxy):
            # Jev enforcement (JEV-03.1/03.2/03.3): applies only when the
            # request is proxied to the active alias while it has a Jev
            # schema attached. `/v1/embeddings` passes through untouched
            # (doc 03); a Swap implies a fresh container ⇒ plain mode.
            extra_headers: dict[str, str] | None = None
            jev_active = False
            if (
                request.url.path != "/v1/embeddings"
                and isinstance(data, dict)
                and snapshot.status.HasField("jev")
            ):
                info = snapshot.status.jev
                schema = await cache.get_active_jev_body(info)
                if schema is None:
                    return _jev_unavailable(info.schema_name)
                incompatible = check_compatible(request.url.path, data)
                if incompatible is not None:
                    logger.info(
                        "jev_rejected",
                        extra={
                            "alias": model or "",
                            "reason": incompatible.reason,
                        },
                    )
                    return _error_response(
                        400,
                        {
                            "message": (
                                f"Jev is active ({info.schema_name}); "
                                f"{incompatible.detail}"
                            ),
                            "type": incompatible.type,
                            "code": incompatible.code,
                        },
                    )
                rewrite = apply_jev(data, schema, info.schema_name)
                if rewrite.overrode:
                    logger.info(
                        "jev_override",
                        extra={"alias": model or "", "overrode": rewrite.overrode},
                    )
                    extra_headers = {
                        "X-Vibe-Jev-Overrode": ", ".join(rewrite.overrode)
                    }
                # JEV-03.3: re-serialise the rewritten body. BODY_LIMIT was
                # applied *before* injection (JEV-03.10); the schema (at most
                # 64 KiB) may push a near-limit body over the limit —
                # acceptable, upstream enforces its own.
                body = json.dumps(rewrite.data, separators=(",", ":")).encode()
                logger.info(
                    "jev_applied",
                    extra={
                        "alias": model or "",
                        "schema": info.schema_name,
                        "sha": info.sha256[:12],
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    },
                )
                extra_headers = {
                    **(extra_headers or {}),
                    "X-Vibe-Jev": f"{info.schema_name}; sha={info.sha256[:12]}",
                }
                jev_active = True
            response = await proxy_or_502(
                request, decision.upstream, body, model, extra_headers
            )
            if jev_active and response.status_code == 400:
                # JEV-03.9: the backend rejected the injected request at
                # request time (the probe passed, but the backend can still
                # say no). Relay verbatim, but invalidate the snapshot so a
                # concurrent detach is noticed. No automatic retry without
                # the schema — that would silently violate Jev.
                cache.invalidate()
            return response
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
            request, f"http://127.0.0.1:{port}", body, decision.alias
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
        Route("/health", health_endpoint, methods=["GET", "HEAD"]),
        Route("/v1/models", models_endpoint, methods=["GET", "HEAD"]),
        Route("/v1/chat/completions", post_endpoint, methods=["POST"]),
        Route("/v1/completions", post_endpoint, methods=["POST"]),
        Route("/v1/embeddings", post_endpoint, methods=["POST"]),
        # Catch-all registered last (FR-03.1): unknown /v1/* and everything
        # else becomes an OpenAI-style 404.
        Route("/{path:path}", not_found_endpoint, methods=_METHODS),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
