"""Streaming reverse proxy to the active vLLM container
(FR-03.3/03.4/03.9, NFR-03.5).

The gateway is a byte-for-byte passthrough: path, query, method, body and all
end-to-end headers are forwarded; hop-by-hop headers are stripped in both
directions. Response bodies are relayed unbuffered via `aiter_raw` so SSE
chunks reach the client exactly as vLLM emits them (NFR-03.1/FR-03.4).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping

import httpx
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import StreamingResponse

from vibe import logging as vlog

logger = vlog.get_logger("vibe.gateway")

# Hop-by-hop headers that must not be forwarded (RFC 9110 §7.6.1 plus the
# gateway-specific proxy-auth pair).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class UpstreamError(RuntimeError):
    """The vLLM upstream could not be reached (FR-03.9)."""


def _forwarded_pairs(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep end-to-end headers only.

    Drops the fixed hop-by-hop set (case-insensitive) and any header named by
    a token in the `Connection` header value (RFC 9110 §7.6.1). Duplicate
    header names pass through untouched (IMP-03.12).
    """
    connection_tokens: set[str] = set()
    for name, value in items:
        if name.lower() == "connection":
            connection_tokens = {
                token.strip().lower() for token in value.split(",") if token.strip()
            }
    forwarded: list[tuple[str, str]] = []
    for name, value in items:
        lower = name.lower()
        if lower in _HOP_BY_HOP or lower in connection_tokens:
            continue
        forwarded.append((name, value))
    return forwarded


def _forwarded(headers: Mapping[str, str]) -> list[tuple[str, str]]:
    """`_forwarded_pairs` over a single-valued header mapping."""
    return _forwarded_pairs(headers.items())


async def stream_proxy(
    request: Request, upstream_base: str, body: bytes, http: httpx.AsyncClient
) -> StreamingResponse:
    """Forward `request` to `upstream_base` and relay the streamed response.

    Raises `UpstreamError` when the upstream cannot be reached (FR-03.9).
    Non-2xx HTTP responses are *not* errors — they are relayed verbatim.
    """
    target = upstream_base + request.url.path
    if request.url.query:
        target += f"?{request.url.query}"
    # IMP-03.12: drop the client's Host (httpx sets its own for the upstream
    # target) and preserve it as X-Forwarded-Host for upstreams that care.
    forwarded = [
        (name, value)
        for name, value in _forwarded(request.headers)
        if name.lower() != "host"
    ]
    client_host = request.headers.get("host")
    if client_host:
        forwarded.append(("x-forwarded-host", client_host))
    try:
        upstream_request = http.build_request(
            request.method,
            target,
            headers=forwarded,
            content=body,
        )
        response = await http.send(upstream_request, stream=True)
    except httpx.TransportError as exc:
        # The superclass covers ConnectError, ConnectTimeout, ReadError,
        # WriteError, CloseError, RemoteProtocolError, TimeoutException and
        # ProxyError alike — every transport failure maps to 502 +
        # snapshot invalidation (IMP-03.9, FR-03.9).
        raise UpstreamError(f"upstream {upstream_base} unreachable: {exc}") from exc

    # IMP-03.12: `multi_items()` keeps duplicate header names (e.g. repeated
    # `set-cookie`) — `items()` would merge them into one value.
    raw_headers = [
        (name.encode("latin-1"), value.encode("latin-1"))
        for name, value in _forwarded_pairs(response.headers.multi_items())
    ]

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        except asyncio.CancelledError:
            # The client went away mid-stream (IMP-03.12): surface it in the
            # logs, then re-raise so the `finally` closes the upstream.
            logger.info(
                "relay aborted",
                extra={"upstream": upstream_base, "reason": "client disconnected"},
            )
            raise
        except Exception as exc:
            logger.info(
                "relay failed", extra={"upstream": upstream_base, "reason": str(exc)}
            )
            raise
        finally:
            await response.aclose()

    return StreamingResponse(
        relay(),
        status_code=response.status_code,
        headers=MutableHeaders(raw=raw_headers),
    )
