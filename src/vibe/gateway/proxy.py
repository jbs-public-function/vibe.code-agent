"""Streaming reverse proxy to the active vLLM container
(FR-03.3/03.4/03.9, NFR-03.5).

The gateway is a byte-for-byte passthrough: path, query, method, body and all
end-to-end headers are forwarded; hop-by-hop headers are stripped in both
directions. Response bodies are relayed unbuffered via `aiter_raw` so SSE
chunks reach the client exactly as vLLM emits them (NFR-03.1/FR-03.4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import httpx
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import StreamingResponse

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


def _forwarded(headers: Mapping[str, str]) -> list[tuple[str, str]]:
    """Keep end-to-end headers only.

    Drops the fixed hop-by-hop set (case-insensitive) and any header named by
    a token in the `Connection` header value (RFC 9110 §7.6.1).
    """
    connection = headers.get("connection", "")
    tokens = {token.strip().lower() for token in connection.split(",") if token.strip()}
    forwarded: list[tuple[str, str]] = []
    for name, value in headers.items():
        lower = name.lower()
        if lower in _HOP_BY_HOP or lower in tokens:
            continue
        forwarded.append((name, value))
    return forwarded


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
    try:
        upstream_request = http.build_request(
            request.method,
            target,
            headers=_forwarded(request.headers),
            content=body,
        )
        response = await http.send(upstream_request, stream=True)
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
        httpx.TimeoutException,
    ) as exc:
        raise UpstreamError(f"upstream {upstream_base} unreachable: {exc}") from exc

    # Response headers get the same hop-by-hop filtering. MutableHeaders
    # preserves duplicate header names (e.g. repeated `set-cookie`).
    raw_headers = [
        (name.encode("latin-1"), value.encode("latin-1"))
        for name, value in _forwarded(response.headers)
    ]

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()

    return StreamingResponse(
        relay(),
        status_code=response.status_code,
        headers=MutableHeaders(raw=raw_headers),
    )
