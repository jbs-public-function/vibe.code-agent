"""Request-security middleware for the admin app (IMP-04.1, IMP-04.28).

`SecurityMiddleware` enforces three properties on every HTTP request:

1. **Host validation** (DNS-rebinding defence): the request `Host` must be in
   the allow-list — `127.0.0.1`, `localhost`, `[::1]` (IPv6 literal) and the
   configured `VIBE_BIND`. When the `Host` header is absent (plain ASGI test
   scopes) the `server` scope value is used instead.
2. **Origin checks for state-changing requests**: non-GET/HEAD requests must
   carry `Sec-Fetch-Site: same-origin` (or `none`); absent that header the
   `Origin` — then `Referer` — host is compared against the `Host` host.
   Requests with **none** of these headers (curl, tests) are allowed: the v1
   admin is loopback-only with no authentication, and CLI ergonomics win.
3. **HEAD support**: `HEAD` is rewritten to `GET` with the response body
   dropped (correct `Content-Length` is preserved by the inner app).

Written as plain ASGI middleware (not `BaseHTTPMiddleware`) so the chunked
chat stream keeps its progressive flush (NFR-01.6). Rejections are logged
with `logging.getLogger("vibe.admin")` and answered with a 403 HTML page
carrying the CSP/nosniff headers.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# NFR-01.1: the exact policy carried by every response; scripts disabled,
# styles allowed only from our own static mount. IMP-04.2 extends it with
# `form-action`/`frame-ancestors`/`base-uri` directives.
CSP = (
    "default-src 'self'; script-src 'none'; style-src 'self'; "
    "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)

logger = logging.getLogger("vibe.admin")

# Loopback hosts always allowed regardless of the configured bind.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# Sec-Fetch-Site values that browsers send for same-origin/navigated requests.
_ALLOWED_FETCH_SITES = {"same-origin", "none"}

# Methods that never mutate state and skip the origin checks.
_SAFE_METHODS = {"GET", "HEAD"}

_FORBIDDEN_BODY = (
    b"<!doctype html>\n"
    b"<html lang=\"en\"><head><meta charset=\"utf-8\">"
    b"<title>403 Forbidden</title></head>\n"
    b"<body><h1>403 Forbidden</h1>"
    b"<p>The request was rejected by the admin security policy.</p></body></html>\n"
)


def _hostname_of(host: str) -> str:
    """The hostname part of a `Host`/`Origin`/`Referer` value.

    `urlsplit` strips brackets from IPv6 literals (`[::1]` → `::1`) so both
    spellings compare equal against the allow-list.
    """
    if not host:
        return ""
    return (urlsplit(f"//{host}").hostname or "").lower()


def _origin_host(value: str) -> str:
    """The host of an `Origin`/`Referer` URL, or `""` when unparseable."""
    if not value:
        return ""
    return (urlsplit(value).hostname or "").lower()


class SecurityMiddleware:
    """Host validation, origin checks and HEAD rewriting (IMP-04.1/04.28)."""

    def __init__(self, app: ASGIApp, allowed_hosts: set[str]) -> None:
        self.app = app
        self.allowed_hosts = {host.lower() for host in allowed_hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        headers = {name.decode("latin-1").lower(): value.decode("latin-1") for name, value in scope.get("headers", [])}
        host_header = headers.get("host") or (scope.get("server") or ("", 0))[0]
        path = scope.get("path", "")

        if _hostname_of(host_header) not in self.allowed_hosts:
            await self._reject(
                send, reason="host", host=host_header, method=method, path=path
            )
            return

        if method not in _SAFE_METHODS:
            fetch_site = headers.get("sec-fetch-site")
            if fetch_site is not None:
                if fetch_site.lower() not in _ALLOWED_FETCH_SITES:
                    await self._reject(
                        send,
                        reason="sec-fetch-site",
                        host=host_header,
                        method=method,
                        path=path,
                    )
                    return
            else:
                origin = headers.get("origin")
                if origin is not None:
                    if _origin_host(origin) != _hostname_of(host_header):
                        await self._reject(
                            send,
                            reason="origin",
                            host=host_header,
                            method=method,
                            path=path,
                        )
                        return
                else:
                    referer = headers.get("referer")
                    if (
                        referer is not None
                        and _origin_host(referer) != _hostname_of(host_header)
                    ):
                        await self._reject(
                            send,
                            reason="referer",
                            host=host_header,
                            method=method,
                            path=path,
                        )
                        return

        if method == "HEAD":
            # IMP-04.28: serve HEAD as GET with the body dropped; the inner
            # app still computes Content-Length as if the body were sent.
            async def send_without_body(message: Message) -> None:
                if message["type"] == "http.response.body":
                    message = {**message, "body": b""}
                await send(message)

            await self.app({**scope, "method": "GET"}, receive, send_without_body)
            return

        await self.app(scope, receive, send)

    async def _reject(
        self, send: Send, *, reason: str, host: str, method: str, path: str
    ) -> None:
        logger.warning(
            "request rejected",
            extra={"reason": reason, "host": host, "method": method, "path": path},
        )
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"content-security-policy", CSP.encode()),
                    (b"x-content-type-options", b"nosniff"),
                ],
            }
        )
        await send(
            {"type": "http.response.body", "body": _FORBIDDEN_BODY, "more_body": False}
        )
