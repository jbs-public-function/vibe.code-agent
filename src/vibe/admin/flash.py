"""Flash messages via a short-lived cookie (FR-01.11).

A flash message is stored in the `vibe_flash` cookie (URL-quoted, 5-minute
lifetime) and rendered exactly once: the first GET to see it clears the cookie
with an expired `Set-Cookie` in that same response. The text is truncated to
`FLASH_MAX_CHARS` before it ever reaches a cookie (IMP-04.3): flash messages
often carry long `RuntimeError` texts verbatim, and browsers cap cookies at
~4 KiB.
"""

from __future__ import annotations

from urllib.parse import quote, unquote

from starlette.requests import Request
from starlette.responses import Response

FLASH_COOKIE = "vibe_flash"
FLASH_TTL_SECONDS = 300  # short-lived by design (FR-01.11)
FLASH_MAX_CHARS = 500  # IMP-04.3: long RuntimeErrors must not blow the 4 KiB cookie cap


def get_flash(request: Request) -> str | None:
    """Read the pending flash message, if the request carries one."""
    raw = request.cookies.get(FLASH_COOKIE)
    if raw is None:
        return None
    return unquote(raw)


def set_flash(response: Response, message: str) -> None:
    """Attach a (truncated) flash message to a response cookie."""
    message = message[:FLASH_MAX_CHARS]
    # Belt and braces: URL-quoting can triple the byte size of non-ASCII
    # text, so cap the *quoted* value under the ~4 KiB cookie limit too.
    while len(quote(message, safe="")) > 3500 and message:
        message = message[:-1]
    response.set_cookie(
        FLASH_COOKIE,
        quote(message, safe=""),
        max_age=FLASH_TTL_SECONDS,
        path="/",
        httponly=True,
        samesite="lax",  # IMP-04.3: explicit, like every other admin cookie
    )


def clear_flash(response: Response) -> None:
    """Expire the cookie so the message is rendered exactly once (FR-01.11)."""
    response.delete_cookie(
        FLASH_COOKIE, path="/", httponly=True, samesite="lax"
    )
