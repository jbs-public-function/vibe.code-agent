"""Container log viewer (FR-01.7, NFR-01.5).

`GET /logs?tail=N` renders the last N lines in a `<pre>` with a 5 s
auto-refresh and a tail selector (100/200/500/1000). `GET /logs/stream`
streams follow mode as `text/plain` forever, for curl or a browser tab.

IMP-04.11: the stream endpoint defaults to `tail=50` so the first chunk is a
replay, and — instead of a first-line deadline that kills quiet containers —
channel readiness is verified with a bounded `status()` call before the
undeadlined iteration begins. IMP-04.12: `tail` is accepted as a string and
coerced manually, so `?tail=abc` renders an HTML error rather than FastAPI's
422 JSON.
"""

from __future__ import annotations

import asyncio
import html
from collections.abc import AsyncIterator

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response, StreamingResponse

from vibe.admin.routes import (
    client_of,
    fmt_ts,
    grpc,
    render,
    safe_collect,
    safe_grpc,
)
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2

router = APIRouter()

TAILS = (100, 200, 500, 1000)  # FR-01.7
DEFAULT_TAIL = 200
DEFAULT_STREAM_TAIL = 50  # IMP-04.11: replay first, then follow
_MAX_TAIL = 10000


def _coerce_tail(raw: str) -> int:
    """`int(raw)` with the admin's own error instead of a FastAPI 422."""
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"tail must be a number, got {raw!r}") from exc


def _tail_error(message: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Bad request</title></head><body><h1>Bad request</h1>"
        f"<p>{html.escape(message)}</p></body></html>",
        status_code=400,
    )


def _format_log_line(line: control_pb2.LogLine) -> bytes:
    return f"{fmt_ts(line.ts_unix_ms)}  {line.line}\n".encode()


@router.get("/logs", name="logs")
async def logs_page(request: Request, tail: str = str(DEFAULT_TAIL)) -> Response:
    """Last `tail` log lines in a `<pre>`, refreshing every 5 s (FR-01.7)."""
    try:
        tail_value = _coerce_tail(tail)
    except ValueError as exc:
        return _tail_error(str(exc))
    if tail_value not in TAILS:
        tail_value = DEFAULT_TAIL
    client = client_of(request)
    (lines, lines_error), (status, status_error) = await asyncio.gather(
        safe_collect(client.logs(tail=tail_value)), safe_grpc(client.status())
    )
    controller_down = lines is None or status is None
    return render(
        request,
        "logs.html",
        {
            "page": "logs",
            "controller_down": controller_down,
            "active_alias": status.active_alias if status is not None else "",
            "lines": lines if lines is not None else [],
            "tail": tail_value,
            "tails": TAILS,
        },
        error=lines_error or status_error,
    )


@router.get("/logs/stream", name="logs_stream")
async def logs_stream(
    request: Request, tail: str = str(DEFAULT_STREAM_TAIL), follow: str = "1"
) -> Response:
    """Follow mode as chunked `text/plain` — streams until the tab closes.

    `tail` defaults to 50 so the first chunk is an immediate replay
    (IMP-04.11); `follow` is accepted for compatibility with the documented
    link format and ignored — this endpoint always follows.
    """
    try:
        tail_value = _coerce_tail(tail)
    except ValueError as exc:
        return _tail_error(str(exc))
    tail_value = max(0, min(tail_value, _MAX_TAIL))
    client = client_of(request)
    # IMP-04.11: verify channel readiness under the 3 s deadline *before*
    # iterating, so a dead controller still fails fast without imposing the
    # deadline on the (potentially quiet) stream itself.
    try:
        status = await grpc(client.status())
    except RuntimeError:
        status = None  # the stream itself will surface the message
    if status is None:
        return StreamingResponse(
            _one_line("controller unreachable\n"),
            media_type="text/plain; charset=utf-8",
        )
    agen = client.logs(tail=tail_value, follow=True)

    async def generate() -> AsyncIterator[bytes]:
        try:
            async for line in agen:
                yield _format_log_line(line)
        except ControllerUnavailable:
            yield "controller unreachable\n".encode()
        except RuntimeError as exc:
            # IMP-04.15: a real error (e.g. "no managed container running")
            # surfaces with its message instead of being mislabeled.
            yield f"error: {exc}\n".encode()

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


async def _one_line(text: str) -> AsyncIterator[bytes]:
    yield text.encode()
