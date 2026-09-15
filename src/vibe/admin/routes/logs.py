"""Container log viewer (FR-01.7, NFR-01.5).

`GET /logs?tail=N` renders the last N lines in a `<pre>` with a 5 s
auto-refresh and a tail selector (100/200/500/1000). `GET /logs/stream`
streams follow mode as `text/plain` forever, for curl or a browser tab.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from vibe.admin.routes import GRPC_TIMEOUT, client_of, collect, fmt_ts, grpc, render
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2

router = APIRouter()

TAILS = (100, 200, 500, 1000)  # FR-01.7
DEFAULT_TAIL = 200


def _format_log_line(line: control_pb2.LogLine) -> bytes:
    return f"{fmt_ts(line.ts_unix_ms)}  {line.line}\n".encode()


@router.get("/logs", name="logs")
async def logs_page(request: Request, tail: int = DEFAULT_TAIL) -> Response:
    """Last `tail` log lines in a `<pre>`, refreshing every 5 s (FR-01.7)."""
    client = client_of(request)
    if tail not in TAILS:
        tail = DEFAULT_TAIL
    lines, status = await asyncio.gather(
        collect(client.logs(tail=tail)), grpc(client.status())
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
            "tail": tail,
            "tails": TAILS,
        },
    )


@router.get("/logs/stream", name="logs_stream")
async def logs_stream(request: Request, tail: int = 0) -> StreamingResponse:
    """Follow mode as chunked `text/plain` — streams until the tab closes (FR-01.7)."""
    client = client_of(request)
    agen = client.logs(tail=tail, follow=True)

    async def generate() -> AsyncIterator[bytes]:
        # Only the first chunk is deadline-bounded so a dead controller fails
        # fast; afterwards follow mode streams indefinitely by design
        # (NFR-01.5).
        try:
            first = await asyncio.wait_for(anext(agen), timeout=GRPC_TIMEOUT)
            yield _format_log_line(first)
        except StopAsyncIteration:
            return
        except (asyncio.TimeoutError, ControllerUnavailable, RuntimeError):
            yield "controller unreachable\n".encode()
            return
        try:
            async for line in agen:
                yield _format_log_line(line)
        except (ControllerUnavailable, RuntimeError):
            return  # the stream just ends when the controller goes away

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
