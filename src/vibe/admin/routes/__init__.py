"""Shared helpers for the admin routers (FR-01.x, NFR-01.5).

Every controller call goes through one of the small wrappers here so the 3 s
deadline and the "controller unreachable" degradation (NFR-01.5) are applied
consistently. POST bodies are parsed with `parse_qsl` — the admin must not
depend on `python-multipart`, so `request.form()` is never used.

IMP-04.15: the wrappers treat only `ControllerUnavailable` and deadline
expiry as "unreachable"; any other `RuntimeError` (e.g. a vLLM 400 surfaced
through `UpstreamError`) is a real error that routes surface with its
message. `safe_grpc`/`safe_collect` pair every value with that message so
page routes can render a distinct error banner instead of a 500.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable
from contextlib import suppress
from datetime import datetime
from typing import TypeVar, cast
from urllib.parse import parse_qsl

from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from vibe.admin.flash import clear_flash, get_flash, set_flash
from vibe.client import ControllerClient, ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2
from vibe.jobs import JobInProgress

# NFR-01.5: no controller call may outlive this deadline.
GRPC_TIMEOUT = 3.0

_T = TypeVar("_T")

# Terminal job-event states (FR-01.4): these end the job, like `terminal=True`.
_TERMINAL_STATES = (control_pb2.READY, control_pb2.AVAILABLE, control_pb2.ERROR)


async def aclose(agen: AsyncIterator[object]) -> None:
    """Best-effort close of an async generator typed as `AsyncIterator`.

    The client methods are annotated `AsyncIterator[...]` but return async
    generators; `aclose()` releases the underlying stream (IMP-04.15).
    """
    await cast(AsyncGenerator[object, None], agen).aclose()


async def grpc(awaitable: Awaitable[_T]) -> _T | None:
    """Await a unary controller call under the 3 s deadline; `None` = down.

    `KeyError` (unknown alias) and `JobInProgress` propagate so routes can
    handle them specifically; `ControllerUnavailable` and deadline expiry
    degrade to `None` (NFR-01.5). Any other `RuntimeError` propagates — it is
    a real error whose message must reach the user (IMP-04.15).
    """
    try:
        return await asyncio.wait_for(awaitable, timeout=GRPC_TIMEOUT)
    except (JobInProgress, KeyError):
        raise
    except (asyncio.TimeoutError, ControllerUnavailable):
        return None


async def collect(agen: AsyncIterator[_T]) -> list[_T] | None:
    """Collect a bounded stream under the 3 s deadline; `None` = down.

    The deadline applies to the whole collect, not per item (NFR-01.5).
    `KeyError` and `JobInProgress` propagate for route-specific handling;
    other `RuntimeError`s propagate (IMP-04.15). On deadline expiry the
    generator is `aclose()`d so the underlying stream is released
    (IMP-04.15).
    """

    async def _collect() -> list[_T]:
        return [item async for item in agen]

    try:
        return await asyncio.wait_for(_collect(), timeout=GRPC_TIMEOUT)
    except (JobInProgress, KeyError):
        raise
    except asyncio.TimeoutError:
        with suppress(Exception):
            await aclose(agen)
        return None
    except ControllerUnavailable:
        return None


async def safe_grpc(awaitable: Awaitable[_T]) -> tuple[_T | None, str | None]:
    """`grpc()` plus a `RuntimeError` message for the error banner (IMP-04.15)."""
    try:
        return await grpc(awaitable), None
    except RuntimeError as exc:
        return None, str(exc)


async def safe_collect(agen: AsyncIterator[_T]) -> tuple[list[_T] | None, str | None]:
    """`collect()` plus a `RuntimeError` message for the error banner (IMP-04.15)."""
    try:
        return await collect(agen), None
    except RuntimeError as exc:
        return None, str(exc)


async def grpc_flash(awaitable: Awaitable[_T]) -> tuple[_T | None, str | None]:
    """`grpc()` plus a `RuntimeError` message for a flash on POST routes (IMP-04.15).

    `None` with `message=None` means "unreachable" (degradation flash); `None`
    with a message is a real controller error to flash verbatim.
    """
    try:
        return await grpc(awaitable), None
    except RuntimeError as exc:
        return None, str(exc)


async def parse_form(request: Request) -> dict[str, str]:
    """Parse a urlencoded POST body; `{}` when the body is not decodable."""
    try:
        text = (await request.body()).decode()
    except UnicodeDecodeError:
        return {}
    return dict(parse_qsl(text, keep_blank_values=True))


def client_of(request: Request) -> ControllerClient:
    """The controller client injected by `build_app`."""
    return cast(ControllerClient, request.app.state.client)


def render(
    request: Request, name: str, context: dict[str, object], error: str | None = None
) -> HTMLResponse:
    """Render a page template, consuming any pending flash (FR-01.11).

    `error` is a distinct controller-error banner (IMP-04.15), separate from
    the "controller unreachable" degradation.
    """
    templates = cast(Jinja2Templates, request.app.state.templates)
    flash_text = get_flash(request)
    response = templates.TemplateResponse(
        request, name, {**context, "flash": flash_text, "error": error}
    )
    if flash_text is not None:
        clear_flash(response)
    return response


def redirect(url: str) -> RedirectResponse:
    """303 See Other — every POST except `/chat` answers with a redirect."""
    return RedirectResponse(url, status_code=303)


def redirect_with_flash(url: str, message: str) -> RedirectResponse:
    """303-redirect carrying a flash message in a cookie (FR-01.11)."""
    response = redirect(url)
    set_flash(response, message)
    return response


async def start_background_job(
    request: Request,
    agen: AsyncIterator[control_pb2.JobEvent],
    kind: str,
    alias: str,
) -> tuple[str, control_pb2.JobEvent]:
    """Start a load/unload stream and hand it to a background task (FR-01.3).

    The first event carries the job id; both are returned so callers can
    react to an error on the very first event (Jev launches flash a
    `jev_schema_invalid` first event instead of redirecting). The remainder
    of the stream is drained by a task held in `app.state.background_tasks`
    (discarded via a done-callback) so the job outlives the HTTP request.
    `KeyError`, `JobInProgress` and controller failures from the first event
    propagate to the caller for route-specific flash handling; on a
    first-event deadline the generator is `aclose()`d (IMP-04.15).
    """
    try:
        first = await asyncio.wait_for(anext(agen), timeout=GRPC_TIMEOUT)
    except asyncio.TimeoutError:
        with suppress(Exception):
            await aclose(agen)
        raise
    job_id = first.job_id

    async def drain() -> None:
        try:
            async for _ in agen:
                pass
        except Exception:
            pass  # a background job must never raise into the loop

    tasks = cast(set[asyncio.Task[None]], request.app.state.background_tasks)
    task = asyncio.create_task(drain())
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    return job_id, first


def is_terminal(event: control_pb2.JobEvent) -> bool:
    """A job event ends its job either via the flag or a terminal state (FR-01.4)."""
    return event.terminal or event.state in _TERMINAL_STATES


# -- Jinja filters (registered on the env by `build_app`) -------------------


def fmt_ts(ts_unix_ms: int) -> str:
    """`ts_unix_ms` (int milliseconds) as local HH:MM:SS."""
    try:
        return datetime.fromtimestamp(ts_unix_ms / 1000).strftime("%H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "--:--:--"


def fmt_uptime(seconds: float | None) -> str:
    """Seconds as a compact `1h 02m` style duration."""
    if seconds is None or seconds < 0:
        return "—"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_thousands(value: int) -> str:
    """Thousands separators only — use `_fmt_size` for byte counts (IMP-04.30)."""
    return f"{value:,}"


def job_kind_name(kind: int) -> str:
    """Human-readable `JobKind` name; unset (0) reads as a dash."""
    if kind == control_pb2.JOB_KIND_UNSPECIFIED:
        return "—"
    return control_pb2.JobKind.Name(kind)
