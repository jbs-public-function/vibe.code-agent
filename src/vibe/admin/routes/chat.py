"""Chat routes with server-side sessions and chunked HTML (FR-01.5/01.6, NFR-01.6).

The conversation lives in an in-memory `SessionStore` keyed by the
`vibe_session` cookie. `POST /chat` streams the reply as one HTML page: the
header with the full history is flushed immediately (with ~1 KiB of padding so
Firefox paints), then each `ChatChunk.delta` is appended HTML-escaped, then
the footer with the send form closes the page. If the controller dies
mid-stream an error block is appended inside the page — headers are already
sent, so this can never become a 500.

IMP-04.4: the first chunk gets a 120 s budget (prefill of a long history is
slow) with a padding heartbeat every 5 s; subsequent chunks get a 30 s idle
budget. IMP-04.5: `ControllerUnavailable` reads as "controller unreachable"
while other `RuntimeError`s (NoModelLoaded, UpstreamError, vLLM 400) render
their message. IMP-04.8: the generator never yields inside `finally` — the
partial turn is persisted there, and the footer is emitted after the `try`.
"""

from __future__ import annotations

import asyncio
import html
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import suppress
from typing import Any, cast

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from vibe.admin.routes import (
    aclose,
    client_of,
    grpc,
    parse_form,
    redirect,
    redirect_with_flash,
    render,
    safe_grpc,
)
from vibe.admin.sessions import (
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    ChatSession,
    SessionStore,
)
from vibe.client import ControllerUnavailable
from vibe.config import GATEWAY_PORT
from vibe.gen.vibe.v1 import control_pb2

router = APIRouter()

DEFAULT_TEMPERATURE = 0.7  # FR-01.6
DEFAULT_MAX_TOKENS = 1024  # FR-01.6

# IMP-04.4: the first chunk may take a long prefill; subsequent chunks must
# arrive within an idle budget. A 5 s padding heartbeat keeps the browser
# connection alive during prefill (NFR-01.6).
CHAT_FIRST_TOKEN_S = 120.0
CHAT_IDLE_S = 30.0
CHAT_HEARTBEAT_S = 5.0

# History exceeding this fraction of the active model's context length shows
# a warning (risk note in the Phase 3 spec) and is truncated server-side
# before the request is built (IMP-04.7).
_CONTEXT_WARN_FRACTION = 0.75


def _session_of(request: Request) -> ChatSession:
    store = cast(SessionStore, request.app.state.sessions)
    return store.get_or_create(request.cookies.get(SESSION_COOKIE))


def _store_of(request: Request) -> SessionStore:
    return cast(SessionStore, request.app.state.sessions)


def _set_session_cookie(response: Response, session: ChatSession) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session.session_id,
        max_age=SESSION_TTL_SECONDS,
        path="/",
        httponly=True,
        samesite="lax",  # IMP-04.3: explicit, like every other admin cookie
    )


def _parse_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except ValueError:
        return default


def _token_estimate(messages: Sequence[dict[str, str]]) -> int:
    """Rough token count (~4 chars/token), matching `ChatSession.token_estimate`."""
    return sum(len(message["content"]) // 4 for message in messages)


def truncate_history(
    messages: list[dict[str, str]], budget: int
) -> tuple[list[dict[str, str]], int]:
    """Drop the oldest non-system turns until the estimate fits `budget`.

    Pure function (unit-testable, IMP-04.7). System turns are never dropped —
    chat templates reject a history whose system prompt vanished; if the
    system turns alone exceed the budget every non-system turn goes.
    """
    kept = list(messages)
    dropped = 0
    while _token_estimate(kept) > budget:
        for index, message in enumerate(kept):
            if message.get("role") != "system":
                del kept[index]
                dropped += 1
                break
        else:
            break
    return kept, dropped


@router.get("/chat", name="chat")
async def chat_page(request: Request) -> Response:
    """Conversation history, send/reset forms and the no-model banner (FR-01.5/01.6)."""
    client = client_of(request)
    session = _session_of(request)
    status, error = await safe_grpc(client.status())
    controller_down = status is None
    ready = status is not None and status.state == control_pb2.READY
    max_model_len: int | None = None
    if status is not None and ready and status.active_alias:
        infos, infos_error = await safe_grpc(client.list_models())
        if error is None:
            error = infos_error
        if infos is not None:
            for info in infos:
                if info.alias == status.active_alias:
                    max_model_len = info.max_model_len
                    break
    total_tokens = session.token_estimate()
    warn = bool(
        max_model_len and total_tokens > int(max_model_len * _CONTEXT_WARN_FRACTION)
    )
    response = render(
        request,
        "chat.html",
        {
            "page": "chat",
            "controller_down": controller_down,
            "active_alias": status.active_alias if status is not None else "",
            "ready": ready,
            "session": session,
            "total_tokens": total_tokens,
            "max_model_len": max_model_len,
            "warn": warn,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
        error=error,
    )
    if request.cookies.get(SESSION_COOKIE) != session.session_id:
        _set_session_cookie(response, session)
    return response


def _chat_stream(
    request: Request,
    session: ChatSession,
    alias: str,
    agen: AsyncIterator[control_pb2.ChatChunk],
    temperature: float,
    max_tokens: int,
    max_model_len: int | None,
    omitted: int,
) -> StreamingResponse:
    """The streamed HTML reply page (NFR-01.6): header, escaped chunks, footer."""
    templates = cast(Jinja2Templates, request.app.state.templates)
    store = _store_of(request)
    path_for = request.app.url_path_for

    async def generate() -> AsyncIterator[bytes]:
        header = templates.get_template("chat_stream.html").render(
            request=request,
            phase="header",
            page="chat",
            active_alias=alias,
            session=session,
            path_for=path_for,
            gateway_port=GATEWAY_PORT,
        )
        yield header.encode()
        reply_parts: list[str] = []
        stream_error: str | None = None
        loop = asyncio.get_running_loop()
        first_deadline = loop.time() + CHAT_FIRST_TOKEN_S
        task = asyncio.create_task(
            cast(Coroutine[Any, Any, control_pb2.ChatChunk], anext(agen))
        )
        got_first = False
        try:
            while True:
                if got_first:
                    timeout_s = CHAT_IDLE_S
                else:
                    timeout_s = min(
                        CHAT_HEARTBEAT_S, max(0.0, first_deadline - loop.time())
                    )
                    if timeout_s <= 0:
                        raise asyncio.TimeoutError
                try:
                    chunk = await asyncio.wait_for(
                        asyncio.shield(task), timeout=timeout_s
                    )
                except StopAsyncIteration:
                    break  # the stream ended cleanly
                except asyncio.TimeoutError:
                    if got_first:
                        stream_error = (
                            "controller unreachable — the reply ended early"
                        )
                        break
                    yield b"<!-- keep-alive -->\n"
                    continue
                got_first = True
                task = asyncio.create_task(
                    cast(Coroutine[Any, Any, control_pb2.ChatChunk], anext(agen))
                )
                if chunk.delta:
                    reply_parts.append(chunk.delta)
                    yield html.escape(chunk.delta).encode()
        except ControllerUnavailable:
            stream_error = "controller unreachable — the reply ended early"
        except asyncio.TimeoutError:
            stream_error = "controller unreachable — the reply ended early"
        except RuntimeError as exc:
            # IMP-04.5: a real upstream error (NoModelLoaded, UpstreamError,
            # vLLM 400 …) renders its message, escaped by the template.
            stream_error = str(exc)
        finally:
            # IMP-04.8: no `yield` in here — persisting the partial turn is
            # plain state, and the footer is emitted after the try block.
            if not task.done():
                task.cancel()
            with suppress(
                asyncio.CancelledError, StopAsyncIteration, RuntimeError
            ):
                await task
            with suppress(Exception):
                await aclose(agen)
            if reply_parts:
                session.append({"role": "assistant", "content": "".join(reply_parts)})
                store.touch(session)
        footer = templates.get_template("chat_stream.html").render(
            request=request,
            phase="footer",
            page="chat",
            active_alias=alias,
            error=stream_error,
            omitted=omitted,
            temperature=temperature,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            path_for=path_for,
            gateway_port=GATEWAY_PORT,
        )
        yield footer.encode()

    response = StreamingResponse(generate(), media_type="text/html; charset=utf-8")
    response.headers["Cache-Control"] = "no-store"  # NFR-01.6
    _set_session_cookie(response, session)
    return response


@router.post("/chat", name="chat_send")
async def chat_send(request: Request) -> Response:
    """Append the user turn and stream the model's reply (FR-01.5, NFR-01.6)."""
    client = client_of(request)
    session = _session_of(request)
    store = _store_of(request)
    form = await parse_form(request)
    message = form.get("message", "").strip()
    if not message:
        return redirect("/chat")
    temperature = min(
        2.0,
        max(
            0.0,
            _parse_float(
                form.get("temperature") or str(DEFAULT_TEMPERATURE),
                DEFAULT_TEMPERATURE,
            ),
        ),
    )  # IMP-04.10: clamped 0–2 server-side
    max_tokens = max(
        1,
        _parse_int(
            form.get("max_tokens") or str(DEFAULT_MAX_TOKENS), DEFAULT_MAX_TOKENS
        ),
    )
    try:
        status = await grpc(client.status())
    except RuntimeError as exc:
        return redirect_with_flash("/chat", str(exc))
    if status is None:
        return redirect_with_flash(
            "/chat", "controller unreachable — cannot chat right now"
        )
    if status.state != control_pb2.READY:
        return redirect_with_flash(
            "/chat", "no model loaded — load one on the models page"
        )
    max_model_len: int | None = None
    if status.active_alias:
        try:
            infos = await grpc(client.list_models())
        except RuntimeError as exc:
            return redirect_with_flash("/chat", str(exc))
        if infos is not None:
            for info in infos:
                if info.alias == status.active_alias:
                    max_model_len = info.max_model_len
                    break
    if max_model_len:
        max_tokens = min(max_tokens, max_model_len)  # IMP-04.10: upper bound
        truncated, dropped = truncate_history(
            session.messages, int(max_model_len * _CONTEXT_WARN_FRACTION)
        )
        session.messages[:] = truncated  # IMP-04.7: history truncated server-side
    else:
        dropped = 0
    # IMP-04.6: only now, after every check passed, does the user turn stick —
    # a failed send must not leave a dangling turn.
    session.append({"role": "user", "content": message})
    store.touch(session)
    proto_messages = [
        control_pb2.ChatMessage(role=turn["role"], content=turn["content"])
        for turn in session.messages
    ]
    agen = client.chat(
        status.active_alias,
        proto_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _chat_stream(
        request,
        session,
        status.active_alias,
        agen,
        temperature,
        max_tokens,
        max_model_len,
        dropped,
    )


@router.post("/chat/reset", name="chat_reset")
async def chat_reset(request: Request) -> Response:
    """Drop the session and clear its cookie (FR-01.5)."""
    store = _store_of(request)
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        store.reset(session_id)
    response = redirect("/chat")
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax")
    return response
