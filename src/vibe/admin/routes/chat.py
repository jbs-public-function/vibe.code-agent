"""Chat routes with server-side sessions and chunked HTML (FR-01.5/01.6, NFR-01.6).

The conversation lives in an in-memory `SessionStore` keyed by the
`vibe_session` cookie. `POST /chat` streams the reply as one HTML page: the
header with the full history is flushed immediately (with ~1 KiB of padding so
Firefox paints), then each `ChatChunk.delta` is appended HTML-escaped, then
the footer with the send form closes the page. If the controller dies
mid-stream an error block is appended inside the page — headers are already
sent, so this can never become a 500.
"""

from __future__ import annotations

import asyncio
import html
from collections.abc import AsyncIterator
from typing import cast

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from vibe.admin.routes import (
    GRPC_TIMEOUT,
    client_of,
    grpc,
    parse_form,
    redirect,
    redirect_with_flash,
    render,
)
from vibe.admin.sessions import SESSION_COOKIE, SESSION_TTL_SECONDS, ChatSession, SessionStore
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2

router = APIRouter()

DEFAULT_TEMPERATURE = 0.7  # FR-01.6
DEFAULT_MAX_TOKENS = 1024  # FR-01.6

# History exceeding this fraction of the active model's context length shows
# a warning (risk note in the Phase 3 spec).
_CONTEXT_WARN_FRACTION = 0.75


def _session_of(request: Request) -> ChatSession:
    store = cast(SessionStore, request.app.state.sessions)
    return store.get_or_create(request.cookies.get(SESSION_COOKIE))


def _set_session_cookie(response: Response, session: ChatSession) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session.session_id,
        max_age=SESSION_TTL_SECONDS,
        path="/",
        httponly=True,
    )


@router.get("/chat", name="chat")
async def chat_page(request: Request) -> Response:
    """Conversation history, send/reset forms and the no-model banner (FR-01.5/01.6)."""
    client = client_of(request)
    session = _session_of(request)
    status = await grpc(client.status())
    controller_down = status is None
    ready = status is not None and status.state == control_pb2.READY
    max_model_len: int | None = None
    if status is not None and ready and status.active_alias:
        infos = await grpc(client.list_models())
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
) -> StreamingResponse:
    """The streamed HTML reply page (NFR-01.6): header, escaped chunks, footer."""
    templates = cast(Jinja2Templates, request.app.state.templates)

    async def generate() -> AsyncIterator[bytes]:
        header = templates.get_template("chat_stream.html").render(
            request=request,
            phase="header",
            page="chat",
            active_alias=alias,
            session=session,
        )
        yield header.encode()
        reply_parts: list[str] = []
        stream_error: str | None = None
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(agen), timeout=GRPC_TIMEOUT)
                except StopAsyncIteration:
                    break
                except (asyncio.TimeoutError, ControllerUnavailable, RuntimeError):
                    stream_error = "controller unreachable — the reply ended early"
                    break
                if chunk.delta:
                    reply_parts.append(chunk.delta)
                    yield html.escape(chunk.delta).encode()
        finally:
            if reply_parts:
                session.messages.append(
                    {"role": "assistant", "content": "".join(reply_parts)}
                )
            footer = templates.get_template("chat_stream.html").render(
                request=request,
                phase="footer",
                page="chat",
                active_alias=alias,
                error=stream_error,
                temperature=temperature,
                max_tokens=max_tokens,
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
    form = await parse_form(request)
    message = form.get("message", "").strip()
    if not message:
        return redirect("/chat")
    try:
        temperature = float(form.get("temperature") or DEFAULT_TEMPERATURE)
    except ValueError:
        temperature = DEFAULT_TEMPERATURE
    try:
        max_tokens = int(form.get("max_tokens") or DEFAULT_MAX_TOKENS)
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS
    session.messages.append({"role": "user", "content": message})
    status = await grpc(client.status())
    if status is None:
        return redirect_with_flash(
            "/chat", "controller unreachable — cannot chat right now"
        )
    if status.state != control_pb2.READY:
        return redirect_with_flash(
            "/chat", "no model loaded — load one on the models page"
        )
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
        request, session, status.active_alias, agen, temperature, max_tokens
    )


@router.post("/chat/reset", name="chat_reset")
async def chat_reset(request: Request) -> Response:
    """Drop the session and clear its cookie (FR-01.5)."""
    store = cast(SessionStore, request.app.state.sessions)
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        store.reset(session_id)
    response = redirect("/chat")
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True)
    return response
