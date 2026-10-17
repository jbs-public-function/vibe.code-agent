"""Jev chat panel routes (improvements-2, doc 07).

The panel sends prompts **through the gateway** (`POST /v1/chat/completions`)
so the active Jev schema is enforced in its single authoritative place
(ADR-J2) — the controller's `Chat` RPC proxies vLLM directly and is
Jev-unaware by design. `POST /jev/chat` streams the SSE answer back as
chunked HTML exactly like `/chat` (two-phase render, heartbeat padding,
first-token/idle budgets, no `yield` in `finally`), then appends an inspected
result: the assembled JSON pretty-printed, a schema-validity verdict computed
with the same `jsonschema` dialect mapping as the activation probe, the
`finish_reason` (with a truncation warning for `length`), token usage and the
gateway's `X-Vibe-Jev` header.

The conversation lives in the same in-memory `SessionStore` as `/chat` but
under a distinct cookie (`vibe_jev_session`), so the two panels never share
history. Gateway failures degrade to a flash redirect before the user turn is
persisted (IMP-04.6), so a rejected send never leaves a dangling turn.
"""

from __future__ import annotations

import asyncio
import html
import json
from collections.abc import AsyncIterator, Coroutine
from contextlib import suppress
from typing import Any, cast

import httpx
from fastapi import APIRouter
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from vibe.admin.routes import client_of, grpc, parse_form, redirect, redirect_with_flash, render, safe_grpc
from vibe.admin.routes.chat import (
    CHAT_FIRST_TOKEN_S,
    CHAT_HEARTBEAT_S,
    CHAT_IDLE_S,
    _CONTEXT_WARN_FRACTION,
    _parse_float,
    _parse_int,
    truncate_history,
)
from vibe.admin.routes.jev import SOURCE_LABELS
from vibe.admin.sessions import SESSION_TTL_SECONDS, ChatSession, SessionStore
from vibe.config import GATEWAY_PORT
from vibe.gen.vibe.v1 import control_pb2
from vibe.jev import instance_validator_for

router = APIRouter()

# A session separate from the plain `/chat` panel's (JEV-07.6).
JEV_SESSION_COOKIE = "vibe_jev_session"

DEFAULT_TEMPERATURE = 0.7  # same as /chat
DEFAULT_MAX_TOKENS = 512  # Jev outputs are small bounded JSON

# The gateway probe reuses the dashboard's 0.5 s budget; the chat stream needs
# generous reads (cold first tokens) but a bounded connect (JEV-07.3).
PROBE_TIMEOUT_S = 0.5
GATEWAY_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)


def _session_of(request: Request) -> ChatSession:
    store = cast(SessionStore, request.app.state.sessions)
    return store.get_or_create(request.cookies.get(JEV_SESSION_COOKIE))


def _store_of(request: Request) -> SessionStore:
    return cast(SessionStore, request.app.state.sessions)


def _set_session_cookie(response: Response, session: ChatSession) -> None:
    response.set_cookie(
        JEV_SESSION_COOKIE,
        session.session_id,
        max_age=SESSION_TTL_SECONDS,
        path="/",
        httponly=True,
        samesite="lax",
    )


def _gateway_chat_url(request: Request) -> str:
    """The gateway's chat-completions URL for the configured bind (JEV-07.2).

    Derived from `app.state.gateway_health_url` (`http://<bind>:8080/health`),
    which already handles wildcard binds and IPv6 bracketing.
    """
    base = cast(str, request.app.state.gateway_health_url)
    if base.endswith("/health"):
        base = base[: -len("/health")]
    return f"{base}/v1/chat/completions"


async def _gateway_reachable(request: Request) -> bool:
    """Cheap `/health` probe; any failure counts as down (JEV-07.8).

    Tests inject `app.state.gateway_transport` (a `httpx.MockTransport`) so the
    probe never dials the real gateway — the same hook the dashboard uses.
    """
    url = cast(str, request.app.state.gateway_health_url)
    transport = getattr(request.app.state, "gateway_transport", None)
    try:
        if transport is None:
            client = cast(httpx.AsyncClient, request.app.state.http)
            response = await client.get(url)
        else:
            async with httpx.AsyncClient(
                transport=transport, timeout=PROBE_TIMEOUT_S
            ) as probe:
                response = await probe.get(url)
    except Exception:
        return False
    return response.status_code == 200


def _gateway_error_message(body: bytes) -> str:
    """The client-facing message of a non-200 gateway answer (JEV-07.5).

    Accepts both the gateway's `{"message": …}` shape and the OpenAI
    `{"error": {"message": …}}` shape; falls back to the raw body.
    """
    try:
        data = json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace")[:300] or "the gateway rejected the request"
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
        error = data.get("error")
        if isinstance(error, dict):
            inner = error.get("message")
            if isinstance(inner, str) and inner:
                return inner
    return "the gateway rejected the request"


def _extract_json(text: str) -> tuple[Any, str | None]:
    """Parse a reply as JSON; falls back to stripping ``` fences (doc 07).

    Jev-mode answers are raw JSON, but plain-mode models often wrap output in
    markdown fences — cheap to support. Returns `(parsed, error)` where the
    error is the last `json.loads` failure message (or `None` on success).
    """
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) > 1 and lines[-1].strip().startswith("```"):
            candidates.append("\n".join(lines[1:-1]).strip())
    error: str | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except ValueError as exc:
            error = f"{exc}"
    return None, error


def _jev_chat_stream(
    request: Request,
    session: ChatSession,
    alias: str,
    http: httpx.AsyncClient,
    upstream: httpx.Response,
    temperature: float,
    max_tokens: int,
    max_model_len: int | None,
    omitted: int,
    schema: dict[str, Any] | None,
    schema_name: str,
    schema_sha: str,
    schema_stale: bool,
    jev_header: str,
) -> StreamingResponse:
    """The streamed HTML reply page: escaped deltas, then the verdict (JEV-07.3/07.4)."""
    templates = cast(Jinja2Templates, request.app.state.templates)
    store = _store_of(request)
    path_for = request.app.url_path_for

    async def generate() -> AsyncIterator[bytes]:
        header = templates.get_template("jev_chat_stream.html").render(
            request=request,
            phase="header",
            page="jev_chat",
            active_alias=alias,
            session=session,
            path_for=path_for,
            gateway_port=GATEWAY_PORT,
            schema_name=schema_name,
        )
        yield header.encode()
        reply_parts: list[str] = []
        stream_error: str | None = None
        finish_reason = ""
        prompt_tokens = 0
        completion_tokens = 0
        loop = asyncio.get_running_loop()
        first_deadline = loop.time() + CHAT_FIRST_TOKEN_S
        lines = upstream.aiter_lines()
        task = asyncio.create_task(cast(Coroutine[Any, Any, str], anext(lines)))
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
                    line = await asyncio.wait_for(
                        asyncio.shield(task), timeout=timeout_s
                    )
                except StopAsyncIteration:
                    break  # the stream ended cleanly
                except asyncio.TimeoutError:
                    if got_first:
                        stream_error = (
                            "gateway stream stalled — the reply ended early"
                        )
                        break
                    yield b"<!-- keep-alive -->\n"
                    continue
                got_first = True
                task = asyncio.create_task(
                    cast(Coroutine[Any, Any, str], anext(lines))
                )
                raw = line.strip()
                if not raw.startswith("data:"):
                    continue
                data = raw[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue  # keep-alive or malformed SSE line
                if not isinstance(obj, dict):
                    continue
                choices = cast(list[Any], obj.get("choices") or [])
                if choices and isinstance(choices[0], dict):
                    choice = cast(dict[str, Any], choices[0])
                    delta = (
                        cast(dict[str, Any], choice.get("delta") or {}).get(
                            "content"
                        )
                        or ""
                    )
                    if delta:
                        reply_parts.append(delta)
                        yield html.escape(delta).encode()
                    if choice.get("finish_reason"):
                        finish_reason = str(choice.get("finish_reason"))
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = int(cast(Any, usage.get("prompt_tokens")) or 0)
                    completion_tokens = int(
                        cast(Any, usage.get("completion_tokens")) or 0
                    )
        except httpx.HTTPError as exc:
            stream_error = f"gateway stream failed: {exc}"
        except asyncio.TimeoutError:
            stream_error = "gateway stream stalled — the reply ended early"
        finally:
            # No `yield` in here (IMP-04.8): persisting the partial turn is
            # plain state, and the footer is emitted after the try block.
            if not task.done():
                task.cancel()
            # CancelledError is a BaseException — suppress it explicitly like
            # chat.py, or the stream dies before the footer (it cancelled the
            # pending read task after `[DONE]`).
            with suppress(asyncio.CancelledError, Exception):
                await task
            with suppress(asyncio.CancelledError, Exception):
                await upstream.aclose()
            with suppress(asyncio.CancelledError, Exception):
                await http.aclose()
            if reply_parts:
                session.append({"role": "assistant", "content": "".join(reply_parts)})
                store.touch(session)

        # -- inspected result (JEV-07.4) ------------------------------------
        reply = "".join(reply_parts)
        parsed, parse_error = _extract_json(reply)
        verdict_valid: bool | None = None
        verdict_errors: list[str] = []
        if schema is not None:
            if parsed is None:
                verdict_valid = False
                verdict_errors = [f"reply is not valid JSON: {parse_error}"]
            else:
                try:
                    validator = instance_validator_for(schema)
                    if validator(schema).is_valid(parsed):
                        verdict_valid = True
                    else:
                        verdict_valid = False
                        verdict_errors = [
                            error.message
                            for error in sorted(
                                validator(schema).iter_errors(parsed),
                                key=lambda error: list(error.path),
                            )[:3]
                        ]
                except Exception as exc:
                    # A store schema is validated on write, but never let a
                    # validator surprise break the verdict block.
                    verdict_valid = None
                    verdict_errors = [f"could not validate: {exc}"]
        pretty_json = json.dumps(parsed, indent=2) if parsed is not None else reply
        footer = templates.get_template("jev_chat_stream.html").render(
            request=request,
            phase="footer",
            page="jev_chat",
            active_alias=alias,
            error=stream_error,
            omitted=omitted,
            temperature=temperature,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            path_for=path_for,
            gateway_port=GATEWAY_PORT,
            reply_pretty=pretty_json,
            verdict_valid=verdict_valid,
            verdict_errors=verdict_errors,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            schema_name=schema_name,
            schema_sha=schema_sha,
            schema_stale=schema_stale,
            jev_header=jev_header,
        )
        yield footer.encode()

    response = StreamingResponse(generate(), media_type="text/html; charset=utf-8")
    response.headers["Cache-Control"] = "no-store"  # NFR-01.6
    _set_session_cookie(response, session)
    return response


@router.get("/jev/chat", name="jev_chat")
async def jev_chat_page(request: Request) -> Response:
    """History, active-schema panel and the send/reset forms (JEV-07.1)."""
    client = client_of(request)
    session = _session_of(request)
    (status, status_error), (active, active_error) = await asyncio.gather(
        safe_grpc(client.status()), safe_grpc(client.get_active_jev())
    )
    error = status_error or active_error
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
    gateway_ok = await _gateway_reachable(request)
    total_tokens = session.token_estimate()
    warn = bool(
        max_model_len and total_tokens > int(max_model_len * _CONTEXT_WARN_FRACTION)
    )
    active_jev = active.info if active is not None and active.HasField("info") else None
    active_source_label = (
        SOURCE_LABELS.get(active_jev.source, active_jev.source)
        if active_jev is not None
        else ""
    )
    response = render(
        request,
        "jev_chat.html",
        {
            "page": "jev_chat",
            "controller_down": controller_down,
            "gateway_ok": gateway_ok,
            "active_alias": status.active_alias if status is not None else "",
            "ready": ready,
            "active_jev": active_jev,
            "active_source_label": active_source_label,
            "session": session,
            "total_tokens": total_tokens,
            "max_model_len": max_model_len,
            "warn": warn,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
        error=error,
    )
    if request.cookies.get(JEV_SESSION_COOKIE) != session.session_id:
        _set_session_cookie(response, session)
    return response


@router.post("/jev/chat", name="jev_chat_send")
async def jev_chat_send(request: Request) -> Response:
    """Append the user turn and stream the Jev-constrained reply (JEV-07.2/07.5)."""
    client = client_of(request)
    session = _session_of(request)
    store = _store_of(request)
    form = await parse_form(request)
    message = form.get("message", "").strip()
    if not message:
        return redirect("/jev/chat")
    temperature = min(
        2.0,
        max(
            0.0,
            _parse_float(
                form.get("temperature") or str(DEFAULT_TEMPERATURE),
                DEFAULT_TEMPERATURE,
            ),
        ),
    )
    max_tokens = max(
        1,
        _parse_int(
            form.get("max_tokens") or str(DEFAULT_MAX_TOKENS), DEFAULT_MAX_TOKENS
        ),
    )
    status = await grpc(client.status())
    if status is None:
        return redirect_with_flash(
            "/jev/chat", "controller unreachable — cannot chat right now"
        )
    if status.state != control_pb2.READY:
        return redirect_with_flash(
            "/jev/chat", "no model loaded — load one on the models page"
        )
    max_model_len: int | None = None
    if status.active_alias:
        try:
            infos = await grpc(client.list_models())
        except RuntimeError as exc:
            return redirect_with_flash("/jev/chat", str(exc))
        if infos is not None:
            for info in infos:
                if info.alias == status.active_alias:
                    max_model_len = info.max_model_len
                    break
    if max_model_len:
        max_tokens = min(max_tokens, max_model_len)  # upper bound, like /chat
        truncated, dropped = truncate_history(
            session.messages, int(max_model_len * _CONTEXT_WARN_FRACTION)
        )
        session.messages[:] = truncated
    else:
        dropped = 0

    # The schema active *now* is what the gateway will enforce (JEV-07.4).
    schema: dict[str, Any] | None = None
    schema_name = ""
    schema_sha = ""
    schema_stale = False
    try:
        active = await grpc(client.get_active_jev())
    except RuntimeError as exc:
        return redirect_with_flash("/jev/chat", str(exc))
    if active is not None and active.HasField("info"):
        info = active.info
        schema_name = info.schema_name
        schema_sha = info.sha256
        schema_stale = info.stale
        try:
            body = await grpc(client.get_jev_schema(info.schema_name))
        except RuntimeError as exc:
            return redirect_with_flash(
                "/jev/chat", f"cannot fetch the active Jev schema: {exc}"
            )
        if body is not None:
            try:
                schema = json.loads(body.schema_json)
            except ValueError:
                schema = None  # the verdict degrades to "not checked"

    url = _gateway_chat_url(request)
    # The new user turn rides in the payload but is persisted only after the
    # gateway accepts the request (IMP-04.6): a rejected send must not leave
    # a dangling turn.
    messages_payload = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in session.messages
    ]
    messages_payload.append({"role": "user", "content": message})
    payload: dict[str, Any] = {
        "model": status.active_alias,
        "messages": messages_payload,
        "stream": True,
        # vLLM answers with a final chunk carrying usage when asked; the
        # gateway passes the key through untouched (JEV-03.3).
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    transport = getattr(request.app.state, "gateway_transport", None)
    http = httpx.AsyncClient(timeout=GATEWAY_TIMEOUT, transport=transport)
    try:
        # `build_request` + `send(stream=True)` (the gateway proxy's pattern)
        # yields a plain `Response` that outlives this handler — the stream
        # generator below consumes and closes it.
        upstream = await http.send(
            http.build_request("POST", url, json=payload), stream=True
        )
    except httpx.HTTPError as exc:
        with suppress(Exception):
            await http.aclose()
        return redirect_with_flash(
            "/jev/chat",
            f"gateway unreachable — is `vibe serve` running? ({exc})",
        )
    if upstream.status_code != 200:
        # JEV-07.5: a gateway rejection (jev_incompatible_request,
        # jev_unavailable, upstream_error, controller-down) flashes back and
        # the user turn is NOT persisted.
        body = b""
        with suppress(Exception):
            body = await upstream.aread()
        with suppress(Exception):
            await upstream.aclose()
        with suppress(Exception):
            await http.aclose()
        return redirect_with_flash("/jev/chat", _gateway_error_message(body))
    # IMP-04.6: only now, after every check passed, does the user turn stick.
    session.append({"role": "user", "content": message})
    store.touch(session)
    jev_header = upstream.headers.get("x-vibe-jev", "")
    return _jev_chat_stream(
        request,
        session,
        status.active_alias,
        http,
        upstream,
        temperature,
        max_tokens,
        max_model_len,
        dropped,
        schema,
        schema_name,
        schema_sha,
        schema_stale,
        jev_header,
    )


@router.post("/jev/chat/reset", name="jev_chat_reset")
async def jev_chat_reset(request: Request) -> Response:
    """Drop the Jev-chat session and clear its cookie (JEV-07.6)."""
    store = _store_of(request)
    session_id = request.cookies.get(JEV_SESSION_COOKIE)
    if session_id:
        store.reset(session_id)
    response = redirect("/jev/chat")
    response.delete_cookie(
        JEV_SESSION_COOKIE, path="/", httponly=True, samesite="lax"
    )
    return response
