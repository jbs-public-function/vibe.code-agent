"""Admin Jev chat panel (improvements-2, doc 07 / JEV-07.9).

Mirrors `tests/test_admin_jev.py`: a duck-typed fake controller client drives
the composed FastAPI app through `httpx.ASGITransport`, and the gateway is
faked with the dashboard's `app.state.gateway_transport` hook — the handler
records the chat-completions payload and returns scripted SSE/error
responses. No GPU, Docker, or real HTTP is involved.
"""

from __future__ import annotations

import asyncio
import html
import json
from collections.abc import AsyncIterator
from typing import Any, cast, override

import httpx
import pytest
from fastapi import FastAPI

from vibe.admin.app import build_app
from vibe.client import ControllerClient, ControllerUnavailable, JevSchemaNotFound
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.jev import load_builtin

CSP = (
    "default-src 'self'; script-src 'none'; style-src 'self'; "
    "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)

SHA_A = "a" * 64
BUILTIN_DEFAULT_TEXT = json.dumps(load_builtin("default"))
VALID_JSON = '{"decision_type": "choice", "result": "YES", "confidence": 0.9}'
INVALID_JSON = '{"decision_type": "choice", "result": "MAYBE", "confidence": 0.9}'


# -- shared helpers ----------------------------------------------------------


def make_info(alias: str, *, max_len: int = 4096) -> control_pb2.ModelInfo:
    return control_pb2.ModelInfo(
        alias=alias,
        display_name=f"{alias} display",
        hf_repo=f"org/{alias}",
        quantization="awq",
        max_model_len=max_len,
        host_port=8020,
        state=control_pb2.AVAILABLE,
        cached_locally=True,
        tags=[],
    )


class FakeJevChatClient(ControllerClient):
    """Scriptable in-memory controller for the Jev chat panel's needs."""

    def __init__(self) -> None:
        super().__init__(Settings(vllm_image_tag="v0.11.2"))
        self.infos: list[control_pb2.ModelInfo] = []
        self.status_value: control_pb2.Status = control_pb2.Status()
        self.status_error: Exception | None = None
        self.active_jev_value: control_pb2.ActiveJev = control_pb2.ActiveJev()
        self.active_jev_error: Exception | None = None
        self.jev_schemas: dict[str, control_pb2.GetJevSchemaResponse] = {}
        self.closed = False

    @override
    async def list_models(self) -> list[control_pb2.ModelInfo]:
        return list(self.infos)

    @override
    async def status(self) -> control_pb2.Status:
        if self.status_error is not None:
            raise self.status_error
        return self.status_value

    @override
    async def get_active_jev(self) -> control_pb2.ActiveJev:
        if self.active_jev_error is not None:
            raise self.active_jev_error
        return self.active_jev_value

    @override
    async def get_jev_schema(self, name: str) -> control_pb2.GetJevSchemaResponse:
        if name not in self.jev_schemas:
            raise JevSchemaNotFound(f"no such Jev schema: {name}")
        return self.jev_schemas[name]

    @override
    async def close(self) -> None:
        self.closed = True
        await super().close()


def populated() -> FakeJevChatClient:
    """A fake with a READY model and the built-in `default` Jev attached."""
    fake = FakeJevChatClient()
    fake.infos = [make_info("qwen2.5-coder-7b")]
    fake.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8020,
        container_id="c" * 12,
    )
    fake.active_jev_value = control_pb2.ActiveJev(
        info=control_pb2.JevInfo(
            schema_name="default",
            sha256=SHA_A,
            source="builtin",
            applied_at_unix_ms=1700000000000,
            stale=False,
            title="JevDecisionOutput",
        )
    )
    fake.jev_schemas["default"] = control_pb2.GetJevSchemaResponse(
        name="default",
        source="builtin",
        sha256=SHA_A,
        size_bytes=len(BUILTIN_DEFAULT_TEXT.encode()),
        mtime_unix_ms=0,
        title="JevDecisionOutput",
        schema_json=BUILTIN_DEFAULT_TEXT,
    )
    return fake


def sse_response(
    deltas: list[str],
    *,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
    jev_header: str = f"default; sha={SHA_A[:12]}",
    first_token_delay_s: float = 0.0,
) -> httpx.Response:
    """A gateway-style SSE chat-completions response.

    `first_token_delay_s` delays the *body* (not the handler — the mock
    transport runs handlers on the event loop), so the panel's first-token
    heartbeat can be exercised.
    """
    lines = []
    for delta in deltas:
        lines.append(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {"delta": {"content": delta}, "finish_reason": None}
                    ],
                    "usage": None,
                }
            )
        )
    lines.append(
        "data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": finish_reason}],
                "usage": usage or {"prompt_tokens": 12, "completion_tokens": 8},
            }
        )
    )
    lines.append("data: [DONE]")
    body = ("\n\n".join(lines) + "\n\n").encode()
    stream: httpx.AsyncByteStream | None = None
    if first_token_delay_s:
        stream = _DelayedStream(body, first_token_delay_s)
    return httpx.Response(
        200,
        headers={
            "content-type": "text/event-stream",
            "x-vibe-jev": jev_header,
        },
        content=None if stream else body,
        stream=stream,
    )


class _DelayedStream(httpx.AsyncByteStream):
    """An SSE body whose first read is delayed (slow first token)."""

    def __init__(self, body: bytes, delay_s: float) -> None:
        self._body = body
        self._delay_s = delay_s

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.sleep(self._delay_s)
        yield self._body


def make_app(
    fake: FakeJevChatClient,
    gateway_script: list[Any] | None = None,
    gateway_calls: list[tuple[str, bytes]] | None = None,
) -> FastAPI:
    """Compose the app with a scripted gateway (the dashboard's hook)."""
    app = build_app(Settings(vllm_image_tag="v0.11.2", vibe_bind="test"), fake)
    script = list(gateway_script or [])
    calls: list[tuple[str, bytes]] = gateway_calls if gateway_calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.read()))
        if script:
            item = script.pop(0)
            if isinstance(item, Exception):
                raise item
            return cast(httpx.Response, item)
        return httpx.Response(200, text="ok")

    app.state.gateway_transport = httpx.MockTransport(handler)
    return app


def make_transport(fake: FakeJevChatClient, **kwargs: Any) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=make_app(fake, **kwargs))


# -- JEV-07.1: page ----------------------------------------------------------


async def test_page_renders_active_schema_and_send_form() -> None:
    fake = populated()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/jev/chat")
    assert response.status_code == 200
    text = response.text
    assert "Jev chat" in text
    assert "Active Jev" in text and "default" in text and "built-in" in text
    assert SHA_A[:12] in text
    assert 'action="/jev/chat"' in text and 'name="message"' in text
    assert 'action="/jev/chat/reset"' in text
    assert 'value="0.7"' in text and 'value="512"' in text
    assert "Gateway unreachable" not in text
    # The panel sets its own cookie, not the plain chat one.
    assert "vibe_jev_session" in response.headers.get("set-cookie", "")


async def test_page_without_jev_shows_unconstrained_banner() -> None:
    fake = populated()
    fake.active_jev_value = control_pb2.ActiveJev()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/jev/chat")
    assert response.status_code == 200
    assert "unconstrained" in response.text


async def test_page_gateway_down_shows_banner_not_500() -> None:
    fake = populated()
    transport = make_transport(fake, gateway_script=[httpx.ConnectError("refused")])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/jev/chat")
    assert response.status_code == 200
    assert "Gateway unreachable" in response.text


async def test_page_controller_down_banner_not_500() -> None:
    fake = populated()
    fake.status_error = ControllerUnavailable("gone")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/jev/chat")
    assert response.status_code == 200
    assert "controller unreachable" in response.text.lower()


# -- JEV-07.2/07.4: stream + verdict ------------------------------------------


async def test_stream_assembles_valid_json_and_renders_verdict() -> None:
    fake = populated()
    calls: list[tuple[str, bytes]] = []
    transport = make_transport(
        fake, gateway_script=[sse_response([VALID_JSON])], gateway_calls=calls
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "Should I go?"})
    assert response.status_code == 200
    text = response.text
    # The user turn and the streamed (escaped) reply are on the page.
    assert "Should I go?" in text
    assert "decision_type" in text
    # Verdict: schema-valid, finish_reason, usage, enforced-by header.
    assert "schema-valid" in text
    assert "stop" in text
    assert "prompt 12" in text and "completion 8" in text
    assert f"default; sha={SHA_A[:12]}" in text
    # Payload: through the gateway, no response_format (the gateway injects).
    assert len(calls) == 1
    path, body = calls[0]
    assert path == "/v1/chat/completions"
    payload = json.loads(body)
    assert payload["model"] == "qwen2.5-coder-7b"
    assert payload["messages"] == [{"role": "user", "content": "Should I go?"}]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "response_format" not in payload


async def test_stream_invalid_json_shows_schema_errors() -> None:
    fake = populated()
    transport = make_transport(fake, gateway_script=[sse_response([INVALID_JSON])])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
    assert response.status_code == 200
    text = response.text
    assert "invalid" in text
    assert "Schema errors" in text
    assert "MAYBE" in text  # the offending value appears in the error detail


async def test_stream_truncation_warns_on_finish_reason_length() -> None:
    fake = populated()
    transport = make_transport(
        fake,
        gateway_script=[sse_response([VALID_JSON], finish_reason="length")],
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
    assert response.status_code == 200
    assert "truncated at max_tokens" in response.text
    assert "length" in response.text


async def test_stream_plain_mode_pretty_prints_without_validity_badge() -> None:
    fake = populated()
    fake.active_jev_value = control_pb2.ActiveJev()
    fenced = '```json\n{"answer": "yes"}\n```'
    transport = make_transport(fake, gateway_script=[sse_response([fenced], jev_header="")])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
    assert response.status_code == 200
    text = response.text
    assert "no schema — unchecked" in text
    assert "answer" in text  # fence-stripped and pretty-printed
    assert "unconstrained" in text


async def test_stream_escapes_html_from_the_model() -> None:
    fake = populated()
    evil = '{"decision_type": "choice", "result": "<script>alert(1)</script>", "confidence": 0.5}'
    transport = make_transport(fake, gateway_script=[sse_response([evil])])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
    assert response.status_code == 200
    text = response.text
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


async def test_stream_heartbeat_during_slow_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibe.admin.routes import jev_chat

    monkeypatch.setattr(jev_chat, "CHAT_HEARTBEAT_S", 0.05)
    fake = populated()
    transport = make_transport(
        fake,
        gateway_script=[sse_response([VALID_JSON], first_token_delay_s=0.2)],
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
    assert response.status_code == 200
    assert "<!-- keep-alive -->" in response.text
    assert "schema-valid" in response.text


async def test_clamps_temperature_and_max_tokens() -> None:
    fake = populated()
    fake.infos = [make_info("qwen2.5-coder-7b", max_len=4096)]
    calls: list[tuple[str, bytes]] = []
    transport = make_transport(
        fake,
        gateway_script=[sse_response([VALID_JSON])],
        gateway_calls=calls,
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/jev/chat",
            data={"message": "hi", "temperature": "9", "max_tokens": "999999"},
        )
    assert response.status_code == 200
    payload = json.loads(calls[0][1])
    assert payload["temperature"] == 2.0
    assert payload["max_tokens"] == 4096


# -- JEV-07.5: gateway failures -----------------------------------------------


async def test_gateway_400_flashes_and_does_not_persist_turn() -> None:
    fake = populated()
    rejection = httpx.Response(
        400,
        json={"message": "Jev is active (default); tool calling is disabled."},
    )
    transport = make_transport(fake, gateway_script=[rejection])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
        assert response.status_code == 303
        assert response.headers["location"] == "/jev/chat"
        page = await client.get("/jev/chat")
    assert "tool calling is disabled" in page.text
    assert "No messages yet." in page.text  # the user turn did not stick


async def test_gateway_unreachable_flashes() -> None:
    fake = populated()
    transport = make_transport(fake, gateway_script=[httpx.ConnectError("refused")])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
        assert response.status_code == 303
        page = await client.get(response.headers["location"])
    assert "gateway unreachable" in page.text
    assert "No messages yet." in page.text


async def test_not_ready_flashes() -> None:
    fake = populated()
    fake.status_value = control_pb2.Status(active_alias="", state=control_pb2.AVAILABLE)
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
        assert response.status_code == 303
        page = await client.get(response.headers["location"])
    assert "no model loaded" in page.text


# -- JEV-07.6: sessions --------------------------------------------------------


async def test_reset_clears_the_jev_conversation() -> None:
    fake = populated()
    transport = make_transport(
        fake,
        gateway_script=[
            sse_response([VALID_JSON]),
            httpx.Response(200, text="ok"),  # the probe on the reloaded page
        ],
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/jev/chat", data={"message": "hello"})
        reset = await client.post("/jev/chat/reset", follow_redirects=False)
        assert reset.status_code == 303
        assert reset.headers["location"] == "/jev/chat"
        assert "vibe_jev_session" in reset.headers.get("set-cookie", "")
        reloaded = await client.get("/jev/chat")
    assert "No messages yet." in reloaded.text
    assert "hello" not in reloaded.text


async def test_jev_session_cookie_is_distinct_from_plain_chat() -> None:
    fake = populated()
    transport = make_transport(fake, gateway_script=[sse_response([VALID_JSON])])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jev/chat", data={"message": "hi"})
    set_cookie = response.headers.get("set-cookie", "")
    assert "vibe_jev_session" in set_cookie
    assert "vibe_session=" not in set_cookie


# -- JEV-07.7: zero-JS / CSP / origin ------------------------------------------


async def test_jev_chat_pages_are_script_free_and_carry_csp() -> None:
    fake = populated()
    transport = make_transport(fake, gateway_script=[sse_response([VALID_JSON])])
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/jev/chat")
        stream = await client.post("/jev/chat", data={"message": "hi"})
    for response in (page, stream):
        assert response.status_code == 200
        assert "<script" not in response.text.lower()
        assert response.headers["content-security-policy"] == CSP


async def test_cross_origin_post_rejected() -> None:
    fake = populated()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/jev/chat",
            data={"message": "hi"},
            headers={"Origin": "http://evil.example"},
        )
    assert response.status_code == 403


async def test_head_jev_chat_200_empty_body() -> None:
    fake = populated()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.head("/jev/chat")
    assert response.status_code == 200
    assert response.content == b""
