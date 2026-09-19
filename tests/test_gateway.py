"""Gateway tests (Phase 2): models payload, router decision table, streaming
proxy, swap coordinator, and the composed Starlette app.

No test touches docker or the network: the controller is a duck-typed
`FakeControllerClient` and the vLLM upstream is an `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from typing import Any, cast, override

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request

from vibe.client import ControllerClient, ControllerUnavailable
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway import app as gateway_app
from vibe.gateway.app import build_app
from vibe.gateway.models import build_models_payload, state_name
from vibe.gateway.proxy import UpstreamError, stream_proxy
from vibe.gateway.router import Proxy, Retry, Swap, decide
from vibe.gateway.swap import SwapConflict, SwapCoordinator, SwapFailed, SwapTimeout
from vibe.jobs import JobInProgress

KNOWN_ALIASES = ["qwen2.5-coder-7b", "llama-3.1-8b", "deepseek-r1-14b"]


# -- shared helpers --------------------------------------------------------


def make_info(
    alias: str,
    *,
    max_len: int = 1024,
    state: control_pb2.ModelState = control_pb2.AVAILABLE,
    cached: bool = True,
) -> control_pb2.ModelInfo:
    return control_pb2.ModelInfo(
        alias=alias,
        display_name=alias.upper(),
        max_model_len=max_len,
        state=state,
        cached_locally=cached,
    )


def make_status(
    active_alias: str = "",
    *,
    state: control_pb2.ModelState = control_pb2.AVAILABLE,
    host_port: int = 0,
) -> control_pb2.Status:
    return control_pb2.Status(
        active_alias=active_alias, state=state, host_port=host_port
    )


def make_event(
    state: control_pb2.ModelState, *, terminal: bool = True, message: str = ""
) -> control_pb2.JobEvent:
    return control_pb2.JobEvent(state=state, terminal=terminal, message=message)


def registry_infos(valid_registry_dict: dict[str, Any]) -> list[control_pb2.ModelInfo]:
    """ModelInfo list mirroring the repo's three `models.yaml` entries."""
    return [
        control_pb2.ModelInfo(
            alias=model["alias"],
            display_name=model["display_name"],
            max_model_len=model["max_model_len"],
            state=control_pb2.AVAILABLE,
            cached_locally=True,
        )
        for model in valid_registry_dict["models"]
    ]


class FakeControllerClient(ControllerClient):
    """Scriptable stand-in for `ControllerClient`.

    Subclasses the real client so tests can pass it where `ControllerClient`
    is annotated, but never calls `super().__init__` (no gRPC channel).
    `load_script` mixes `JobEvent`s (yielded in order) and floats (sleep that
    many seconds before continuing); `status_script` is consumed first by
    `status()` so tests can script successive snapshots.
    """

    def __init__(self) -> None:
        super().__init__(Settings(vllm_image_tag="v0.11.2"))
        self.infos: list[control_pb2.ModelInfo] = []
        self.status_value: control_pb2.Status = control_pb2.Status()
        self.status_script: deque[control_pb2.Status] = deque()
        self.load_script: list[control_pb2.JobEvent | float] = []
        self.load_calls: list[str] = []
        self.load_error: Exception | None = None
        self.status_error: Exception | None = None
        self.list_error: Exception | None = None
        self.list_calls: int = 0
        self.status_calls: int = 0
        self.closed: bool = False

    @override
    async def list_models(self) -> list[control_pb2.ModelInfo]:
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return list(self.infos)

    @override
    async def status(self) -> control_pb2.Status:
        self.status_calls += 1
        if self.status_error is not None:
            raise self.status_error
        if self.status_script:
            return self.status_script.popleft()
        return self.status_value

    @override
    async def load(
        self,
        alias: str,
        force: bool = False,
        keep_failed: bool = False,
        timeout_s: int | None = None,
    ) -> AsyncIterator[control_pb2.JobEvent]:
        self.load_calls.append(alias)
        if self.load_error is not None:
            raise self.load_error
        for item in self.load_script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            event = cast(control_pb2.JobEvent, item)
            yield event

    @override
    async def close(self) -> None:
        self.closed = True
        await super().close()


class _ChunkedStream(httpx.AsyncByteStream):
    """Async byte stream that replays a fixed list of chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks: list[bytes] = chunks
        self.aclosed = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    @override
    async def aclose(self) -> None:
        self.aclosed = True


def make_request(
    method: str = "POST",
    path: str = "/v1/chat/completions",
    query: str = "",
    headers: dict[str, str] | None = None,
) -> Request:
    """A minimal Starlette `Request` for exercising `stream_proxy` directly."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "headers": [
            (name.lower().encode(), value.encode())
            for name, value in (headers or {}).items()
        ],
        "scheme": "http",
        "server": ("127.0.0.1", 8080),
        "client": ("127.0.0.1", 12345),
        "root_path": "",
    }
    return Request(scope)


# -- models ----------------------------------------------------------------


def test_state_name_maps_unset_to_available() -> None:
    assert state_name(control_pb2.MODEL_STATE_UNSPECIFIED) == "AVAILABLE"
    assert state_name(control_pb2.AVAILABLE) == "AVAILABLE"
    assert state_name(control_pb2.LOADING) == "LOADING"
    assert state_name(control_pb2.READY) == "READY"
    assert state_name(control_pb2.UNLOADING) == "UNLOADING"
    assert state_name(control_pb2.ERROR) == "ERROR"


def test_build_models_payload_shape(valid_registry_dict: dict[str, Any]) -> None:
    infos = registry_infos(valid_registry_dict)
    status = make_status(
        active_alias="llama-3.1-8b",
        state=control_pb2.READY,
        host_port=8020,
    )
    payload = build_models_payload(infos, status)
    assert payload["object"] == "list"
    data = payload["data"]
    assert isinstance(data, list)
    assert [entry["id"] for entry in data] == KNOWN_ALIASES
    for entry in data:
        assert entry["object"] == "model"
        assert entry["created"] == 0
        assert entry["owned_by"] == "vibe"
    # Exactly one active entry.
    active = [entry for entry in data if entry["vibe_active"]]
    assert len(active) == 1
    assert active[0]["id"] == "llama-3.1-8b"
    qwen = data[0]
    assert qwen["vibe_display_name"] == "Qwen2.5 Coder 7B Instruct (AWQ)"
    assert qwen["vibe_max_model_len"] == 32768
    assert qwen["vibe_state"] == "AVAILABLE"
    assert qwen["vibe_cached"] is True


def test_build_models_payload_status_none(
    valid_registry_dict: dict[str, Any],
) -> None:
    payload = build_models_payload(registry_infos(valid_registry_dict), None)
    assert payload["object"] == "list"
    data = payload["data"]
    assert isinstance(data, list)
    for entry in data:
        assert entry["vibe_active"] is False
        assert entry["vibe_state"] == "AVAILABLE"


# -- router ----------------------------------------------------------------


def test_decide_missing_model() -> None:
    status = make_status()
    bad_values: list[Any] = [None, "", 123, ["a"], {}]
    for bad in bad_values:
        decision = decide(bad, KNOWN_ALIASES, status, False, None)
        assert decision == Retry(
            400,
            0,
            {
                "message": "missing 'model' field",
                "type": "invalid_request_error",
                "code": None,
            },
        )


def test_decide_unknown_model() -> None:
    status = make_status(active_alias="qwen2.5-coder-7b")
    expected = Retry(
        404,
        0,
        {
            "message": (
                "model 'nope' not found; valid aliases: "
                "qwen2.5-coder-7b, llama-3.1-8b, deepseek-r1-14b"
            ),
            "type": "model_not_found",
            "code": "model_not_found",
        },
    )
    # Unknown is 404 regardless of the swap setting.
    assert decide("nope", KNOWN_ALIASES, status, False, None) == expected
    assert decide("nope", KNOWN_ALIASES, status, True, None) == expected


def test_decide_active_ready_proxies() -> None:
    status = make_status(
        active_alias="llama-3.1-8b",
        state=control_pb2.READY,
        host_port=8020,
    )
    decision = decide("llama-3.1-8b", KNOWN_ALIASES, status, False, None)
    assert decision == Proxy("http://127.0.0.1:8020")


def test_decide_active_but_busy_retries_quickly() -> None:
    for state in (
        control_pb2.LOADING,
        control_pb2.UNLOADING,
        control_pb2.AVAILABLE,
        control_pb2.MODEL_STATE_UNSPECIFIED,
    ):
        status = make_status(
            active_alias="llama-3.1-8b", state=state, host_port=8020
        )
        decision = decide("llama-3.1-8b", KNOWN_ALIASES, status, False, None)
        assert decision == Retry(
            503,
            10,
            {
                "message": (
                    f"model 'llama-3.1-8b' is {state_name(state)}; retry shortly"
                ),
                "type": "model_not_ready",
                "code": "model_not_ready",
            },
        )
    # READY with port 0 is not proxyable either.
    status = make_status(
        active_alias="llama-3.1-8b", state=control_pb2.READY, host_port=0
    )
    decision = decide("llama-3.1-8b", KNOWN_ALIASES, status, False, None)
    assert isinstance(decision, Retry)
    assert decision.status == 503 and decision.after == 10


def test_decide_active_error_reports_and_hints() -> None:
    """IMP-03.13: ERROR is terminal — Retry-After 30, `last_error` + hint."""
    status = make_status(active_alias="llama-3.1-8b", state=control_pb2.ERROR)
    status.last_error = "container died during startup"
    decision = decide("llama-3.1-8b", KNOWN_ALIASES, status, False, None)
    assert decision == Retry(
        503,
        30,
        {
            "message": (
                "model 'llama-3.1-8b' is in ERROR: container died during "
                "startup. Run: vibe load llama-3.1-8b"
            ),
            "type": "model_not_ready",
            "code": "model_not_ready",
        },
    )
    # Without a last_error the hint still appears.
    decision = decide(
        "llama-3.1-8b",
        KNOWN_ALIASES,
        make_status(active_alias="llama-3.1-8b", state=control_pb2.ERROR),
        False,
        None,
    )
    assert isinstance(decision, Retry)
    assert decision.after == 30
    assert "Run: vibe load llama-3.1-8b" in (decision.error["message"] or "")


def test_decide_active_error_swappable_when_swap_on() -> None:
    """IMP-03.13: with swap-on-request, ERROR is treated as swappable."""
    status = make_status(active_alias="llama-3.1-8b", state=control_pb2.ERROR)
    decision = decide("llama-3.1-8b", KNOWN_ALIASES, status, True, None)
    assert decision == Swap("llama-3.1-8b")


def test_decide_not_loaded_swap_off() -> None:
    status = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    decision = decide("llama-3.1-8b", KNOWN_ALIASES, status, False, None)
    assert decision == Retry(
        503,
        30,
        {
            "message": (
                "model llama-3.1-8b is not loaded (active: qwen2.5-coder-7b). "
                "Run: vibe load llama-3.1-8b"
            ),
            "type": "model_not_loaded",
            "code": "model_not_loaded",
        },
    )


def test_decide_not_loaded_no_active_model() -> None:
    decision = decide("llama-3.1-8b", KNOWN_ALIASES, make_status(), False, None)
    assert decision == Retry(
        503,
        30,
        {
            "message": "no model loaded. Run: vibe load llama-3.1-8b",
            "type": "model_not_loaded",
            "code": "model_not_loaded",
        },
    )


def test_decide_swap_on_starts_or_attaches() -> None:
    status = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    # No job in flight: start a fresh swap.
    assert decide("llama-3.1-8b", KNOWN_ALIASES, status, True, None) == Swap(
        "llama-3.1-8b"
    )
    # Job for the same alias: attach.
    assert decide(
        "llama-3.1-8b", KNOWN_ALIASES, status, True, "llama-3.1-8b"
    ) == Swap("llama-3.1-8b")


def test_decide_swap_conflict() -> None:
    status = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    decision = decide(
        "llama-3.1-8b", KNOWN_ALIASES, status, True, "deepseek-r1-14b"
    )
    assert decision == Retry(
        503,
        30,
        {
            "message": "swap to 'deepseek-r1-14b' is in progress",
            "type": "swap_in_progress",
            "code": "swap_in_progress",
        },
    )


# -- proxy -----------------------------------------------------------------

UPSTREAM_BYTES = (
    b'data: {"delta":"Hi"}\n\ndata: {"delta":" there"}\n\ndata: [DONE]\n\n'
)


async def test_stream_proxy_relays_sse_byte_for_byte() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["content"] = request.content
        captured["headers"] = dict(request.headers)
        chunks = [UPSTREAM_BYTES[:10], UPSTREAM_BYTES[10:40], UPSTREAM_BYTES[40:]]
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "connection": "keep-alive",
                "transfer-encoding": "chunked",
                "keep-alive": "timeout=5",
                "x-extra": "kept",
            },
            stream=_ChunkedStream(chunks),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    request = make_request(
        headers={
            "content-type": "application/json",
            "authorization": "Bearer test-key",
            "connection": "close, x-hop",
            "x-hop": "drop-me",
            "keep-alive": "timeout=5",
            "proxy-authorization": "Basic dXNlcjpwdw==",
            "x-keep": "yes",
        }
    )
    body = b'{"model": "llama-3.1-8b", "stream": true}'
    try:
        response = await stream_proxy(
            request, "http://127.0.0.1:8020", body, client
        )
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/event-stream"
        assert response.headers.get("x-extra") == "kept"
        for hop in ("connection", "keep-alive", "transfer-encoding"):
            assert response.headers.get(hop) is None
        relayed = b""
        async for chunk in response.body_iterator:
            assert isinstance(chunk, bytes)
            relayed += chunk
        assert relayed == UPSTREAM_BYTES
    finally:
        await client.aclose()

    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8020/v1/chat/completions"
    assert captured["content"] == body
    forwarded = {name.lower(): value for name, value in captured["headers"].items()}
    assert forwarded["authorization"] == "Bearer test-key"
    assert forwarded["content-type"] == "application/json"
    assert forwarded["x-keep"] == "yes"
    # httpx adds its own `Connection: keep-alive` transport default; the
    # client-supplied value (and the header it names) must not leak through.
    assert forwarded.get("connection") != "close, x-hop"
    for hop in (
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-hop",
    ):
        assert hop not in forwarded


async def test_stream_proxy_forwards_query_string() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, stream=_ChunkedStream([b"ok"]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await stream_proxy(
            make_request(path="/v1/chat/completions", query="foo=bar&x=1"),
            "http://127.0.0.1:8020",
            b"{}",
            client,
        )
        async for _ in response.body_iterator:
            pass
    finally:
        await client.aclose()
    assert captured["url"] == "http://127.0.0.1:8020/v1/chat/completions?foo=bar&x=1"


async def test_stream_proxy_relays_non_2xx_verbatim() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            headers={"content-type": "application/json"},
            stream=_ChunkedStream([b'{"error": "busy"}']),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await stream_proxy(
            make_request(), "http://127.0.0.1:8020", b"{}", client
        )
        assert response.status_code == 503
        relayed = b""
        async for chunk in response.body_iterator:
            assert isinstance(chunk, bytes)
            relayed += chunk
        assert relayed == b'{"error": "busy"}'
    finally:
        await client.aclose()


async def test_stream_proxy_connect_error_raises_upstream_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstreamError):
            await stream_proxy(
                make_request(), "http://127.0.0.1:8020", b"{}", client
            )
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError("read failed"),
        httpx.WriteError("write failed"),
        httpx.CloseError("close failed"),
        httpx.ProxyError("proxy failed"),
    ],
)
async def test_stream_proxy_transport_errors_raise_upstream_error(
    error: httpx.TransportError,
) -> None:
    """IMP-03.9: every httpx transport failure maps to `UpstreamError`."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        raise error

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstreamError):
            await stream_proxy(
                make_request(), "http://127.0.0.1:8020", b"{}", client
            )
    finally:
        await client.aclose()


async def test_stream_proxy_preserves_duplicate_response_headers() -> None:
    """IMP-03.12: repeated upstream headers (set-cookie) are not merged."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[
                ("content-type", "application/json"),
                ("set-cookie", "a=1"),
                ("set-cookie", "b=2"),
            ],
            stream=_ChunkedStream([b"ok"]),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await stream_proxy(
            make_request(), "http://127.0.0.1:8020", b"{}", client
        )
        cookies = response.headers.getlist("set-cookie")
        assert cookies == ["a=1", "b=2"]
        async for _ in response.body_iterator:
            pass
    finally:
        await client.aclose()


async def test_stream_proxy_drops_host_and_sets_x_forwarded_host() -> None:
    """IMP-03.12: the client's Host is not forwarded; it becomes X-Forwarded-Host."""
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, stream=_ChunkedStream([b"ok"]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await stream_proxy(
            make_request(headers={"host": "gateway.example:8080"}),
            "http://127.0.0.1:8020",
            b"{}",
            client,
        )
        async for _ in response.body_iterator:
            pass
    finally:
        await client.aclose()
    forwarded = {name.lower(): value for name, value in captured["headers"].items()}
    assert forwarded["x-forwarded-host"] == "gateway.example:8080"
    assert forwarded["host"] == "127.0.0.1:8020"  # httpx's own target host


async def test_stream_proxy_client_disconnect_closes_upstream() -> None:
    """A client going away mid-SSE closes the upstream stream."""
    stream = _ChunkedStream([b"chunk1", b"chunk2"])

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await stream_proxy(
            make_request(), "http://127.0.0.1:8020", b"{}", client
        )
        iterator = cast(Any, response.body_iterator)
        first = await anext(iterator)
        assert first == b"chunk1"
        # The client went away mid-stream: closing the iterator must close
        # the upstream httpx stream.
        await iterator.aclose()
        assert stream.aclosed is True
    finally:
        await client.aclose()


# -- swap ------------------------------------------------------------------


async def test_swap_concurrent_same_alias_attach_one_job() -> None:
    client = FakeControllerClient()
    client.load_script = [
        make_event(control_pb2.LOADING, terminal=False),
        0.05,
        make_event(control_pb2.READY, terminal=True),
    ]
    client.status_value = make_status(
        active_alias="m", state=control_pb2.READY, host_port=8030
    )
    coordinator = SwapCoordinator(client, timeout_s=5.0)
    port1, port2 = await asyncio.gather(
        coordinator.ensure_ready("m"), coordinator.ensure_ready("m")
    )
    assert port1 == port2 == 8030
    assert client.load_calls == ["m"]  # exactly one LoadModel
    assert coordinator.in_flight_alias is None


async def test_swap_different_alias_conflicts_immediately() -> None:
    client = FakeControllerClient()
    client.load_script = [
        make_event(control_pb2.LOADING, terminal=False),
        0.2,
        make_event(control_pb2.READY, terminal=True),
    ]
    client.status_value = make_status(
        active_alias="a", state=control_pb2.READY, host_port=8030
    )
    coordinator = SwapCoordinator(client, timeout_s=5.0)
    first = asyncio.create_task(coordinator.ensure_ready("a"))
    await asyncio.sleep(0)  # let the first call start its job
    assert coordinator.in_flight_alias == "a"
    with pytest.raises(SwapConflict):
        await coordinator.ensure_ready("b")
    await asyncio.sleep(0)  # let the background job reach the controller
    assert client.load_calls == ["a"]  # the conflict never called the controller
    assert await first == 8030


async def test_swap_timeout_keeps_job_running() -> None:
    client = FakeControllerClient()
    client.load_script = [
        make_event(control_pb2.LOADING, terminal=False),
        0.2,
        make_event(control_pb2.READY, terminal=True),
    ]
    client.status_value = make_status(
        active_alias="m", state=control_pb2.READY, host_port=8030
    )
    coordinator = SwapCoordinator(client, timeout_s=0.05)
    with pytest.raises(SwapTimeout) as excinfo:
        await coordinator.ensure_ready("m")
    assert "0.05" in str(excinfo.value)
    # The background job is untouched by the timed-out caller.
    assert coordinator.in_flight_alias == "m"
    assert client.load_calls == ["m"]
    await asyncio.sleep(0.3)  # the job finishes on its own
    assert coordinator.in_flight_alias is None
    # A later request starts a fresh load and succeeds.
    client.load_script = [make_event(control_pb2.READY, terminal=True)]
    assert await coordinator.ensure_ready("m") == 8030
    assert client.load_calls == ["m", "m"]


async def test_swap_terminal_error_fails() -> None:
    client = FakeControllerClient()
    client.load_script = [
        make_event(control_pb2.ERROR, terminal=True, message="boom")
    ]
    coordinator = SwapCoordinator(client, timeout_s=5.0)
    with pytest.raises(SwapFailed) as excinfo:
        await coordinator.ensure_ready("m")
    assert "boom" in str(excinfo.value)
    assert coordinator.in_flight_alias is None
    assert client.load_calls == ["m"]


async def test_swap_post_check_failure_fails() -> None:
    client = FakeControllerClient()
    client.load_script = [make_event(control_pb2.READY, terminal=True)]
    # The load claims READY but the controller disagrees afterwards.
    client.status_value = make_status(active_alias="other")
    coordinator = SwapCoordinator(client, timeout_s=5.0)
    with pytest.raises(SwapFailed) as excinfo:
        await coordinator.ensure_ready("m")
    assert "active=other" in str(excinfo.value)
    assert coordinator.in_flight_alias is None


async def test_swap_controller_error_fails() -> None:
    client = FakeControllerClient()
    client.load_error = RuntimeError("controller exploded")
    coordinator = SwapCoordinator(client, timeout_s=5.0)
    with pytest.raises(SwapFailed) as excinfo:
        await coordinator.ensure_ready("m")
    assert "controller exploded" in str(excinfo.value)
    assert coordinator.in_flight_alias is None


# -- app -------------------------------------------------------------------


class UpstreamRecorder:
    """`httpx.MockTransport` handler factory for scripted upstream responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status: int = 200
        self.headers: dict[str, str] = {"content-type": "application/json"}
        self.chunks: list[bytes] | None = None  # None → the default body
        self.default_body: bytes = b'{"ok": true}'
        self.raise_connect_error: bool = False
        self.error: Exception | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_connect_error:
            raise httpx.ConnectError("connection refused")
        if self.error is not None:
            raise self.error
        return httpx.Response(
            self.status,
            headers=self.headers,
            stream=_ChunkedStream(self.chunks or [self.default_body]),
        )


@pytest.fixture
def upstream() -> UpstreamRecorder:
    return UpstreamRecorder()


def build_test_app(
    settings: Settings, client: FakeControllerClient, upstream: UpstreamRecorder
) -> Starlette:
    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle))
    return build_app(settings, client=client, http=http)


def asgi_client(app: Starlette) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_app_models_lists_registry(
    settings: Settings, valid_registry_dict: dict[str, Any], upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = registry_infos(valid_registry_dict)
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [entry["id"] for entry in body["data"]] == KNOWN_ALIASES
    assert sum(1 for entry in body["data"] if entry["vibe_active"]) == 1


async def test_app_models_controller_down_503(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.list_error = ControllerUnavailable("controller unreachable")
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.get("/v1/models")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    body = response.json()
    assert body["error"] == {
        "message": "controller unreachable",
        "type": "controller_unavailable",
        "code": "controller_unavailable",
    }


async def test_app_posts_proxied_when_ready(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    app = build_test_app(settings, client, upstream)
    payload = (
        b'{"model": "qwen2.5-coder-7b", '
        b'"messages": [{"role": "user", "content": "hi"}]}'
    )
    async with asgi_client(app) as ac:
        for path in (
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
        ):
            response = await ac.post(
                path,
                content=payload,
                headers={
                    "content-type": "application/json",
                    "authorization": "Bearer zed",
                },
            )
            assert response.status_code == 200
            assert response.json() == {"ok": True}
    assert len(upstream.requests) == 3
    for sent, path in zip(
        upstream.requests,
        ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"),
    ):
        assert sent.method == "POST"
        assert sent.url.path == path
        assert str(sent.url) == f"http://127.0.0.1:8030{path}"
        assert sent.content == payload
        assert sent.headers["authorization"] == "Bearer zed"


async def test_app_not_loaded_swap_off_503(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b"), make_info("llama-3.1-8b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"model": "llama-3.1-8b"}
        )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    body = response.json()
    assert body["error"]["type"] == "model_not_loaded"
    assert "vibe load llama-3.1-8b" in body["error"]["message"]
    assert upstream.requests == []  # nothing reached vLLM


async def test_app_unknown_model_404(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"model": "nope"}
        )
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["type"] == "model_not_found"
    assert "valid aliases: qwen2.5-coder-7b" in body["error"]["message"]


async def test_app_missing_model_400(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"messages": []}
        )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "missing 'model' field",
        "type": "invalid_request_error",
        "code": None,
    }


async def test_app_invalid_json_400(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", content=b"not json"
        )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "request body must be valid JSON"


async def test_app_body_too_large_413(
    settings: Settings,
    upstream: UpstreamRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gateway_app, "BODY_LIMIT", 8)
    client = FakeControllerClient()
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", content=b'{"model": "qwen2.5-coder-7b"}' * 4
        )
    assert response.status_code == 413
    assert response.json()["error"] == {
        "message": "request body too large",
        "type": "invalid_request_error",
        "code": None,
    }
    assert client.list_calls == 0  # rejected before touching the controller


async def test_app_health_ok(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "gateway": "ok",
        "controller": "ok",
        "active": "qwen2.5-coder-7b",
        "state": "READY",
    }


async def test_app_health_controller_down(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.list_error = ControllerUnavailable("controller unreachable")
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.get("/health")
    assert response.status_code == 200  # /health is never a 503 (FR-03.10)
    # IMP-03.10: the controller's state is unknown while it is down.
    assert response.json() == {
        "gateway": "ok",
        "controller": "down",
        "active": None,
        "state": None,
    }


async def test_app_unknown_path_404(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.get("/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"] == {
        "message": "Unknown request URL: /v1/nope.",
        "type": "invalid_request_error",
        "code": None,
    }


async def test_app_sse_relay_end_to_end(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    upstream.headers = {"content-type": "text/event-stream"}
    upstream.chunks = [b'data: {"delta":"Hi"}\n\n', b"data: [DONE]\n\n"]
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5-coder-7b", "stream": True},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == b'data: {"delta":"Hi"}\n\ndata: [DONE]\n\n'


async def test_app_upstream_error_502_invalidates_cache(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    upstream.raise_connect_error = True
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        first = await ac.post(
            "/v1/chat/completions", json={"model": "qwen2.5-coder-7b"}
        )
        second = await ac.post(
            "/v1/chat/completions", json={"model": "qwen2.5-coder-7b"}
        )
    assert first.status_code == 502
    assert first.json()["error"]["type"] == "upstream_error"
    assert second.status_code == 502
    # FR-03.9: the 502 invalidated the snapshot, so the second request
    # re-asked the controller instead of reusing the cached routing.
    assert client.list_calls == 2


async def test_app_swap_on_request_proxies_after_load(
    upstream: UpstreamRecorder,
) -> None:
    swap_settings = Settings(
        vllm_image_tag="v0.11.2", vibe_swap_on_request=True, vibe_swap_timeout=10
    )
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b"), make_info("llama-3.1-8b")]
    client.status_script.append(
        make_status(
            active_alias="qwen2.5-coder-7b",
            state=control_pb2.READY,
            host_port=8030,
        )
    )  # routing snapshot
    client.status_script.append(
        make_status(
            active_alias="llama-3.1-8b",
            state=control_pb2.READY,
            host_port=8020,
        )
    )  # post-load check
    client.load_script = [
        make_event(control_pb2.LOADING, terminal=False),
        0.01,
        make_event(control_pb2.READY, terminal=True),
    ]
    app = build_test_app(swap_settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={"model": "llama-3.1-8b", "stream": True},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert client.load_calls == ["llama-3.1-8b"]
    assert len(upstream.requests) == 1
    assert (
        str(upstream.requests[0].url)
        == "http://127.0.0.1:8020/v1/chat/completions"
    )


async def test_app_shutdown_closes_dependencies(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    app = build_test_app(settings, client, upstream)
    # Run the lifespan manually: ASGITransport does not drive it.
    async with app.router.lifespan_context(app):
        pass
    assert client.closed is True


# -- IMP-03 additions --------------------------------------------------------


async def test_swap_attach_to_cli_initiated_load() -> None:
    """IMP-03.7: a controller-side JobInProgress for the *same* alias is
    attached to by polling GetStatus instead of failing the request."""
    client = FakeControllerClient()
    client.load_error = JobInProgress("a job is already running")
    client.status_script.extend(
        [
            control_pb2.Status(
                state=control_pb2.LOADING,
                current_job=control_pb2.Job(
                    kind=control_pb2.LOAD, alias="m", terminal=False
                ),
            ),
            make_status(active_alias="m", state=control_pb2.READY, host_port=8030),
        ]
    )
    coordinator = SwapCoordinator(client, timeout_s=5.0)
    assert await coordinator.ensure_ready("m") == 8030
    assert client.load_calls == ["m"]  # the gateway's load was still attempted
    assert coordinator.in_flight_alias is None


async def test_swap_job_in_progress_other_alias_conflicts() -> None:
    """IMP-03.7: a controller-side job for a *different* alias is a conflict."""
    client = FakeControllerClient()
    client.load_error = JobInProgress("a job is already running")
    client.status_script.append(
        control_pb2.Status(
            state=control_pb2.LOADING,
            current_job=control_pb2.Job(
                kind=control_pb2.LOAD, alias="other", terminal=False
            ),
        )
    )
    coordinator = SwapCoordinator(client, timeout_s=5.0)
    with pytest.raises(SwapConflict):
        await coordinator.ensure_ready("m")
    assert coordinator.in_flight_alias is None


async def test_swap_hung_stream_fails_after_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMP-03.5: a load stream that never terminates expires the job."""
    monkeypatch.setattr("vibe.gateway.swap._LOAD_STREAM_MARGIN", 0.1)
    client = FakeControllerClient()
    client.load_script = [make_event(control_pb2.LOADING, terminal=False), 60.0]
    coordinator = SwapCoordinator(client, timeout_s=5.0, load_timeout_s=0.1)
    with pytest.raises(SwapFailed) as excinfo:
        await coordinator.ensure_ready("m")
    assert "did not finish" in str(excinfo.value)
    # The slot is cleared: a later request starts a fresh load (FR-03.6).
    assert coordinator.in_flight_alias is None


async def test_app_head_health_200_empty(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.head("/health")
    assert response.status_code == 200
    assert response.content == b""


async def test_app_head_root_404(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    app = build_test_app(settings, FakeControllerClient(), upstream)
    async with asgi_client(app) as ac:
        response = await ac.head("/")
    assert response.status_code == 404
    assert response.content == b""


async def test_app_active_loading_retry_after_10(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    """HTTP-level Retry-After: 10 for the active-but-LOADING alias."""
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b", state=control_pb2.LOADING
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"model": "qwen2.5-coder-7b"}
        )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "10"
    assert response.json()["error"]["type"] == "model_not_ready"
    assert upstream.requests == []


async def test_app_active_error_retry_after_30_with_last_error(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    """IMP-03.13 at the HTTP level: last_error + reload hint, Retry-After 30."""
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b", state=control_pb2.ERROR
    )
    client.status_value.last_error = "container died"
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"model": "qwen2.5-coder-7b"}
        )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    body = response.json()["error"]
    assert "container died" in body["message"]
    assert "vibe load qwen2.5-coder-7b" in body["message"]


async def test_app_snapshot_ttl_refreshes_after_5s(
    settings: Settings,
    upstream: UpstreamRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NFR-03.2: the snapshot is reused within 5 s and refreshed after."""
    clock = {"now": 100.0}
    monkeypatch.setattr(
        gateway_app.time, "monotonic", lambda: clock["now"]
    )
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        await ac.get("/v1/models")
        await ac.get("/v1/models")  # within the TTL: snapshot reused
    assert client.list_calls == 1
    clock["now"] += 5.1  # past SNAPSHOT_TTL
    async with asgi_client(app) as ac:
        await ac.get("/v1/models")
    assert client.list_calls == 2


async def test_app_read_error_502_invalidates_cache(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    """IMP-03.9: httpx.ReadError maps to 502 + snapshot invalidation."""
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    client.status_value = make_status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8030,
    )
    upstream.error = httpx.ReadError("connection dropped mid-response")
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        first = await ac.post(
            "/v1/chat/completions", json={"model": "qwen2.5-coder-7b"}
        )
        second = await ac.post(
            "/v1/chat/completions", json={"model": "qwen2.5-coder-7b"}
        )
    assert first.status_code == 502
    assert second.status_code == 502
    assert client.list_calls == 2  # FR-03.9: the snapshot was re-fetched


async def test_app_413_rejected_on_content_length_before_read(
    settings: Settings,
    upstream: UpstreamRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IMP-03.11: an oversized Content-Length is rejected up front."""
    monkeypatch.setattr(gateway_app, "BODY_LIMIT", 16)
    client = FakeControllerClient()
    client.infos = [make_info("qwen2.5-coder-7b")]
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions",
            content=b"{\"model\": \"x\"}",
            headers={"content-length": "1024"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert client.list_calls == 0  # rejected before touching the controller
