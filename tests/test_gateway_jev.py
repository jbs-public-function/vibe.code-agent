"""Gateway Jev enforcement tests (improvements-2 doc 03; doc 06 §C, C1–C12).

No test touches docker or the network: the controller is a duck-typed
`FakeControllerClient` (this file's own, kept disjoint from
`tests/test_gateway.py`) and the vLLM upstream is an `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, cast, override

import httpx
import pytest
from starlette.applications import Starlette

from vibe.client import ControllerClient, ControllerUnavailable
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway import app as gateway_app
from vibe.gateway.app import build_app
from vibe.gateway.jev import apply_jev, check_compatible
from vibe.jev import canonical_hash, wire_schema

MODEL = "qwen2.5-coder-7b"
OTHER = "llama-3.1-8b"

# A minimal but representative schema: a title (drives the OpenAI name), a
# `$schema` to strip on the wire, and an enum like the built-in decisions.
SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Decision",
    "type": "object",
    "properties": {"result": {"type": "string", "enum": ["YES", "NO"]}},
    "required": ["result"],
    "additionalProperties": False,
}
SCHEMA_SHA = canonical_hash(SCHEMA)
WIRE = wire_schema(SCHEMA)


# -- shared helpers --------------------------------------------------------


def make_info(alias: str) -> control_pb2.ModelInfo:
    return control_pb2.ModelInfo(
        alias=alias,
        display_name=alias.upper(),
        max_model_len=1024,
        state=control_pb2.AVAILABLE,
        cached_locally=True,
    )


def jev_info(
    sha: str = SCHEMA_SHA, *, name: str = "default", stale: bool = False
) -> control_pb2.JevInfo:
    return control_pb2.JevInfo(
        schema_name=name,
        sha256=sha,
        source="builtin",
        applied_at_unix_ms=1_700_000_000_000,
        stale=stale,
        title="Decision",
    )


def make_status(
    active_alias: str = MODEL,
    host_port: int = 8030,
    jev: control_pb2.JevInfo | None = None,
) -> control_pb2.Status:
    status = control_pb2.Status(
        active_alias=active_alias, state=control_pb2.READY, host_port=host_port
    )
    if jev is not None:
        status.jev.CopyFrom(jev)
    return status


def make_event(
    state: control_pb2.ModelState, *, terminal: bool = True, message: str = ""
) -> control_pb2.JobEvent:
    return control_pb2.JobEvent(state=state, terminal=terminal, message=message)


def attach_jev(
    client: "FakeControllerClient",
    *,
    sha: str = SCHEMA_SHA,
    active_alias: str = MODEL,
) -> None:
    """Make the fake controller report Jev active and serve its schema."""
    info = jev_info(sha=sha)
    client.status_value = make_status(active_alias=active_alias, jev=info)
    client.active_jev_value = control_pb2.ActiveJev(
        info=info, schema_json=json.dumps(SCHEMA)
    )


class FakeControllerClient(ControllerClient):
    """Scriptable stand-in for `ControllerClient` (Jev-aware).

    Subclasses the real client so tests can pass it where `ControllerClient`
    is annotated, but never calls `super().__init__` (no gRPC channel).
    `active_jev_value` is returned by `get_active_jev()` (empty `ActiveJev`
    when `None`); `jev_error` makes it raise instead. `status_script` is
    consumed first by `status()` so tests can script successive snapshots.
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
        self.active_jev_value: control_pb2.ActiveJev | None = None
        self.jev_error: Exception | None = None
        self.jev_calls: int = 0
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
    async def get_active_jev(self) -> control_pb2.ActiveJev:
        self.jev_calls += 1
        if self.jev_error is not None:
            raise self.jev_error
        if self.active_jev_value is None:
            return control_pb2.ActiveJev()
        return self.active_jev_value

    @override
    async def load(
        self,
        alias: str,
        force: bool = False,
        keep_failed: bool = False,
        timeout_s: int | None = None,
        jev_mode: int = 0,
        jev: control_pb2.JevSpec | None = None,
    ) -> AsyncIterator[control_pb2.JobEvent]:
        self.load_calls.append(alias)
        if self.load_error is not None:
            raise self.load_error
        for item in self.load_script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            yield cast(control_pb2.JobEvent, item)

    @override
    async def close(self) -> None:
        self.closed = True
        await super().close()


class _ChunkedStream(httpx.AsyncByteStream):
    """Async byte stream that replays a fixed list of chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks: list[bytes] = chunks

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    @override
    async def aclose(self) -> None:
        pass


class UpstreamRecorder:
    """`httpx.MockTransport` handler factory for scripted upstream responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status: int = 200
        self.headers: dict[str, str] = {"content-type": "application/json"}
        self.chunks: list[bytes] | None = None  # None → the default body
        self.default_body: bytes = b'{"ok": true}'

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
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


# -- pure helpers (JEV-03.1) ------------------------------------------------


def test_apply_jev_removes_client_keys_and_injects() -> None:
    data: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "text"},
        "guided_json": {"type": "object"},
        "guided_regex": "^[a-z]+$",
        "guided_choice": ["a"],
        "guided_grammar": "root ::= \"yes\"",
        "structured_outputs": True,
        "temperature": 0.5,
    }
    original = deepcopy(data)
    rewrite = apply_jev(data, SCHEMA, "default")
    assert data == original  # the input is never mutated
    assert rewrite.overrode == [
        "response_format",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_grammar",
        "structured_outputs",
    ]
    out = rewrite.data
    for key in rewrite.overrode:
        if key != "response_format":  # that one is replaced by the Jev form
            assert key not in out
    assert out["temperature"] == 0.5  # unrelated keys survive
    assert out["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Decision", "schema": WIRE, "strict": True},
    }
    assert "$schema" not in out["response_format"]["json_schema"]["schema"]


def test_apply_jev_name_falls_back_to_store_name() -> None:
    rewrite = apply_jev({"model": MODEL}, {"type": "object"}, "my-schema")
    assert (
        rewrite.data["response_format"]["json_schema"]["name"] == "my-schema"
    )


def test_check_compatible_rejections() -> None:
    err = check_compatible("/v1/chat/completions", {"tools": [{}]})
    assert err is not None
    assert err.reason == "tools"
    assert err.code == "jev_incompatible_request"
    assert check_compatible(
        "/v1/chat/completions", {"tool_choice": "auto"}
    ) is not None
    assert check_compatible(
        "/v1/chat/completions", {"tool_choice": "required"}
    ) is not None
    assert check_compatible(
        "/v1/chat/completions",
        {"tool_choice": {"type": "function", "function": {"name": "f"}}},
    ) is not None
    err = check_compatible("/v1/completions", {})
    assert err is not None
    assert err.reason == "path"


def test_check_compatible_allows_clean_requests() -> None:
    assert check_compatible("/v1/chat/completions", {}) is None
    assert check_compatible("/v1/chat/completions", {"tools": []}) is None
    assert (
        check_compatible("/v1/chat/completions", {"tool_choice": "none"})
        is None
    )


# -- C1: Jev off is byte-for-byte unchanged ----------------------------------


async def test_c1_jev_off_passthrough_and_null_markers(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL), make_info(OTHER)]
    client.status_value = make_status()  # no Jev
    app = build_test_app(settings, client, upstream)
    payload = (
        b'{\n  "model": "qwen2.5-coder-7b",\n'
        b'  "response_format": {"type": "text"},\n'
        b'  "messages": [{"role": "user", "content": "hi"}]\n}'
    )
    async with asgi_client(app) as ac:
        response = await ac.post("/v1/chat/completions", content=payload)
        models = await ac.get("/v1/models")
        health = await ac.get("/health")
    assert response.status_code == 200
    # JEV-X.8: the proxied body is byte-for-byte the client's — the client's
    # response_format passes through untouched, nothing is re-serialised.
    assert upstream.requests[0].content == payload
    # No X-Vibe-* response headers in plain mode.
    assert "x-vibe-jev" not in response.headers
    assert "x-vibe-jev-overrode" not in response.headers
    assert models.status_code == 200
    for entry in models.json()["data"]:
        assert entry["vibe_jev"] is None
    assert health.json()["jev"] is None


# -- C2: Jev on injects the wire schema --------------------------------------


async def test_c2_jev_on_injects_response_format(
    settings: Settings, upstream: UpstreamRecorder, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    app = build_test_app(settings, client, upstream)
    caplog.set_level(logging.INFO, logger="vibe.gateway")
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    sent = json.loads(upstream.requests[0].content)
    assert sent["model"] == MODEL
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Decision",  # from the schema title
            "schema": WIRE,  # wire form: no $schema
            "strict": True,
        },
    }
    # JEV-03.5: the response carries the provenance header, only in Jev mode.
    assert response.headers["x-vibe-jev"] == f"default; sha={SCHEMA_SHA[:12]}"
    assert "x-vibe-jev-overrode" not in response.headers
    # JEV-03.8: one structured `jev_applied` line, no schema body in it.
    applied = [r for r in caplog.records if r.getMessage() == "jev_applied"]
    assert len(applied) == 1
    assert getattr(applied[0], "alias") == MODEL
    assert getattr(applied[0], "schema") == "default"
    assert getattr(applied[0], "sha") == SCHEMA_SHA[:12]
    assert getattr(applied[0], "elapsed_ms") >= 0


# -- C3: client response-format keys are removed and surfaced ----------------


async def test_c3_client_constraints_removed_and_surfaced(
    settings: Settings, upstream: UpstreamRecorder, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    app = build_test_app(settings, client, upstream)
    caplog.set_level(logging.INFO, logger="vibe.gateway")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "text"},
        "guided_json": {"type": "object"},
        "guided_regex": "^[a-z]+$",
        "structured_outputs": True,
    }
    async with asgi_client(app) as ac:
        response = await ac.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    sent = json.loads(upstream.requests[0].content)
    for key in ("guided_json", "guided_regex", "structured_outputs"):
        assert key not in sent
    # The client's response_format was overwritten by the Jev one.
    assert sent["response_format"]["type"] == "json_schema"
    assert response.headers["x-vibe-jev-overrode"] == (
        "response_format, guided_json, guided_regex, structured_outputs"
    )
    overrides = [r for r in caplog.records if r.getMessage() == "jev_override"]
    assert len(overrides) == 1
    assert getattr(overrides[0], "overrode") == [
        "response_format",
        "guided_json",
        "guided_regex",
        "structured_outputs",
    ]


# -- C4: tools / tool_choice compatibility -----------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
        {"tools": [{"type": "function"}], "tool_choice": "auto"},
        {"tool_choice": "auto"},
        {"tool_choice": "required"},
        {"tool_choice": {"type": "function", "function": {"name": "f"}}},
    ],
)
async def test_c4_incompatible_requests_rejected(
    settings: Settings, upstream: UpstreamRecorder, extra: dict[str, Any]
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={**extra, "model": MODEL}
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "jev_incompatible_request"
    assert error["type"] == "invalid_request_error"
    assert "Jev is active (default)" in error["message"]
    assert "vibe jev off" in error["message"]
    assert upstream.requests == []  # rejected before touching the upstream


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"tools": []},
        {"tool_choice": "none"},
        {"tools": [], "tool_choice": "none"},
    ],
)
async def test_c4_compatible_tool_choices_pass(
    settings: Settings, upstream: UpstreamRecorder, extra: dict[str, Any]
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={**extra, "model": MODEL}
        )
    assert response.status_code == 200
    sent = json.loads(upstream.requests[0].content)
    assert sent["response_format"]["type"] == "json_schema"


# -- C5: completions rejected, embeddings pass through -----------------------


async def test_c5_completions_rejected_embeddings_passthrough(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        rejected = await ac.post(
            "/v1/completions", json={"model": MODEL, "prompt": "hi"}
        )
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "jev_incompatible_request"
    assert upstream.requests == []
    # /v1/embeddings passes through untouched, even in Jev mode.
    payload = b'{"model": "qwen2.5-coder-7b", "input": "hi"}'
    async with asgi_client(app) as ac:
        response = await ac.post("/v1/embeddings", content=payload)
    assert response.status_code == 200
    assert upstream.requests[0].content == payload
    assert "response_format" not in json.loads(upstream.requests[0].content)
    assert "x-vibe-jev" not in response.headers


# -- C6: stale Content-Length is never forwarded -----------------------------


async def test_c6_stale_content_length_recomputed(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    app = build_test_app(settings, client, upstream)
    payload = b'{"model": "qwen2.5-coder-7b", "messages": [{"role": "user", "content": "hi"}]}'
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions",
            content=payload,
            headers={"content-length": "99999"},
        )
    assert response.status_code == 200
    sent = upstream.requests[0]
    # The re-serialised (injected) body's length, not the client's header.
    assert sent.headers["content-length"] == str(len(sent.content))
    assert len(sent.content) != 99999
    assert json.loads(sent.content)["response_format"]["type"] == "json_schema"


# -- C7: GetActiveJev failure / sha mismatch fails closed --------------------


async def test_c7_jev_unavailable_on_get_active_jev_failure(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    client.jev_error = ControllerUnavailable("controller unreachable")
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"model": MODEL, "messages": []}
        )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    error = response.json()["error"]
    assert error["type"] == "jev_unavailable"
    assert error["code"] == "jev_unavailable"
    assert "Jev is active (default)" in error["message"]
    assert upstream.requests == []  # fail closed: the upstream is not called


async def test_c7_jev_unavailable_on_sha_mismatch_or_empty(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    # GetActiveJev answers with a different sha than Status.jev.
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    client.active_jev_value = control_pb2.ActiveJev(
        info=jev_info(sha="beef" * 16), schema_json=json.dumps(SCHEMA)
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"model": MODEL, "messages": []}
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_unavailable"
    assert upstream.requests == []
    # GetActiveJev answers with no info at all (Jev vanished concurrently).
    client2 = FakeControllerClient()
    client2.infos = [make_info(MODEL)]
    attach_jev(client2)
    client2.active_jev_value = control_pb2.ActiveJev()
    app2 = build_test_app(settings, client2, upstream)
    async with asgi_client(app2) as ac:
        response = await ac.post(
            "/v1/chat/completions", json={"model": MODEL, "messages": []}
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_unavailable"
    assert upstream.requests == []


# -- C8: the schema body is cached by sha ------------------------------------


async def test_c8_jev_body_cached_by_sha(
    settings: Settings,
    upstream: UpstreamRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(gateway_app.time, "monotonic", lambda: clock["now"])
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        for _ in range(3):
            response = await ac.post(
                "/v1/chat/completions", json={"model": MODEL}
            )
            assert response.status_code == 200
    assert client.jev_calls == 1  # fetched once for this sha
    # The active sha changes → the body is fetched again.
    new_sha = "ab" * 32
    other_schema = {**SCHEMA, "title": "Other"}
    client.status_value = make_status(jev=jev_info(sha=new_sha))
    client.active_jev_value = control_pb2.ActiveJev(
        info=jev_info(sha=new_sha), schema_json=json.dumps(other_schema)
    )
    clock["now"] += 5.1  # past SNAPSHOT_TTL so the status is re-read
    async with asgi_client(app) as ac:
        response = await ac.post("/v1/chat/completions", json={"model": MODEL})
    assert response.status_code == 200
    assert client.jev_calls == 2
    sent = json.loads(upstream.requests[-1].content)
    assert sent["response_format"]["json_schema"]["name"] == "Other"
    assert response.headers["x-vibe-jev"] == f"default; sha={new_sha[:12]}"


# -- C9: SSE relayed chunk-for-chunk in Jev mode -----------------------------


async def test_c9_sse_relayed_chunk_for_chunk(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    upstream.headers = {"content-type": "text/event-stream"}
    chunks = [b'data: {"delta":"Hi"}\n\n', b"data: [DONE]\n\n"]
    upstream.chunks = chunks
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={"model": MODEL, "stream": True},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == b"".join(chunks)  # identical chunk bytes
    assert response.headers["x-vibe-jev"] == f"default; sha={SCHEMA_SHA[:12]}"
    sent = json.loads(upstream.requests[0].content)
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["stream"] is True


# -- C10: swap-on-request after Jev lands in plain mode ----------------------


async def test_c10_swap_after_jev_is_plain(upstream: UpstreamRecorder) -> None:
    swap_settings = Settings(
        vllm_image_tag="v0.11.2", vibe_swap_on_request=True, vibe_swap_timeout=10
    )
    client = FakeControllerClient()
    client.infos = [make_info(MODEL), make_info(OTHER)]
    client.status_script.append(
        make_status(jev=jev_info())
    )  # routing snapshot: Jev active on MODEL
    client.status_script.append(
        make_status(active_alias=OTHER)
    )  # post-load check: plain
    client.status_value = make_status(active_alias=OTHER)  # later snapshots
    client.load_script = [
        make_event(control_pb2.LOADING, terminal=False),
        make_event(control_pb2.READY, terminal=True),
    ]
    app = build_test_app(swap_settings, client, upstream)
    payload = {"model": OTHER, "messages": [{"role": "user", "content": "hi"}]}
    async with asgi_client(app) as ac:
        first = await ac.post("/v1/chat/completions", json=payload)
        second = await ac.post("/v1/chat/completions", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert client.load_calls == [OTHER]
    # Jev is never carried across swaps: neither request was injected.
    assert len(upstream.requests) == 2
    for sent in upstream.requests:
        assert "response_format" not in json.loads(sent.content)
    assert "x-vibe-jev" not in first.headers
    assert "x-vibe-jev" not in second.headers


# -- C11: upstream 400 relayed verbatim; snapshot invalidated; no retry ------


async def test_c11_upstream_400_relayed_and_snapshot_invalidated(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL)]
    attach_jev(client)
    upstream.status = 400
    upstream.default_body = (
        b'{"error": {"message": "schema rejected", '
        b'"type": "invalid_request_error", "code": null}}'
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        first = await ac.post("/v1/chat/completions", json={"model": MODEL})
        second = await ac.post("/v1/chat/completions", json={"model": MODEL})
    assert first.status_code == 400
    assert first.content == upstream.default_body  # relayed verbatim
    assert second.status_code == 400
    assert client.list_calls == 2  # the 400 invalidated the snapshot
    # One upstream call per client request: no unconstrained retry, and the
    # second attempt was still injected (the cached schema body survived).
    assert len(upstream.requests) == 2
    for sent in upstream.requests:
        assert json.loads(sent.content)["response_format"]["type"] == "json_schema"
    assert client.jev_calls == 1


# -- C12: /health and /v1/models markers -------------------------------------


async def test_c12_health_and_models_markers(
    settings: Settings, upstream: UpstreamRecorder
) -> None:
    client = FakeControllerClient()
    client.infos = [make_info(MODEL), make_info(OTHER)]
    client.status_value = make_status(jev=jev_info(stale=True))
    client.active_jev_value = control_pb2.ActiveJev(
        info=jev_info(stale=True), schema_json=json.dumps(SCHEMA)
    )
    app = build_test_app(settings, client, upstream)
    async with asgi_client(app) as ac:
        health = await ac.get("/health")
        models = await ac.get("/v1/models")
    assert health.status_code == 200
    assert health.json()["jev"] == {"name": "default", "sha256": SCHEMA_SHA}
    assert models.status_code == 200
    data = models.json()["data"]
    active = [entry for entry in data if entry["id"] == MODEL][0]
    assert active["vibe_jev"] == {
        "name": "default",
        "sha256": SCHEMA_SHA,
        "source": "builtin",
        "stale": True,
    }
    for entry in data:
        if entry["id"] != MODEL:
            assert entry["vibe_jev"] is None
    # Controller down: /health still answers 200 with the marker null, and
    # /v1/models keeps its existing 503.
    down = FakeControllerClient()
    down.infos = [make_info(MODEL)]
    down.list_error = ControllerUnavailable("controller unreachable")
    down_app = build_test_app(settings, down, upstream)
    async with asgi_client(down_app) as ac:
        health = await ac.get("/health")
        models = await ac.get("/v1/models")
    assert health.status_code == 200
    assert health.json()["jev"] is None
    assert health.json()["controller"] == "down"
    assert models.status_code == 503
