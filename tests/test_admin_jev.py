"""Admin Jev schema pages (doc 05 / doc 06 §E, phase J4).

Mirrors `tests/test_admin.py`: a duck-typed fake controller client drives the
composed FastAPI app through `httpx.ASGITransport`. The fake implements the
Jev part of the client contract (plus the small surface the shared pages
need) so the E1–E13 acceptance tests run without a real controller.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, cast, override

import httpx
import pytest
from fastapi import FastAPI
from markupsafe import escape as esc

from vibe.admin.app import build_app
from vibe.admin.jev_builder import build_decision_schema
from vibe.admin.routes.jev import MINIMAL_TEMPLATE
from vibe.client import (
    ControllerClient,
    ControllerUnavailable,
    JevReadOnly,
    JevSchemaNotFound,
)
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.jev import canonical_hash, load_builtin, validate_schema_text
from vibe.jobs import JobInProgress

CSP = (
    "default-src 'self'; script-src 'none'; style-src 'self'; "
    "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)

SHA_A = "a" * 64
SHA_B = "b" * 64

VALID_SCHEMA = '{\n  "type": "object",\n  "properties": {"a": {"type": "string"}}\n}'
BUILTIN_DEFAULT_TEXT = json.dumps(load_builtin("default"), indent=2)


# -- shared helpers ----------------------------------------------------------


def make_info(alias: str, *, tags: tuple[str, ...] = ()) -> control_pb2.ModelInfo:
    return control_pb2.ModelInfo(
        alias=alias,
        display_name=f"{alias} display",
        hf_repo=f"org/{alias}",
        quantization="awq",
        max_model_len=4096,
        host_port=8020,
        state=control_pb2.AVAILABLE,
        cached_locally=True,
        tags=list(tags),
    )


def make_event(
    job_id: str,
    *,
    state: control_pb2.ModelState = control_pb2.LOADING,
    message: str = "",
    terminal: bool = False,
    error_code: str = "",
    kind: Any = control_pb2.LOAD,
) -> control_pb2.JobEvent:
    return control_pb2.JobEvent(
        job_id=job_id,
        ts_unix_ms=int(time.time() * 1000),
        level="INFO",
        message=message,
        state=state,
        terminal=terminal,
        error_code=error_code,
        kind=kind,
    )


def make_entry(
    name: str,
    *,
    source: str = "user",
    sha: str = SHA_A,
    size: int = 100,
    mtime: int = 0,
    active: bool = False,
) -> control_pb2.JevSchemaEntry:
    return control_pb2.JevSchemaEntry(
        name=name,
        source=source,
        sha256=sha,
        size_bytes=size,
        mtime_unix_ms=mtime,
        title="",
        active=active,
    )


def make_schema(
    name: str,
    *,
    source: str = "user",
    sha: str = SHA_A,
    text: str = VALID_SCHEMA,
) -> control_pb2.GetJevSchemaResponse:
    return control_pb2.GetJevSchemaResponse(
        name=name,
        source=source,
        sha256=sha,
        size_bytes=len(text.encode()),
        mtime_unix_ms=0,
        title="",
        schema_json=text,
    )


class FakeJevClient(ControllerClient):
    """Scriptable in-memory controller implementing the Jev client contract.

    Subclasses the real client so it can be injected where
    `ControllerClient` is annotated, but never issues a real RPC.
    """

    def __init__(self) -> None:
        super().__init__(Settings(vllm_image_tag="v0.11.2"))
        self.infos: list[control_pb2.ModelInfo] = []
        self.list_error: Exception | None = None
        self.status_value: control_pb2.Status = control_pb2.Status()
        self.status_error: Exception | None = None
        self.gpu_value: control_pb2.GpuStats = control_pb2.GpuStats()
        self.gpu_error: Exception | None = None
        self.jev_list_value: list[control_pb2.JevSchemaEntry] = []
        self.jev_list_error: Exception | None = None
        self.jev_schemas: dict[str, control_pb2.GetJevSchemaResponse] = {}
        self.jev_get_error: Exception | None = None
        self.active_jev_value: control_pb2.ActiveJev = control_pb2.ActiveJev()
        self.active_jev_error: Exception | None = None
        self.put_response: control_pb2.PutJevSchemaResponse = (
            control_pb2.PutJevSchemaResponse(
                validation=control_pb2.JevValidation(ok=True)
            )
        )
        self.put_error: Exception | None = None
        self.put_calls: list[tuple[str, str]] = []
        self.validate_response: control_pb2.JevValidation = control_pb2.JevValidation(
            ok=True
        )
        self.validate_error: Exception | None = None
        self.validate_calls: list[str] = []
        self.delete_response: control_pb2.DeleteJevSchemaResponse = (
            control_pb2.DeleteJevSchemaResponse(ok=True)
        )
        self.delete_error: Exception | None = None
        self.delete_calls: list[str] = []
        self.load_calls: list[tuple[str, bool, Any, control_pb2.JevSpec | None]] = []
        self.load_script: list[control_pb2.JobEvent | float] = []
        self.load_error: Exception | None = None
        self.unload_script: list[control_pb2.JobEvent | float] = []
        self.job_events_map: dict[str, list[control_pb2.JobEvent]] = {}
        self.log_lines: list[control_pb2.LogLine] = []
        self.closed = False

    @override
    async def list_models(self) -> list[control_pb2.ModelInfo]:
        if self.list_error is not None:
            raise self.list_error
        return list(self.infos)

    @override
    async def status(self) -> control_pb2.Status:
        if self.status_error is not None:
            raise self.status_error
        return self.status_value

    @override
    async def gpu(self) -> control_pb2.GpuStats:
        if self.gpu_error is not None:
            raise self.gpu_error
        return self.gpu_value

    @override
    async def get_active_jev(self) -> control_pb2.ActiveJev:
        if self.active_jev_error is not None:
            raise self.active_jev_error
        return self.active_jev_value

    @override
    async def list_jev_schemas(self) -> list[control_pb2.JevSchemaEntry]:
        if self.jev_list_error is not None:
            raise self.jev_list_error
        return list(self.jev_list_value)

    @override
    async def get_jev_schema(self, name: str) -> control_pb2.GetJevSchemaResponse:
        if self.jev_get_error is not None:
            raise self.jev_get_error
        if name not in self.jev_schemas:
            raise JevSchemaNotFound(f"no such Jev schema: {name}")
        return self.jev_schemas[name]

    @override
    async def put_jev_schema(
        self, name: str, schema_json: str
    ) -> control_pb2.PutJevSchemaResponse:
        self.put_calls.append((name, schema_json))
        if self.put_error is not None:
            raise self.put_error
        return self.put_response

    @override
    async def delete_jev_schema(
        self, name: str
    ) -> control_pb2.DeleteJevSchemaResponse:
        self.delete_calls.append(name)
        if self.delete_error is not None:
            raise self.delete_error
        return self.delete_response

    @override
    async def validate_jev_schema(self, schema_json: str) -> control_pb2.JevValidation:
        self.validate_calls.append(schema_json)
        if self.validate_error is not None:
            raise self.validate_error
        return self.validate_response

    @override
    async def load(
        self,
        alias: str,
        force: bool = False,
        keep_failed: bool = False,
        timeout_s: int | None = None,
        jev_mode: Any = control_pb2.JEV_MODE_UNSPECIFIED,
        jev: control_pb2.JevSpec | None = None,
    ) -> AsyncIterator[control_pb2.JobEvent]:
        self.load_calls.append((alias, force, jev_mode, jev))
        if self.load_error is not None:
            raise self.load_error
        for item in self.load_script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            yield cast(control_pb2.JobEvent, item)

    @override
    async def unload(self) -> AsyncIterator[control_pb2.JobEvent]:
        for item in self.unload_script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            yield cast(control_pb2.JobEvent, item)

    @override
    async def job_events(
        self, job_id: str, *, follow: bool = False
    ) -> AsyncIterator[control_pb2.JobEvent]:
        if job_id not in self.job_events_map:
            raise KeyError(job_id)
        for event in self.job_events_map[job_id]:
            yield event

    @override
    async def logs(
        self, tail: int | None = None, follow: bool = False
    ) -> AsyncIterator[control_pb2.LogLine]:
        for line in self.log_lines:
            yield line

    @override
    async def close(self) -> None:
        self.closed = True
        await super().close()


def populated_fake() -> FakeJevClient:
    """A fake with models, schemas and an active Jev attachment."""
    fake = FakeJevClient()
    fake.infos = [
        make_info("qwen2.5-coder-7b"),
        make_info("deepseek-r1-14b", tags=("reasoning",)),
    ]
    fake.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        host_port=8020,
        container_id="c" * 12,
    )
    fake.jev_list_value = [
        make_entry("default", source="builtin", sha=SHA_A, size=800, active=True),
        make_entry("yes-no", source="builtin", sha=SHA_B, size=300),
        make_entry("my-choice", source="user", sha=SHA_A, size=100, mtime=1700000000000),
    ]
    fake.jev_schemas = {
        "default": make_schema("default", source="builtin", sha=SHA_A, text=BUILTIN_DEFAULT_TEXT),
        "yes-no": make_schema("yes-no", source="builtin", sha=SHA_B, text='{"type": "object"}'),
        "my-choice": make_schema("my-choice", source="user", sha=SHA_A, text=VALID_SCHEMA),
    }
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
    return fake


def make_app(fake: FakeJevClient, *, gateway_status: int = 200) -> FastAPI:
    """Compose the app with a mocked gateway probe (IMP-04.24)."""
    app = build_app(Settings(vllm_image_tag="v0.11.2", vibe_bind="test"), fake)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(gateway_status, text="ok")

    app.state.gateway_transport = httpx.MockTransport(handler)
    return app


def make_transport(
    fake: FakeJevClient, *, gateway_status: int = 200
) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=make_app(fake, gateway_status=gateway_status))


async def get(transport: httpx.ASGITransport, path: str) -> httpx.Response:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def post(
    transport: httpx.ASGITransport,
    path: str,
    data: dict[str, str] | None = None,
    *,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, data=data, content=content, headers=headers)


async def post_and_follow(
    transport: httpx.ASGITransport,
    path: str,
    data: dict[str, str] | None = None,
) -> tuple[httpx.Response, str]:
    """POST and follow the 303 in one client so the flash cookie survives."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(path, data=data)
        page = await client.get(response.headers["location"])
        return response, page.text


def hidden_value(text: str, name: str) -> str:
    """The value of a hidden form field in rendered HTML."""
    match = re.search(rf'<input type="hidden" name="{name}" value="([^"]*)">', text)
    assert match is not None, f"hidden field {name!r} not found"
    return html.unescape(match.group(1))


def textarea_content(text: str) -> str:
    """The textarea body of the rendered editor, unescaped."""
    match = re.search(
        r'<textarea name="schema_json"[^>]*>(.*?)</textarea>', text, re.S
    )
    assert match is not None, "schema_json textarea not found"
    return html.unescape(match.group(1))


def textarea_of(text: str) -> str:
    """The raw (still-escaped) body of the schema textarea."""
    match = re.search(
        r'<textarea name="schema_json"[^>]*>(.*?)</textarea>', text, re.S
    )
    assert match is not None, "schema_json textarea not found"
    return match.group(1)


# -- E1: list page -----------------------------------------------------------


async def test_list_page_renders_schemas_active_panel_and_badges() -> None:
    fake = populated_fake()
    fake.jev_list_value[0].active = True
    fake.active_jev_value.info.stale = True
    transport = make_transport(fake)
    response = await get(transport, "/jev")
    assert response.status_code == 200
    text = response.text
    # Table: every schema with source badge, sha12, size, active marker.
    assert "default" in text and "yes-no" in text and "my-choice" in text
    assert "built-in" in text and "user" in text
    assert SHA_A[:12] in text
    assert "800" in text and "bytes" in text
    assert "active" in text
    # Active panel: name, source, sha12, applied-at, stale badge, detach button.
    assert "Active Jev" in text
    assert 'action="/jev/detach"' in text
    assert "stale" in text
    # Launch form with model + schema selects and force checkbox.
    assert 'action="/jev/launch"' in text
    assert "qwen2.5-coder-7b" in text
    assert 'name="force"' in text
    # Reasoning model note (JEV-05.11).
    assert "deepseek-r1-14b" in text and "reasoning" in text
    # Nav shows the Jev page as active.
    assert '<a href="/jev" class="active">Jev</a>' in text


async def test_list_page_without_active_jev_shows_no_attachment() -> None:
    fake = populated_fake()
    fake.active_jev_value = control_pb2.ActiveJev()
    fake.jev_list_value = [make_entry("default", source="builtin")]
    transport = make_transport(fake)
    response = await get(transport, "/jev")
    assert response.status_code == 200
    assert "No schema attached" in response.text
    assert "Detach" not in response.text


async def test_list_page_controller_down_banner_no_500() -> None:
    fake = populated_fake()
    fake.jev_list_error = ControllerUnavailable("down")
    fake.status_error = ControllerUnavailable("down")
    fake.active_jev_error = ControllerUnavailable("down")
    fake.list_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    response = await get(transport, "/jev")
    assert response.status_code == 200
    assert "Controller unreachable" in response.text
    assert "disabled" in response.text  # launch form disabled (JEV-05.11)


# -- E2: save invalid preserves text -----------------------------------------


async def test_save_invalid_rerenders_preserving_text_byte_for_byte() -> None:
    fake = populated_fake()
    bad = '{\n  "title": "x",\n  "nope\n'
    fake.put_response = control_pb2.PutJevSchemaResponse(
        validation=control_pb2.JevValidation(
            ok=False, errors=["not valid JSON: unterminated string at line 4, column 1"]
        )
    )
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": bad, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert f">{esc(bad)}</textarea>" in response.text
    assert 'role="alert"' in response.text
    assert "not valid JSON" in response.text
    assert fake.put_calls == [("my-choice", bad)]


async def test_save_valid_redirects_with_flash() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response, page = await post_and_follow(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jev/my-choice"
    assert "saved &#39;my-choice&#39;" in page
    assert fake.put_calls == [("my-choice", VALID_SCHEMA)]


async def test_save_active_schema_flash_mentions_stale() -> None:
    fake = populated_fake()
    fake.active_jev_value = control_pb2.ActiveJev(
        info=control_pb2.JevInfo(schema_name="my-choice", sha256=SHA_A, source="store")
    )
    transport = make_transport(fake)
    _, page = await post_and_follow(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert "active Jev is now stale" in page
    assert "re-launch" in page


async def test_save_inactive_schema_flash_has_no_stale_note() -> None:
    fake = populated_fake()
    fake.active_jev_value = control_pb2.ActiveJev()  # nothing attached
    transport = make_transport(fake)
    _, page = await post_and_follow(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert "stale" not in page


async def test_save_put_store_error_rendered_inline() -> None:
    fake = populated_fake()
    fake.put_response = control_pb2.PutJevSchemaResponse(
        error="at most 256 user schemas are allowed"
    )
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert "at most 256 user schemas" in response.text
    assert textarea_content(response.text) == VALID_SCHEMA


async def test_save_controller_down_rerenders_inline() -> None:
    fake = populated_fake()
    fake.put_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert "controller unreachable" in response.text
    assert esc(VALID_SCHEMA) in response.text


# -- E3: validate / save / save_launch RPC calls -----------------------------


async def test_validate_calls_validate_rpc_not_put() -> None:
    fake = populated_fake()
    fake.validate_response = control_pb2.JevValidation(
        ok=True, sha256=SHA_A, size_bytes=123, warnings=["no '$schema' declared"]
    )
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "validate", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert fake.validate_calls == [VALID_SCHEMA]
    assert fake.put_calls == []
    assert f"valid (sha {SHA_A[:12]}, 123 bytes)" in response.text
    assert "warning: no" in response.text


async def test_validate_failure_rerenders_with_errors() -> None:
    fake = populated_fake()
    fake.validate_response = control_pb2.JevValidation(
        ok=False, errors=["schema root must be a JSON object, got list"]
    )
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "validate", "schema_json": "[1, 2]", "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert "schema root must be a JSON object" in response.text
    assert esc("[1, 2]") in response.text
    assert fake.put_calls == []


async def test_save_calls_put_once() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 303
    assert fake.put_calls == [("my-choice", VALID_SCHEMA)]
    assert fake.validate_calls == []


async def test_save_launch_puts_then_loads_with_jev_on() -> None:
    fake = populated_fake()
    fake.load_script = [make_event("j-9", message="jev: probing")]
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {
            "action": "save_launch",
            "schema_json": VALID_SCHEMA,
            "base_sha": SHA_A,
            "alias": "qwen2.5-coder-7b",
            "force": "on",
        },
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/j-9"
    assert fake.put_calls == [("my-choice", VALID_SCHEMA)]
    assert len(fake.load_calls) == 1
    alias, force, jev_mode, spec = fake.load_calls[0]
    assert alias == "qwen2.5-coder-7b"
    assert force is True
    assert jev_mode == control_pb2.JEV_ON
    assert spec is not None
    assert spec.schema_name == "my-choice"
    assert spec.schema_json == ""


async def test_save_launch_missing_alias_inline_error_no_put() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save_launch", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert "pick a model to launch on" in response.text
    assert fake.put_calls == []
    assert fake.load_calls == []


async def test_save_launch_validation_error_rerenders_not_redirect() -> None:
    fake = populated_fake()
    fake.put_response = control_pb2.PutJevSchemaResponse(
        validation=control_pb2.JevValidation(ok=False, errors=["empty enum at #/enum"])
    )
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {
            "action": "save_launch",
            "schema_json": '{"enum": []}',
            "base_sha": SHA_A,
            "alias": "qwen2.5-coder-7b",
        },
    )
    assert response.status_code == 200
    assert "empty enum" in response.text
    assert esc('{"enum": []}') in response.text
    assert fake.load_calls == []


# -- E4: launch_unsaved ------------------------------------------------------


async def test_launch_unsaved_sends_inline_schema_json() -> None:
    fake = populated_fake()
    fake.load_script = [make_event("j-9", message="jev: probing")]
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {
            "action": "launch_unsaved",
            "schema_json": VALID_SCHEMA,
            "alias": "qwen2.5-coder-7b",
        },
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/j-9"
    alias, force, jev_mode, spec = fake.load_calls[0]
    assert jev_mode == control_pb2.JEV_ON
    assert spec is not None
    assert spec.schema_name == ""
    assert spec.schema_json == VALID_SCHEMA
    assert fake.put_calls == []


async def test_launch_unsaved_first_event_error_flashes_not_500() -> None:
    fake = populated_fake()
    fake.load_script = [
        make_event(
            "j-1",
            state=control_pb2.ERROR,
            terminal=True,
            message="jev: not valid JSON: unexpected end at line 1, column 3",
            error_code="jev_schema_invalid",
        )
    ]
    transport = make_transport(fake)
    response, page = await post_and_follow(
        transport,
        "/jev/my-choice",
        {
            "action": "launch_unsaved",
            "schema_json": '{"bad',
            "alias": "qwen2.5-coder-7b",
        },
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jev/my-choice"
    assert "launch failed" in page
    assert "not valid JSON" in page


async def test_launch_unsaved_missing_alias_inline_error() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport, "/jev/my-choice", {"action": "launch_unsaved", "schema_json": VALID_SCHEMA}
    )
    assert response.status_code == 200
    assert "pick a model to launch on" in response.text
    assert fake.load_calls == []


async def test_launch_unsaved_empty_name_on_new_flow() -> None:
    # Regression (runbook C5): the name gate must not block the
    # name-less "Launch without saving" action on /jev/new.
    fake = populated_fake()
    fake.load_script = [make_event("j-10", message="jev: probing")]
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/new",
        {
            "action": "launch_unsaved",
            "schema_json": VALID_SCHEMA,
            "alias": "qwen2.5-coder-7b",
        },
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/j-10"
    _, _, jev_mode, spec = fake.load_calls[0]
    assert spec is not None
    assert jev_mode == control_pb2.JEV_ON
    assert spec.schema_name == ""
    assert spec.schema_json == VALID_SCHEMA
    assert fake.put_calls == []


async def test_validate_empty_name_on_new_flow() -> None:
    # Regression (runbook C2/C4): validating pasted text must not
    # require a name first.
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/new",
        {"action": "validate", "schema_json": VALID_SCHEMA},
    )
    assert response.status_code == 200
    assert "valid (sha" in response.text
    assert "enter a schema name" not in response.text


# -- E5: escaping ------------------------------------------------------------


async def test_escaping_schema_and_error_text_inert() -> None:
    fake = populated_fake()
    payload = "</textarea><script>alert(1)</script>"
    evil_error = '<img onerror="alert(1)">boom'
    fake.put_response = control_pb2.PutJevSchemaResponse(
        validation=control_pb2.JevValidation(ok=False, errors=[evil_error])
    )
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": payload, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    text = response.text
    assert "</textarea><script>" not in text
    assert esc(payload) in text
    assert "<img onerror" not in text
    assert esc(evil_error) in text


async def test_editor_get_escapes_stored_schema_text() -> None:
    fake = populated_fake()
    fake.jev_schemas["my-choice"] = make_schema(
        "my-choice", text='{"x": "</textarea><script>alert(1)</script>"}'
    )
    transport = make_transport(fake)
    response = await get(transport, "/jev/my-choice")
    assert response.status_code == 200
    assert "</textarea><script>" not in response.text
    assert esc('{"x": "</textarea><script>alert(1)</script>"}') in response.text


# -- E6: body cap ------------------------------------------------------------


async def test_body_over_128k_rejected_413_before_parse() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    big = b"schema_json=" + b"x" * (200 * 1024)
    response = await post(
        transport, "/jev/my-choice", content=big + b"&action=save"
    )
    assert response.status_code == 413
    assert "413" in response.text
    assert fake.put_calls == []
    assert fake.validate_calls == []
    assert response.headers["content-security-policy"] == CSP
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_body_cap_does_not_affect_other_routes() -> None:
    fake = populated_fake()
    fake.unload_script = []  # empty stream → flash, not 413
    transport = make_transport(fake)
    response = await post(transport, "/models/unload", content=b"x" * (200 * 1024))
    assert response.status_code == 303


async def test_body_just_under_cap_is_parsed() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "validate", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert fake.validate_calls == [VALID_SCHEMA]


# -- E7: cross-origin 403 ----------------------------------------------------


JEVI_POST_PATHS = [
    "/jev/new",
    "/jev/yes-no",
    "/jev/yes-no/delete",
    "/jev/launch",
    "/jev/detach",
    "/jev/build",
]


@pytest.mark.parametrize("path", JEVI_POST_PATHS)
async def test_cross_origin_post_rejected_on_every_jev_route(path: str) -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport, path, headers={"Origin": "http://evil.example"}
    )
    assert response.status_code == 403


async def test_same_origin_post_passes() -> None:
    fake = populated_fake()
    fake.load_script = [make_event("j-9")]
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/launch",
        {"alias": "qwen2.5-coder-7b", "schema_name": "yes-no"},
        headers={"Origin": "http://test"},
    )
    assert response.status_code == 303


# -- E8: name validation -----------------------------------------------------


@pytest.mark.parametrize("name", ["new", "launch", "detach", "build"])
async def test_reserved_names_inline_error_on_new_flow(name: str) -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/new",
        {"action": "save", "name": name, "schema_json": VALID_SCHEMA},
    )
    assert response.status_code == 200
    assert "reserved" in response.text
    assert fake.put_calls == []


async def test_invalid_name_inline_error_on_new_flow() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/new",
        {"action": "save", "name": "Bad Name", "schema_json": VALID_SCHEMA},
    )
    assert response.status_code == 200
    assert "invalid schema name" in response.text
    assert fake.put_calls == []


async def test_new_flow_missing_name_inline_error() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport, "/jev/new", {"action": "save", "schema_json": VALID_SCHEMA}
    )
    assert response.status_code == 200
    assert "enter a schema name" in response.text
    assert fake.put_calls == []


async def test_invalid_path_param_name_404() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    for path in ("/jev/Bad%20Name", "/jev/launch"):
        response = await get(transport, path)
        assert response.status_code == 404, path
        assert "not found" in response.text
    # Path traversal is normalised away before routing; the router's own 404.
    response = await get(transport, "/jev/..%2Fx")
    assert response.status_code == 404


async def test_invalid_path_param_name_post_404_before_parse() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(transport, "/jev/Bad%20Name", {"action": "save"})
    assert response.status_code == 404
    assert fake.put_calls == []


async def test_unknown_schema_edit_page_404() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/jev/nope")
    assert response.status_code == 404


# -- E9: delete --------------------------------------------------------------


async def test_delete_confirm_override_shows_revert_copy() -> None:
    fake = populated_fake()
    fake.jev_schemas["default"] = make_schema("default", source="override", text=VALID_SCHEMA)
    transport = make_transport(fake)
    response = await get(transport, "/jev/default/delete")
    assert response.status_code == 200
    assert "reverts to the built-in" in response.text
    assert 'action="/jev/default/delete"' in response.text


async def test_delete_confirm_user_has_no_revert_copy() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/jev/my-choice/delete")
    assert response.status_code == 200
    assert "reverts" not in response.text
    assert "removed from the store" in response.text


async def test_delete_confirm_builtin_readonly_no_form() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/jev/default/delete")
    assert response.status_code == 200
    assert "built-ins cannot be deleted" in response.text
    assert "<form" not in response.text


async def test_delete_override_flashes_revert() -> None:
    fake = populated_fake()
    fake.delete_response = control_pb2.DeleteJevSchemaResponse(
        ok=True, reverted_to="builtin"
    )
    transport = make_transport(fake)
    response, page = await post_and_follow(transport, "/jev/default/delete")
    assert response.status_code == 303
    assert response.headers["location"] == "/jev"
    assert "reverted to the built-in" in page
    assert fake.delete_calls == ["default"]


async def test_delete_user_flashes_removed() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    _, page = await post_and_follow(transport, "/jev/my-choice/delete")
    assert "deleted my-choice" in page


async def test_delete_builtin_refused() -> None:
    fake = populated_fake()
    fake.delete_error = JevReadOnly("built-in schemas are read-only")
    transport = make_transport(fake)
    _, page = await post_and_follow(transport, "/jev/default/delete")
    assert "built-ins cannot be deleted" in page


async def test_delete_unknown_schema_flashes() -> None:
    fake = populated_fake()
    fake.delete_error = JevSchemaNotFound("no such Jev schema")
    transport = make_transport(fake)
    _, page = await post_and_follow(transport, "/jev/nope/delete")
    assert "no schema named nope" in page


async def test_delete_controller_down_flashes() -> None:
    fake = populated_fake()
    fake.delete_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    _, page = await post_and_follow(transport, "/jev/my-choice/delete")
    assert "controller unreachable" in page


# -- E10: zero-JS guard ------------------------------------------------------


JEVI_GET_PAGES = ["/jev", "/jev/new", "/jev/yes-no", "/jev/my-choice/delete"]


@pytest.mark.parametrize("path", JEVI_GET_PAGES)
async def test_jev_pages_script_free_and_csp_unchanged(path: str) -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, path)
    assert response.status_code == 200
    assert "<script" not in response.text
    assert re.search(r"\son[a-z]+\s*=", response.text) is None
    assert "style=" not in response.text
    assert response.headers["content-security-policy"] == CSP
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_jev_rerender_carries_csp() -> None:
    fake = populated_fake()
    fake.put_response = control_pb2.PutJevSchemaResponse(
        validation=control_pb2.JevValidation(ok=False, errors=["bad"])
    )
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == CSP


# -- E11: quick builder ------------------------------------------------------


def test_builder_defaults_equal_builtin_default_semantically() -> None:
    text = build_decision_schema(
        ("choice", "boolean", "score"), ["YES", "NO"], 0.0, 1.0
    )
    result = validate_schema_text(text)
    assert result.ok, result.errors
    assert result.schema is not None
    assert result.schema["oneOf"][0]["properties"]["result"]["enum"] == ["YES", "NO"]
    assert result.sha256 == canonical_hash(load_builtin("default"))


async def test_build_route_renders_generated_schema_without_saving() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/build",
        {
            "name": "",
            "name_editable": "on",
            "base_sha": "",
            "schema_json": "old",
            "decision_choice": "on",
            "decision_boolean": "on",
            "decision_score": "on",
            "choice_options": "YES\nNO",
            "score_min": "0",
            "score_max": "1",
            "include_confidence": "on",
            "additional_properties": "on",
        },
    )
    assert response.status_code == 200
    assert "generated" in response.text
    generated = textarea_content(response.text)
    result = validate_schema_text(generated)
    assert result.ok, result.errors
    assert result.sha256 == canonical_hash(load_builtin("default"))
    assert fake.put_calls == []  # does not save


async def test_build_custom_options_generate_custom_schema() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/build",
        {
            "name": "my-choice",
            "name_editable": "on",
            "base_sha": "",
            "schema_json": "old",
            "decision_choice": "on",
            "choice_options": "YES\nNO\nMAYBE",
            "score_min": "0",
            "score_max": "1",
            "include_confidence": "on",
            "additional_properties": "on",
        },
    )
    assert response.status_code == 200
    generated = textarea_content(response.text)
    result = validate_schema_text(generated)
    assert result.ok, result.errors
    assert result.schema is not None
    assert result.schema["oneOf"][0]["properties"]["result"]["enum"] == [
        "YES",
        "NO",
        "MAYBE",
    ]
    assert len(result.schema["oneOf"]) == 1  # only choice selected
    assert "YES/NO/MAYBE choice" in result.schema["description"]


async def test_build_no_decision_types_inline_error() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/build",
        {
            "name": "",
            "name_editable": "on",
            "base_sha": "",
            "schema_json": "keep-me",
            "choice_options": "YES\nNO",
        },
    )
    assert response.status_code == 200
    assert "pick at least one decision type" in response.text
    assert "keep-me" in textarea_content(response.text)


async def test_build_choice_without_options_inline_error() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/build",
        {"name": "", "base_sha": "", "schema_json": "x", "decision_choice": "on"},
    )
    assert response.status_code == 200
    assert "enter at least one choice option" in response.text


async def test_build_score_min_over_max_inline_error() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/build",
        {
            "name": "",
            "base_sha": "",
            "schema_json": "x",
            "decision_score": "on",
            "score_min": "5",
            "score_max": "1",
        },
    )
    assert response.status_code == 200
    assert "score minimum must not exceed the maximum" in response.text


async def test_build_non_numeric_score_bounds_inline_error() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/build",
        {
            "name": "",
            "base_sha": "",
            "schema_json": "x",
            "decision_score": "on",
            "score_min": "abc",
            "score_max": "1",
        },
    )
    assert response.status_code == 200
    assert "score minimum and maximum must be numbers" in response.text


# -- E12: optimistic concurrency ---------------------------------------------


async def test_save_conflict_rerenders_with_current_text() -> None:
    fake = populated_fake()
    fake.jev_schemas["my-choice"] = make_schema("my-choice", sha=SHA_A, text='{"a": 1}')
    transport = make_transport(fake)
    page = await get(transport, "/jev/my-choice")
    assert hidden_value(page.text, "base_sha") == SHA_A
    # Someone else saved meanwhile.
    fake.jev_schemas["my-choice"] = make_schema("my-choice", sha=SHA_B, text='{"b": 2}')
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": '{"edited": true}', "base_sha": SHA_A},
    )
    assert response.status_code == 200
    assert "changed since you opened it" in response.text
    assert textarea_content(response.text) == '{"b": 2}'
    assert hidden_value(response.text, "base_sha") == SHA_B
    assert fake.put_calls == []


async def test_save_matching_sha_proceeds() -> None:
    fake = populated_fake()
    fake.jev_schemas["my-choice"] = make_schema("my-choice", sha=SHA_A)
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": SHA_A},
    )
    assert response.status_code == 303
    assert fake.put_calls == [("my-choice", VALID_SCHEMA)]


async def test_save_new_schema_without_base_sha_skips_conflict_check() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": VALID_SCHEMA, "base_sha": ""},
    )
    assert response.status_code == 303
    assert fake.put_calls == [("my-choice", VALID_SCHEMA)]


# -- E13: HEAD ---------------------------------------------------------------


async def test_head_jev_200_empty_body() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.head("/jev")
    assert response.status_code == 200
    assert response.content == b""


# -- editor pages ------------------------------------------------------------


async def test_new_page_minimal_template_and_empty_name() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/jev/new")
    assert response.status_code == 200
    assert textarea_content(response.text) == MINIMAL_TEMPLATE
    assert 'name="name" id="name" value=""' in response.text
    assert hidden_value(response.text, "base_sha") == ""


async def test_new_save_creates_and_redirects() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/new",
        {"action": "save", "name": "my-choice", "schema_json": VALID_SCHEMA},
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jev/my-choice"
    assert fake.put_calls == [("my-choice", VALID_SCHEMA)]


async def test_copy_link_prefills_name_and_builtin_content() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/jev/new?name=default")
    assert response.status_code == 200
    assert 'name="name" id="name" value="default"' in response.text
    assert textarea_content(response.text) == BUILTIN_DEFAULT_TEXT
    assert hidden_value(response.text, "base_sha") == SHA_A


async def test_edit_page_shows_text_and_base_sha() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/jev/my-choice")
    assert response.status_code == 200
    assert textarea_content(response.text) == VALID_SCHEMA
    assert hidden_value(response.text, "base_sha") == SHA_A
    for label in ("Validate", "Save", "Save &amp; launch", "Launch without saving"):
        assert label in response.text


async def test_edit_page_builtin_readonly_banner_and_copy_link() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/jev/default")
    assert response.status_code == 200
    assert "read-only" in response.text
    assert "Create editable copy" in response.text
    assert 'href="/jev/new?name=default"' in response.text
    assert "readonly" in response.text
    assert 'name="action"' not in response.text


async def test_crlf_normalised_before_put() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/my-choice",
        {"action": "save", "schema_json": '{\r\n  "a": 1\r\n}', "base_sha": SHA_A},
    )
    assert response.status_code == 303
    assert fake.put_calls == [("my-choice", '{\n  "a": 1\n}')]


# -- launch / detach routes --------------------------------------------------


async def test_launch_route_happy_redirects_to_job() -> None:
    fake = populated_fake()
    fake.load_script = [make_event("j-9", message="jev: probing")]
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/launch",
        {"alias": "qwen2.5-coder-7b", "schema_name": "yes-no", "force": "on"},
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/j-9"
    alias, force, jev_mode, spec = fake.load_calls[0]
    assert (alias, force) == ("qwen2.5-coder-7b", True)
    assert jev_mode == control_pb2.JEV_ON
    assert spec is not None and spec.schema_name == "yes-no"


async def test_launch_route_missing_selections_flash() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    _, page = await post_and_follow(transport, "/jev/launch", {})
    assert "pick a model to launch" in page
    _, page = await post_and_follow(
        transport, "/jev/launch", {"alias": "qwen2.5-coder-7b"}
    )
    assert "pick a schema to launch" in page
    assert fake.load_calls == []


async def test_launch_route_invalid_schema_name_flash() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await post(
        transport,
        "/jev/launch",
        {"alias": "qwen2.5-coder-7b", "schema_name": "Bad Name"},
    )
    assert response.status_code == 303
    assert fake.load_calls == []


async def test_launch_route_unknown_alias_flash() -> None:
    fake = populated_fake()
    fake.load_error = KeyError("nope")
    transport = make_transport(fake)
    _, page = await post_and_follow(
        transport,
        "/jev/launch",
        {"alias": "nope", "schema_name": "yes-no"},
    )
    assert "unknown alias: nope" in page


async def test_launch_route_job_in_progress_flash() -> None:
    fake = populated_fake()
    fake.load_error = JobInProgress("busy")
    transport = make_transport(fake)
    _, page = await post_and_follow(
        transport,
        "/jev/launch",
        {"alias": "qwen2.5-coder-7b", "schema_name": "yes-no"},
    )
    assert "a job is already running" in page


async def test_launch_route_controller_down_flash() -> None:
    fake = populated_fake()
    fake.load_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    _, page = await post_and_follow(
        transport,
        "/jev/launch",
        {"alias": "qwen2.5-coder-7b", "schema_name": "yes-no"},
    )
    assert "controller unreachable" in page


async def test_detach_loads_jev_off_and_redirects() -> None:
    fake = populated_fake()
    fake.load_script = [
        make_event("j-5", state=control_pb2.READY, terminal=True, message="jev: detached")
    ]
    transport = make_transport(fake)
    response = await post(transport, "/jev/detach")
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/j-5"
    alias, force, jev_mode, spec = fake.load_calls[0]
    assert (alias, force) == ("qwen2.5-coder-7b", False)
    assert jev_mode == control_pb2.JEV_OFF
    assert spec is None


async def test_detach_no_model_loaded_flash() -> None:
    fake = populated_fake()
    fake.status_value = control_pb2.Status(active_alias="", state=control_pb2.AVAILABLE)
    transport = make_transport(fake)
    _, page = await post_and_follow(transport, "/jev/detach")
    assert "no model loaded" in page
    assert fake.load_calls == []


async def test_detach_controller_down_flash() -> None:
    fake = populated_fake()
    fake.status_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    _, page = await post_and_follow(transport, "/jev/detach")
    assert "controller unreachable" in page


# -- P2: dashboard / models / job page markers -------------------------------


async def test_dashboard_shows_jev_badge() -> None:
    fake = populated_fake()
    fake.status_value.jev.CopyFrom(
        control_pb2.JevInfo(schema_name="default", sha256=SHA_A, source="store", stale=True)
    )
    transport = make_transport(fake)
    response = await get(transport, "/")
    assert response.status_code == 200
    assert f"Jev: default ({SHA_A[:12]})" in response.text
    assert "stale" in response.text


async def test_dashboard_without_jev_has_no_badge() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    response = await get(transport, "/")
    assert response.status_code == 200
    assert "Jev:" not in response.text


async def test_models_table_shows_jev_marker_on_active_row() -> None:
    fake = populated_fake()
    fake.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        jev=control_pb2.JevInfo(schema_name="default", sha256=SHA_A, source="store"),
    )
    transport = make_transport(fake)
    response = await get(transport, "/models")
    assert response.status_code == 200
    assert '<span class="badge badge-ready">Jev</span>' in response.text


async def test_job_page_renders_error_code_badge() -> None:
    fake = populated_fake()
    fake.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        current_job=control_pb2.Job(
            id="j-1", kind=control_pb2.LOAD, alias="qwen2.5-coder-7b"
        ),
    )
    fake.job_events_map["j-1"] = [
        make_event("j-1", message="jev: probing", kind=control_pb2.LOAD),
        make_event(
            "j-1",
            state=control_pb2.ERROR,
            terminal=True,
            kind=control_pb2.LOAD,
            message="jev: vLLM rejected the schema",
            error_code="jev_probe_failed",
        ),
    ]
    transport = make_transport(fake)
    response = await get(transport, "/jobs/j-1")
    assert response.status_code == 200
    assert "jev_probe_failed" in response.text
