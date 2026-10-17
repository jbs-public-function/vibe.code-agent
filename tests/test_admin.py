"""Admin UI tests (Phase 3).

The controller is a duck-typed `FakeControllerClient` implementing the full
client contract; requests go through `httpx.ASGITransport` against the
composed FastAPI app. Streaming assertions drive the ASGI app directly with a
`send` recorder because `ASGITransport` buffers the chunked body.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from http.cookies import SimpleCookie
from typing import Any, cast, override
from urllib.parse import unquote

import httpx
import pytest
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from vibe.admin.app import build_app
from vibe.admin.flash import FLASH_MAX_CHARS, get_flash, set_flash
from vibe.admin.routes.chat import truncate_history
from vibe.admin.sessions import (
    MAX_SESSIONS,
    MAX_TURNS,
    SESSION_TTL_SECONDS,
    ChatSession,
    SessionStore,
)
from vibe.client import ControllerClient, ControllerUnavailable
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.jobs import JobInProgress

CSP = (
    "default-src 'self'; script-src 'none'; style-src 'self'; "
    "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)


# -- shared helpers ----------------------------------------------------------


def make_info(
    alias: str,
    *,
    max_len: int = 4096,
    state: control_pb2.ModelState = control_pb2.AVAILABLE,
) -> control_pb2.ModelInfo:
    return control_pb2.ModelInfo(
        alias=alias,
        display_name=f"{alias} display",
        hf_repo=f"org/{alias}",
        quantization="awq",
        max_model_len=max_len,
        host_port=8020,
        state=state,
        cached_locally=True,
    )


def make_event(
    job_id: str,
    *,
    state: control_pb2.ModelState = control_pb2.LOADING,
    message: str = "",
    terminal: bool = False,
    level: str = "INFO",
    progress: float = 0.0,
) -> control_pb2.JobEvent:
    return control_pb2.JobEvent(
        job_id=job_id,
        ts_unix_ms=int(time.time() * 1000),
        level=level,
        message=message,
        state=state,
        progress=progress,
        terminal=terminal,
    )


def make_download_info(
    download_id: str,
    repo: str,
    *,
    done: bool = False,
    ok: bool = True,
    error: str = "",
    progress: float = 0.5,
    bytes_done: int = 500,
    bytes_total: int = 1000,
) -> control_pb2.DownloadInfo:
    return control_pb2.DownloadInfo(
        id=download_id,
        hf_repo=repo,
        include=["*.gguf"],
        progress=progress,
        bytes_done=bytes_done,
        bytes_total=bytes_total,
        done=done,
        ok=ok,
        error=error,
    )


def make_download_progress(
    download_id: str,
    *,
    progress: float = 0.5,
    bytes_done: int = 500,
    bytes_total: int = 1000,
    current_file: str = "model-00001.safetensors",
    terminal: bool = False,
    error: str = "",
) -> control_pb2.DownloadProgress:
    return control_pb2.DownloadProgress(
        id=download_id,
        progress=progress,
        bytes_done=bytes_done,
        bytes_total=bytes_total,
        current_file=current_file,
        terminal=terminal,
        error=error,
    )


class FakeControllerClient(ControllerClient):
    """Scriptable in-memory controller implementing the whole client contract.

    Subclasses the real client so tests can pass it where `ControllerClient`
    is annotated, but never issues a real RPC. `*_error` attributes raise from
    the matching method (from inside async generators, like the real client);
    `load_script`/`unload_script`/`chat_script`/`log_script` mix events with
    floats (sleep seconds).
    """

    def __init__(self) -> None:
        super().__init__(Settings(vllm_image_tag="v0.11.2"))
        self.infos: list[control_pb2.ModelInfo] = []
        self.status_value: control_pb2.Status = control_pb2.Status()
        self.status_error: Exception | None = None
        self.list_error: Exception | None = None
        self.gpu_value: control_pb2.GpuStats = control_pb2.GpuStats()
        self.gpu_error: Exception | None = None
        self.log_lines: list[control_pb2.LogLine] = []
        self.log_script: list[control_pb2.LogLine | float] = []
        self.logs_calls: list[tuple[int | None, bool]] = []
        self.logs_error: Exception | None = None
        self.job_events_map: dict[str, list[control_pb2.JobEvent]] = {}
        self.job_events_error: Exception | None = None
        self.jobs_map: dict[str, control_pb2.Job] = {}
        self.get_job_error: Exception | None = None
        self.load_script: list[control_pb2.JobEvent | float] = []
        self.load_error: Exception | None = None
        self.load_calls: list[tuple[str, bool]] = []
        self.load_consumed: list[control_pb2.JobEvent] = []
        self.unload_script: list[control_pb2.JobEvent | float] = []
        self.unload_error: Exception | None = None
        self.unload_consumed: list[control_pb2.JobEvent] = []
        self.chat_chunks: list[control_pb2.ChatChunk] = []
        self.chat_script: list[control_pb2.ChatChunk | float] = []
        self.chat_error: Exception | None = None
        self.chat_calls: list[dict[str, Any]] = []
        self.download_infos: list[control_pb2.DownloadInfo] = []
        self.downloads_error: Exception | None = None
        self.download_script: list[control_pb2.DownloadProgress] = []
        self.download_error: Exception | None = None
        self.download_calls: list[tuple[str, list[str], str | None]] = []
        self.download_consumed: list[control_pb2.DownloadProgress] = []
        self.cancel_calls: list[str] = []
        self.cancel_error: Exception | None = None
        self.cache_value: control_pb2.ListCacheResponse = control_pb2.ListCacheResponse()
        self.cache_error: Exception | None = None
        self.remove_calls: list[str] = []
        self.remove_result: tuple[bool, str] = (True, "")
        self.remove_error: Exception | None = None
        self.reload_errors: list[str] = []
        self.reload_error: Exception | None = None
        self.reload_calls: int = 0
        self.propose_response: control_pb2.ProposeRegistrationResponse = (
            control_pb2.ProposeRegistrationResponse()
        )
        self.propose_error: Exception | None = None
        self.propose_calls: list[control_pb2.RegistrationSpec] = []
        self.write_response: control_pb2.WriteRegistrationResponse = (
            control_pb2.WriteRegistrationResponse(ok=True)
        )
        self.write_error: Exception | None = None
        self.write_calls: list[control_pb2.RegistrationSpec] = []
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
    async def load(
        self,
        alias: str,
        force: bool = False,
        keep_failed: bool = False,
        timeout_s: int | None = None,
        jev_mode: int = 0,
        jev: control_pb2.JevSpec | None = None,
    ) -> AsyncIterator[control_pb2.JobEvent]:
        self.load_calls.append((alias, force))
        if self.load_error is not None:
            raise self.load_error
        for item in self.load_script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            event = cast(control_pb2.JobEvent, item)
            self.load_consumed.append(event)
            yield event

    @override
    async def unload(self) -> AsyncIterator[control_pb2.JobEvent]:
        if self.unload_error is not None:
            raise self.unload_error
        for item in self.unload_script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            event = cast(control_pb2.JobEvent, item)
            self.unload_consumed.append(event)
            yield event

    @override
    async def job_events(
        self, job_id: str, *, follow: bool = False
    ) -> AsyncIterator[control_pb2.JobEvent]:
        if self.job_events_error is not None:
            raise self.job_events_error
        if job_id not in self.job_events_map:
            raise KeyError(job_id)
        for event in self.job_events_map[job_id]:
            yield event

    @override
    async def get_job(self, job_id: str) -> control_pb2.Job:
        if self.get_job_error is not None:
            raise self.get_job_error
        if job_id not in self.jobs_map:
            raise KeyError(job_id)
        return self.jobs_map[job_id]

    @override
    async def logs(
        self, tail: int | None = None, follow: bool = False
    ) -> AsyncIterator[control_pb2.LogLine]:
        self.logs_calls.append((tail, follow))
        if self.logs_error is not None:
            raise self.logs_error
        if self.log_script:
            for item in self.log_script:
                if isinstance(item, float):
                    await asyncio.sleep(item)
                    continue
                yield cast(control_pb2.LogLine, item)
            return
        for line in self.log_lines:
            yield line

    @override
    async def chat(
        self,
        alias: str,
        messages: Sequence[control_pb2.ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[control_pb2.ChatChunk]:
        self.chat_calls.append(
            {
                "alias": alias,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        script = self.chat_script if self.chat_script else self.chat_chunks
        for item in script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            yield cast(control_pb2.ChatChunk, item)
        if self.chat_error is not None:
            # Simulates the controller dying mid-stream: chunks already
            # yielded reach the page, then the stream errors out.
            raise self.chat_error

    @override
    async def close(self) -> None:
        self.closed = True
        await super().close()

    # Phase 4 client contract (FR-04.3..04.5): added by the controller half;
    # mirrored here with the exact signatures so the admin routes type-check.
    async def list_downloads(self) -> list[control_pb2.DownloadInfo]:
        if self.downloads_error is not None:
            raise self.downloads_error
        return list(self.download_infos)

    async def download(
        self,
        hf_repo: str,
        include: Sequence[str] | None = None,
        revision: str | None = None,
    ) -> AsyncIterator[control_pb2.DownloadProgress]:
        self.download_calls.append((hf_repo, list(include or []), revision))
        if self.download_error is not None:
            raise self.download_error
        for event in self.download_script:
            self.download_consumed.append(event)
            yield event

    async def cancel_download(self, download_id: str) -> bool:
        self.cancel_calls.append(download_id)
        if self.cancel_error is not None:
            raise self.cancel_error
        return True

    async def list_cache(self) -> control_pb2.ListCacheResponse:
        if self.cache_error is not None:
            raise self.cache_error
        return self.cache_value

    async def remove_cached(self, repo_id: str) -> tuple[bool, str]:
        self.remove_calls.append(repo_id)
        if self.remove_error is not None:
            raise self.remove_error
        return self.remove_result

    async def reload_registry(self) -> list[str]:
        self.reload_calls += 1
        if self.reload_error is not None:
            raise self.reload_error
        return list(self.reload_errors)

    async def propose_registration(
        self, spec: control_pb2.RegistrationSpec
    ) -> control_pb2.ProposeRegistrationResponse:
        self.propose_calls.append(spec)
        if self.propose_error is not None:
            raise self.propose_error
        return self.propose_response

    async def write_registration(
        self, spec: control_pb2.RegistrationSpec
    ) -> control_pb2.WriteRegistrationResponse:
        self.write_calls.append(spec)
        if self.write_error is not None:
            raise self.write_error
        return self.write_response


def populated_fake() -> FakeControllerClient:
    """A fake with a ready model, GPU stats and job/log data for page tests."""
    fake = FakeControllerClient()
    now_ms = int(time.time() * 1000)
    fake.infos = [
        make_info("qwen2.5-coder-7b"),
        make_info("llama-3.1-8b"),
        make_info("deepseek-r1-14b", max_len=7984),
    ]
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b",
        state=control_pb2.READY,
        host_port=8020,
        container_id="c" * 12,
        started_at_unix=int(time.time()) - 60,
    )
    fake.gpu_value = control_pb2.GpuStats(
        used_mib=12000,
        total_mib=24576,
        utilization_pct=48,
        processes=[control_pb2.GpuProcess(pid=1234, name="python", used_mib=9000)],
    )
    fake.job_events_map["j-ok"] = [
        make_event("j-ok", message="pulling weights"),
        make_event("j-ok", state=control_pb2.READY, terminal=True, message="ready"),
    ]
    fake.job_events_map["j-err"] = [
        make_event("j-err", message="pulling"),
        make_event("j-err", state=control_pb2.ERROR, terminal=True, message="boom"),
    ]
    fake.log_lines = [
        control_pb2.LogLine(ts_unix_ms=now_ms, line="INFO: server started"),
        control_pb2.LogLine(ts_unix_ms=now_ms, line="ERROR: CUDA out of memory"),
    ]
    fake.download_infos = [make_download_info("dl-1", "org/repo-a")]
    fake.cache_value = control_pb2.ListCacheResponse(
        entries=[
            control_pb2.CacheEntry(
                repo_id="org/repo-a",
                size_bytes=2 * 1024**3,
                last_modified="2026-09-15",
            )
        ],
        disk_total_bytes=100 * 1024**3,
        disk_free_bytes=40 * 1024**3,
    )
    return fake


def make_app(fake: FakeControllerClient, *, gateway_status: int = 200) -> Any:
    """Compose the app with a mocked gateway probe (IMP-04.24).

    `vibe_bind="test"` puts the `Host: test` used by the httpx clients into
    the middleware's allow-list; the `MockTransport` stops the dashboard probe
    from dialing the real gateway.
    """
    app = build_app(Settings(vllm_image_tag="v0.11.2", vibe_bind="test"), fake)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(gateway_status, text="ok")

    app.state.gateway_transport = httpx.MockTransport(handler)
    return app


def make_transport(
    fake: FakeControllerClient, *, gateway_status: int = 200
) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=make_app(fake, gateway_status=gateway_status))


async def drive(
    app: Any,
    method: str,
    path: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, list[tuple[bytes, bytes]], list[bytes]]:
    """Drive the ASGI app directly, recording headers and body chunk order.

    `httpx.ASGITransport` buffers the streamed body into one blob, so chunk
    boundaries (the NFR-01.6 initial flush) must be observed at the ASGI level.
    """
    messages: list[dict[str, Any]] = []
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # A real client keeps the connection open while reading the streamed
        # response; StreamingResponse cancels this wait when the stream ends.
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
        ],
        "scheme": "http",
        "server": ("127.0.0.1", 8081),
        "client": ("127.0.0.1", 12345),
        "root_path": "",
    }
    await app(scope, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    chunks = [m["body"] for m in messages if m["type"] == "http.response.body"]
    return start["status"], start["headers"], chunks


def extract_hidden(text: str) -> dict[str, str]:
    """The hidden form fields of a rendered register preview."""
    return dict(
        re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', text)
    )


# -- pages: script-free + security headers (NFR-01.1) ------------------------


ADMIN_PAGES = [
    "/",
    "/models",
    "/jobs/j-ok",
    "/jobs/j-err",
    "/jobs/missing",
    "/chat",
    "/logs",
    "/developer",
    "/developer/register",
    "/developer/downloads/dl-1",
    "/developer/cache/org/repo-a/confirm",
]


@pytest.mark.parametrize("path", ADMIN_PAGES)
async def test_pages_are_script_free_and_carry_security_headers(path: str) -> None:
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 200
    assert "<script" not in response.text
    assert "style=" not in response.text
    # IMP-04.26: every link is root-relative (the gateway URL lives in <code>).
    assert 'href="http' not in response.text
    assert response.headers["content-security-policy"] == CSP
    assert response.headers["x-content-type-options"] == "nosniff"


# -- dashboard (FR-01.1) ------------------------------------------------------


async def test_dashboard_shows_status_gpu_and_job() -> None:
    fake = populated_fake()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b",
        state=control_pb2.READY,
        host_port=8020,
        container_id="c" * 12,
        started_at_unix=int(time.time()) - 60,
        current_job=control_pb2.Job(
            id="j-ok", kind=control_pb2.LOAD, alias="llama-3.1-8b"
        ),
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    text = response.text
    assert response.status_code == 200
    assert "llama-3.1-8b" in text
    assert "READY" in text
    assert "12,000" in text and "24,576" in text
    assert "48%" in text
    assert "python" in text
    assert "bar-45" in text  # 12000/24576 = 48.8% → 45% step class
    assert '<meta http-equiv="refresh" content="5">' in text
    assert 'href="/jobs/j-ok"' in text
    assert "http://localhost:8080/v1" in text
    assert 'aria-label="memory"' in text


async def test_gateway_probe_uses_injected_transport() -> None:
    up = make_transport(populated_fake(), gateway_status=200)
    async with httpx.AsyncClient(transport=up, base_url="http://test") as client:
        ok = await client.get("/")
    assert "Gateway</dt><dd>ok" in ok.text

    down = make_transport(populated_fake(), gateway_status=503)
    async with httpx.AsyncClient(transport=down, base_url="http://test") as client:
        not_ok = await client.get("/")
    assert "Gateway</dt><dd>down" in not_ok.text


async def test_gateway_probe_transport_error_counts_as_down() -> None:
    app = make_app(populated_fake())

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    app.state.gateway_transport = httpx.MockTransport(boom)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "Gateway</dt><dd>down" in response.text


# -- security middleware (IMP-04.1, IMP-04.28) --------------------------------


async def test_cross_origin_post_rejected_same_origin_passes() -> None:
    fake = FakeControllerClient()
    fake.unload_script = [make_event("j-u", state=control_pb2.AVAILABLE, terminal=True)]
    app = make_app(fake)
    # A page served from elsewhere POSTing at the admin: rejected.
    status, _, _ = await drive(
        app, "POST", "/models/unload",
        headers={"origin": "http://evil.example", "host": "test"},
    )
    assert status == 403
    # The same POST from the admin's own origin passes.
    status, _, _ = await drive(
        app, "POST", "/models/unload",
        headers={"origin": "http://test", "host": "test"},
    )
    assert status == 303
    # Sec-Fetch-Site: cross-site rejected; none and same-origin allowed.
    status, _, _ = await drive(
        app, "POST", "/models/unload",
        headers={"sec-fetch-site": "cross-site", "host": "test"},
    )
    assert status == 403
    status, _, _ = await drive(
        app, "POST", "/models/unload",
        headers={"sec-fetch-site": "none", "host": "test"},
    )
    assert status == 303
    # No Origin and no Sec-Fetch-Site (curl): allowed by design.
    status, _, _ = await drive(
        app, "POST", "/models/unload", headers={"host": "test"}
    )
    assert status == 303
    # Referer fallback: foreign host rejected, same host passes.
    status, _, _ = await drive(
        app, "POST", "/models/unload",
        headers={"referer": "http://evil.example/x", "host": "test"},
    )
    assert status == 403
    status, _, _ = await drive(
        app, "POST", "/models/unload",
        headers={"referer": "http://test/x", "host": "test"},
    )
    assert status == 303


async def test_host_outside_allow_list_rejected() -> None:
    app = make_app(populated_fake())
    status, _, _ = await drive(app, "GET", "/", headers={"host": "evil.example"})
    assert status == 403
    status, _, _ = await drive(app, "GET", "/", headers={"host": "127.0.0.1:8081"})
    assert status == 200
    status, _, _ = await drive(app, "GET", "/", headers={"host": "[::1]:8081"})
    assert status == 200


async def test_head_request_supported() -> None:
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.head("/")
    assert response.status_code == 200
    assert response.content == b""


# -- models load/unload (FR-01.3) ---------------------------------------------


async def test_load_redirects_and_background_task_completes() -> None:
    fake = FakeControllerClient()
    fake.load_script = [
        make_event("j-load", message="starting"),
        0.02,
        make_event("j-load", message="pulling"),
        0.02,
        make_event("j-load", state=control_pb2.READY, terminal=True, message="done"),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/models/load", data={"alias": "llama-3.1-8b"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/jobs/j-load"
        # The job keeps draining after the HTTP response is done: the
        # remaining events are consumed by a background task (FR-01.3).
        for _ in range(200):
            if len(fake.load_consumed) == 3:
                break
            await asyncio.sleep(0.01)
        assert len(fake.load_consumed) == 3
        assert fake.load_calls == [("llama-3.1-8b", False)]


async def test_load_passes_force_flag() -> None:
    fake = FakeControllerClient()
    fake.load_script = [make_event("j-f", state=control_pb2.READY, terminal=True)]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/models/load",
            data={"alias": "llama-3.1-8b", "force": "on"},
            follow_redirects=False,
        )
    assert response.status_code == 303
    assert fake.load_calls == [("llama-3.1-8b", True)]


async def test_unload_redirects_to_job() -> None:
    fake = FakeControllerClient()
    fake.unload_script = [make_event("j-un", state=control_pb2.AVAILABLE, terminal=True)]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/models/unload", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/j-un"


async def test_load_job_in_progress_redirects_to_existing_job_with_flash() -> None:
    fake = FakeControllerClient()
    fake.load_error = JobInProgress("busy")
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b",
        state=control_pb2.LOADING,
        current_job=control_pb2.Job(
            id="j-existing", kind=control_pb2.LOAD, alias="llama-3.1-8b"
        ),
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/models/load", data={"alias": "deepseek-r1-14b"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/jobs/j-existing"
        followed = await client.get("/jobs/j-existing")
    assert "a job is already running" in followed.text


async def test_load_unknown_alias_flashes_and_redirects() -> None:
    fake = FakeControllerClient()
    fake.load_error = KeyError("nope")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/models/load", data={"alias": "nope"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/models"
        followed = await client.get("/models")
    assert "unknown alias: nope" in followed.text


# -- job page (FR-01.4) -------------------------------------------------------


async def test_job_page_refreshes_while_running_only() -> None:
    fake = populated_fake()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b",
        state=control_pb2.LOADING,
        current_job=control_pb2.Job(
            id="j-ok",
            kind=control_pb2.LOAD,
            alias="llama-3.1-8b",
            started_at_unix=int(time.time()) - 5,
        ),
    )
    fake.job_events_map["j-ok"] = [
        make_event("j-ok", message="pulling weights"),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        running = await client.get("/jobs/j-ok")
        assert '<meta http-equiv="refresh" content="2">' in running.text
        assert "LOAD" in running.text
        assert "llama-3.1-8b" in running.text
        # Once the last event is terminal the refresh stops.
        fake.job_events_map["j-ok"] = [
            make_event("j-ok", message="pulling weights"),
            make_event("j-ok", state=control_pb2.READY, terminal=True, message="ready"),
        ]
        terminal = await client.get("/jobs/j-ok")
    assert 'http-equiv="refresh" content="2"' not in terminal.text
    assert "Job finished successfully" in terminal.text
    assert "READY" in terminal.text


async def test_job_page_error_shows_container_logs() -> None:
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/jobs/j-err")
    assert response.status_code == 200
    assert "boom" in response.text
    assert "CUDA out of memory" in response.text
    assert 'http-equiv="refresh"' not in response.text


async def test_job_page_unknown_job_is_not_found() -> None:
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/jobs/nope")
    assert response.status_code == 200
    assert "job not found" in response.text.lower()
    assert 'http-equiv="refresh"' not in response.text


async def test_job_page_shows_kind_alias_from_events_or_get_job() -> None:
    """A CLI-started job carries kind/alias on its events (IMP-04.13)."""
    fake = FakeControllerClient()
    now_ms = int(time.time() * 1000)
    fake.job_events_map["cli-1"] = [
        control_pb2.JobEvent(
            job_id="cli-1",
            ts_unix_ms=now_ms,
            level="INFO",
            message="downloading",
            kind=control_pb2.DOWNLOAD,
            alias="org/repo",
        ),
    ]
    # Fallback: events without kind/alias → GetJob fills them in (IMP-01.2).
    fake.job_events_map["cli-2"] = [make_event("cli-2", message="loading")]
    fake.jobs_map["cli-2"] = control_pb2.Job(
        id="cli-2",
        kind=control_pb2.LOAD,
        alias="llama-3.1-8b",
        started_at_unix=int(time.time()) - 10,
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        from_events = await client.get("/jobs/cli-1")
        assert "DOWNLOAD" in from_events.text
        assert "org/repo" in from_events.text
        from_job = await client.get("/jobs/cli-2")
        assert "LOAD" in from_job.text
        assert "llama-3.1-8b" in from_job.text


async def test_job_page_refreshes_while_controller_down_and_notes_unknown() -> None:
    fake = FakeControllerClient()
    fake.status_error = ControllerUnavailable("down")
    fake.job_events_map["j-1"] = [make_event("j-1", message="pulling")]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        down = await client.get("/jobs/j-1")
        # IMP-04.14: the controller may come back — keep refreshing.
        assert '<meta http-equiv="refresh" content="2">' in down.text

    # Non-terminal job that is no longer current, controller up: unknown.
    fake.status_error = None
    fake.status_value = control_pb2.Status(state=control_pb2.AVAILABLE)
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        orphaned = await client.get("/jobs/j-1")
    assert "outcome is unknown" in orphaned.text
    assert '<meta http-equiv="refresh" content="2">' not in orphaned.text


# -- chat (FR-01.5/01.6, NFR-01.6) --------------------------------------------


async def test_chat_stream_escapes_and_flushes_header_first() -> None:
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b", state=control_pb2.READY
    )
    fake.chat_chunks = [
        control_pb2.ChatChunk(delta="hello "),
        control_pb2.ChatChunk(delta="<script>alert(1)</script>"),
        control_pb2.ChatChunk(delta="!"),
    ]
    app = make_app(fake)
    status, headers, chunks = await drive(
        app,
        "POST",
        "/chat",
        body=b"message=hi&temperature=0.5&max_tokens=512",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert status == 200
    header_dict = dict(headers)
    assert header_dict[b"content-type"].startswith(b"text/html")
    assert header_dict[b"cache-control"] == b"no-store"
    assert header_dict[b"content-security-policy"] == CSP.encode()
    assert "samesite=lax" in header_dict[b"set-cookie"].decode().lower()
    body = b"".join(chunks)
    # Escaped form present, raw payload never appears.
    assert b"&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert b"<script>alert" not in body
    assert b"<script" not in body
    # The initial flush (head/nav/history + open pre) precedes the first delta.
    assert b'<pre id="stream">' in chunks[0]
    assert b"hello " not in chunks[0]
    assert b"hello " in body
    assert b"</pre>" in body
    assert re.search(rb"reply took \d+\.\d{2} s", body)
    # Form fields parsed manually from the urlencoded body.
    call = fake.chat_calls[0]
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 512
    assert [message.role for message in call["messages"]] == ["user"]
    assert call["messages"][0].content == "hi"
    # The finished turn is stored for the next page load.
    cookie = header_dict[b"set-cookie"].decode().split(";")[0]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        reloaded = await client.get("/chat", headers={"cookie": cookie})
    assert "hello " in reloaded.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in reloaded.text


async def test_chat_stream_appends_error_block_when_controller_dies_midstream() -> None:
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b", state=control_pb2.READY
    )
    fake.chat_chunks = [control_pb2.ChatChunk(delta="partial")]
    fake.chat_error = ControllerUnavailable("gone")  # raises after the first chunk
    app = make_app(fake)
    status, _, chunks = await drive(
        app, "POST", "/chat", body=b"message=hi"
    )
    assert status == 200  # never a 500 once streaming has started
    body = b"".join(chunks)
    assert b"partial" in body
    assert b"controller unreachable" in body
    assert b"</pre>" in body
    assert re.search(rb"reply took \d+\.\d{2} s", body)  # elapsed renders even on failure


async def test_chat_runtime_error_midstream_renders_message() -> None:
    """IMP-04.5: a real upstream error surfaces with its message."""
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b", state=control_pb2.READY
    )
    fake.chat_chunks = [control_pb2.ChatChunk(delta="partial")]
    fake.chat_error = RuntimeError("vLLM 400: context too long")
    app = make_app(fake)
    status, _, chunks = await drive(app, "POST", "/chat", body=b"message=hi")
    assert status == 200
    body = b"".join(chunks)
    assert b"partial" in body
    assert b"vLLM 400: context too long" in body
    assert b"controller unreachable" not in body


async def test_chat_first_token_after_slow_prefill_still_streams() -> None:
    """IMP-04.4: a >3 s prefill streams with keep-alive padding, not an error."""
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b", state=control_pb2.READY
    )
    fake.chat_script = [5.2, control_pb2.ChatChunk(delta="late reply")]
    app = make_app(fake)
    status, _, chunks = await drive(app, "POST", "/chat", body=b"message=hi")
    assert status == 200
    body = b"".join(chunks)
    assert b"late reply" in body
    assert b"controller unreachable" not in body
    assert b"<!-- keep-alive -->" in body


async def test_chat_page_shows_no_model_loaded_when_not_ready() -> None:
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(active_alias="", state=control_pb2.AVAILABLE)
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/chat")
    assert response.status_code == 200
    assert "no model loaded" in response.text.lower()
    assert 'href="/models"' in response.text


async def test_chat_not_ready_does_not_persist_user_turn() -> None:
    """IMP-04.6: a failed send must not leave a dangling user turn."""
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(active_alias="", state=control_pb2.AVAILABLE)
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat", data={"message": "hi"}, follow_redirects=False
        )
        assert response.status_code == 303
        reloaded = await client.get("/chat")
    assert "No messages yet" in reloaded.text
    assert '<div class="role">user' not in reloaded.text


async def test_chat_reset_clears_the_conversation() -> None:
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b", state=control_pb2.READY
    )
    fake.chat_chunks = [control_pb2.ChatChunk(delta="ok")]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/chat", data={"message": "hi"})
        response = await client.post("/chat/reset", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/chat"
        reloaded = await client.get("/chat")
    assert "No messages yet" in reloaded.text


async def test_chat_clamps_temperature_and_max_tokens() -> None:
    """IMP-04.10: out-of-range values are clamped server-side."""
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b", state=control_pb2.READY
    )
    fake.infos = [make_info("llama-3.1-8b", max_len=2048)]
    fake.chat_chunks = [control_pb2.ChatChunk(delta="ok")]
    app = make_app(fake)
    status, _, _ = await drive(
        app, "POST", "/chat", body=b"message=hi&temperature=9&max_tokens=999999"
    )
    assert status == 200
    call = fake.chat_calls[0]
    assert call["temperature"] == 2.0
    assert call["max_tokens"] == 2048


async def test_chat_truncation_drops_oldest_turns_and_notes_in_footer() -> None:
    """IMP-04.7: history is truncated at 75 % and the footer says so."""
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(
        active_alias="llama-3.1-8b", state=control_pb2.READY
    )
    fake.infos = [make_info("llama-3.1-8b", max_len=400)]  # budget = 300 tokens
    fake.chat_chunks = [control_pb2.ChatChunk(delta="ok")]
    app = make_app(fake)
    store = cast(SessionStore, app.state.sessions)
    session = store.get_or_create(None)
    for i in range(5):  # ~101 tokens each → 505 total, 3 must go
        session.append(
            {"role": "user" if i % 2 else "assistant", "content": f"turn {i} " + "z" * 400}
        )
    status, _, chunks = await drive(
        app,
        "POST",
        "/chat",
        body=b"message=hi",
        headers={"cookie": f"vibe_session={session.session_id}"},
    )
    assert status == 200
    body = b"".join(chunks)
    assert b"older turns omitted" in body
    assert b"turn 0" not in body  # oldest non-system turns were dropped


# -- logs (FR-01.7) -----------------------------------------------------------


async def test_logs_page_shows_lines_escaped_with_tail_selector() -> None:
    fake = FakeControllerClient()
    fake.log_lines = [
        control_pb2.LogLine(ts_unix_ms=int(time.time() * 1000), line="line one"),
        control_pb2.LogLine(ts_unix_ms=int(time.time() * 1000), line="<script>x</script>"),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/logs", params={"tail": 100})
    assert response.status_code == 200
    assert "line one" in response.text
    assert "&lt;script&gt;x&lt;/script&gt;" in response.text
    assert "<script" not in response.text
    assert '<meta http-equiv="refresh" content="5">' in response.text
    assert 'name="tail"' in response.text
    assert fake.logs_calls == [(100, False)]


async def test_logs_page_invalid_tail_renders_html_error() -> None:
    """IMP-04.12: ?tail=abc is an HTML error, not a FastAPI 422 JSON."""
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/logs", params={"tail": "abc"})
        assert page.status_code == 400
        assert "tail must be a number" in page.text
        stream = await client.get("/logs/stream", params={"tail": "abc"})
        assert stream.status_code == 400
        assert "tail must be a number" in stream.text


async def test_logs_stream_is_chunked_text_plain() -> None:
    fake = FakeControllerClient()
    fake.log_lines = [
        control_pb2.LogLine(ts_unix_ms=int(time.time() * 1000), line="streamed line"),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/logs/stream", params={"tail": "0", "follow": "1"}
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "streamed line" in response.text
    assert fake.logs_calls == [(0, True)]


async def test_logs_stream_defaults_to_replay_tail() -> None:
    """IMP-04.11: the default first chunk is a 50-line replay."""
    fake = FakeControllerClient()
    fake.log_lines = [
        control_pb2.LogLine(ts_unix_ms=int(time.time() * 1000), line="replayed line"),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/logs/stream")
    assert response.status_code == 200
    assert "replayed line" in response.text
    assert fake.logs_calls == [(50, True)]


async def test_logs_stream_quiet_container_stays_open() -> None:
    """IMP-04.11: a >3 s quiet stream is not killed by a first-line deadline."""
    fake = FakeControllerClient()
    now_ms = int(time.time() * 1000)
    fake.log_script = [
        4.0,
        control_pb2.LogLine(ts_unix_ms=now_ms, line="late line"),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/logs/stream", params={"tail": "0", "follow": "1"}
        )
    assert response.status_code == 200
    assert "late line" in response.text
    assert "controller unreachable" not in response.text
    assert fake.logs_calls == [(0, True)]


# -- flash (FR-01.11) ----------------------------------------------------------


async def test_flash_rendered_once_then_cleared() -> None:
    fake = FakeControllerClient()
    fake.load_error = KeyError("nope")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/models/load", data={"alias": "nope"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert "vibe_flash" in response.headers.get("set-cookie", "")
        first = await client.get("/models")
        assert "unknown alias: nope" in first.text
        assert "vibe_flash" in first.headers.get("set-cookie", "")
        second = await client.get("/models")
        assert "unknown alias: nope" not in second.text


def test_flash_roundtrip_special_chars_and_truncation() -> None:
    message = 'quote"s;semicolons ✓ unicode'
    response = StarletteResponse()
    set_flash(response, message)
    assert "samesite=lax" in response.headers["set-cookie"].lower()
    jar = SimpleCookie()
    jar.load(response.headers["set-cookie"])
    raw = jar["vibe_flash"].value
    request = StarletteRequest(
        {"type": "http", "headers": [(b"cookie", f"vibe_flash={raw}".encode())]}
    )
    assert get_flash(request) == message

    # A >4 KiB message is truncated so the cookie always fits (IMP-04.3).
    big = "✓" * 5000
    response2 = StarletteResponse()
    set_flash(response2, big)
    jar2 = SimpleCookie()
    jar2.load(response2.headers["set-cookie"])
    raw2 = jar2["vibe_flash"].value
    assert len(raw2) <= 3500
    assert len(unquote(raw2)) <= FLASH_MAX_CHARS


# -- controller down (NFR-01.5) ------------------------------------------------


async def test_controller_down_renders_banner_not_500() -> None:
    fake = FakeControllerClient()
    fake.status_error = ControllerUnavailable("down")
    fake.gpu_error = ControllerUnavailable("down")
    fake.list_error = ControllerUnavailable("down")
    fake.logs_error = ControllerUnavailable("down")
    fake.job_events_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ["/", "/models", "/chat", "/logs", "/jobs/j-1", "/developer"]:
            response = await client.get(path)
            assert response.status_code == 200
            assert "controller unreachable" in response.text.lower()


async def test_controller_runtime_error_shows_message_banner() -> None:
    """IMP-04.15: a real controller error gets its message, not a 500."""
    fake = FakeControllerClient()
    fake.status_error = RuntimeError("registry failed to load models.yaml")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "registry failed to load models.yaml" in response.text
    assert "Controller error" in response.text


CONTROLLER_DOWN_POSTS = [
    ("/models/load", {"alias": "llama-3.1-8b"}),
    ("/models/unload", {}),
    ("/chat", {"message": "hi"}),
    ("/developer/downloads", {"repo": "org/repo"}),
    ("/developer/register", {"repo": "org/repo", "alias": "x"}),
    ("/developer/register/confirm", {"repo": "org/repo", "alias": "x"}),
    ("/developer/downloads/dl-1/cancel", {}),
    ("/developer/cache/org/repo/remove", {}),
]


@pytest.mark.parametrize("path,data", CONTROLLER_DOWN_POSTS)
async def test_controller_down_on_every_post_route_flashes(
    path: str, data: dict[str, str]
) -> None:
    fake = FakeControllerClient()
    fake.status_error = ControllerUnavailable("down")
    fake.load_error = ControllerUnavailable("down")
    fake.unload_error = ControllerUnavailable("down")
    fake.download_error = ControllerUnavailable("down")
    fake.cancel_error = ControllerUnavailable("down")
    fake.remove_error = ControllerUnavailable("down")
    fake.propose_error = ControllerUnavailable("down")
    fake.write_error = ControllerUnavailable("down")
    fake.reload_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(path, data=data, follow_redirects=False)
        assert response.status_code == 303
        followed = await client.get(response.headers["location"])
    assert "controller unreachable" in followed.text.lower()


# -- health (FR-01.9) ----------------------------------------------------------


async def test_health_reports_controller_state() -> None:
    fake = populated_fake()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.get("/health")
        assert ok.json() == {"admin": "ok", "controller": "ok"}
        fake.status_error = ControllerUnavailable("down")
        down = await client.get("/health")
        assert down.json() == {"admin": "ok", "controller": "down"}


# -- developer page (FR-04.7, FR-04.9) ---------------------------------------


async def test_developer_page_shows_downloads_cache_disk_and_zed(monkeypatch) -> None:
    fake = populated_fake()
    fake.download_infos = [
        make_download_info("dl-1", "org/repo-a"),
        make_download_info("dl-2", "org/repo-b", done=True, ok=True, progress=1.0),
    ]
    fake.cache_value = control_pb2.ListCacheResponse(
        entries=[
            control_pb2.CacheEntry(
                repo_id="org/repo-a",
                size_bytes=3 * 1024**3,
                last_modified="2026-09-15 10:00",
                aliases=["qwen2.5-coder-7b"],
            )
        ],
        disk_total_bytes=512 * 1024**3,
        disk_free_bytes=100 * 1024**3,
    )
    monkeypatch.setattr(
        "vibe.admin.routes.developer.zed_settings_json",
        lambda models: json.dumps(
            {"profiles": {"vibe-local": {"models": [m.alias for m in models]}}}
        ),
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/developer")
    text = response.text
    assert response.status_code == 200
    assert "dl-1" in text and "org/repo-a" in text and "org/repo-b" in text
    assert "DONE" in text and "DOWNLOADING" in text
    assert "3.0 GiB" in text
    assert "512.0 GiB total / 100.0 GiB free" in text
    assert "qwen2.5-coder-7b" in text
    assert "vibe-local" in text
    assert "Reload registry" in text
    assert "<script" not in text


async def test_start_download_redirects_and_background_drains() -> None:
    fake = FakeControllerClient()
    fake.download_script = [
        make_download_progress("dl-9", progress=0.0, bytes_done=0, current_file="a.safetensors"),
        make_download_progress("dl-9", progress=1.0, bytes_done=1000, terminal=True, current_file=""),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/developer/downloads",
            data={"repo": "org/repo", "include": "*.gguf\n*.json", "revision": "main"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/developer/downloads/dl-9"
        # The download keeps draining after the HTTP response is done (FR-04.7a).
        for _ in range(200):
            if len(fake.download_consumed) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(fake.download_consumed) == 2
        assert fake.download_calls == [("org/repo", ["*.gguf", "*.json"], "main")]


async def test_start_download_without_repo_flashes() -> None:
    fake = FakeControllerClient()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/developer/downloads", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/developer"
    assert fake.download_calls == []


async def test_download_page_refreshes_until_done_then_cancel() -> None:
    fake = FakeControllerClient()
    fake.download_infos = [
        make_download_info(
            "dl-1", "org/repo", progress=0.45, bytes_done=450, bytes_total=1000
        )
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        running = await client.get("/developer/downloads/dl-1")
        assert '<meta http-equiv="refresh" content="2">' in running.text
        assert "bar-45" in running.text
        assert "450 B / 1000 B" in running.text
        # Once done the refresh stops and the success block shows.
        fake.download_infos = [
            make_download_info(
                "dl-1", "org/repo", done=True, ok=True, progress=1.0,
                bytes_done=1000, bytes_total=1000,
            )
        ]
        done = await client.get("/developer/downloads/dl-1")
        assert 'http-equiv="refresh"' not in done.text
        assert "Download finished successfully" in done.text
        # Error view: done, ok=False, with the controller's error text.
        fake.download_infos = [
            make_download_info(
                "dl-1", "org/repo", done=True, ok=False, error="boom",
                progress=0.2, bytes_done=200, bytes_total=1000,
            )
        ]
        failed = await client.get("/developer/downloads/dl-1")
        assert 'http-equiv="refresh"' not in failed.text
        assert "download failed: boom" in failed.text
        # Cancel POSTs back to the same page.
        cancel = await client.post(
            "/developer/downloads/dl-1/cancel", follow_redirects=False
        )
        assert cancel.status_code == 303
        assert cancel.headers["location"] == "/developer/downloads/dl-1"
        assert fake.cancel_calls == ["dl-1"]


async def test_download_page_unknown_id_is_not_found() -> None:
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/developer/downloads/nope")
    assert response.status_code == 200
    assert "download not found" in response.text.lower()
    assert 'http-equiv="refresh"' not in response.text


async def test_download_cancel_unknown_id_flashes() -> None:
    """IMP-04.21: cancelling an unknown download flashes instead of 500ing."""
    fake = FakeControllerClient()
    fake.cancel_error = KeyError("nope")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/developer/downloads/dl-9/cancel", follow_redirects=False
        )
        assert response.status_code == 303
        followed = await client.get(response.headers["location"])
    assert "unknown download" in followed.text


async def test_cache_confirm_and_remove() -> None:
    fake = FakeControllerClient()
    fake.cache_value = control_pb2.ListCacheResponse(
        entries=[
            control_pb2.CacheEntry(
                repo_id="org/repo-a", size_bytes=2 * 1024**3, last_modified="2026-09-15"
            )
        ]
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        confirm = await client.get("/developer/cache/org/repo-a/confirm")
        assert confirm.status_code == 200
        assert "org/repo-a" in confirm.text
        assert '/developer/cache/org/repo-a/remove"' in confirm.text
        removed = await client.post(
            "/developer/cache/org/repo-a/remove", follow_redirects=False
        )
        assert removed.status_code == 303
        assert removed.headers["location"] == "/developer"
        assert fake.remove_calls == ["org/repo-a"]
        followed = await client.get("/developer")
    assert "removed org/repo-a from the HF cache" in followed.text


async def test_cache_remove_unknown_repo_flashes() -> None:
    """IMP-04.21: removing an unknown repo flashes instead of 500ing."""
    fake = FakeControllerClient()
    fake.remove_error = KeyError("nope")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/developer/cache/org/nope/remove", follow_redirects=False
        )
        assert response.status_code == 303
        followed = await client.get("/developer")
    assert "unknown repo" in followed.text


# -- registration (FR-04.7d) --------------------------------------------------


async def test_register_flow_preview_then_confirm() -> None:
    fake = FakeControllerClient()
    fake.propose_response = control_pb2.ProposeRegistrationResponse(
        model=control_pb2.ModelInfo(
            alias="qwen2.5-coder-1.5b",
            display_name="Qwen2.5 Coder 1.5B",
            hf_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            quantization="none",
            max_model_len=32768,
            gpu_memory_utilization=0.6,
            host_port=8011,
            use_case="code gen",
            tags=["code", "tools"],
            state=control_pb2.AVAILABLE,
            cached_locally=True,
            yaml_preview=(
                "models:\n"
                "- alias: qwen2.5-coder-1.5b\n"
                "  display_name: Qwen2.5 Coder 1.5B\n"
                "  hf_repo: Qwen/Qwen2.5-Coder-1.5B-Instruct\n"
                "  container_name: vibe-qwen2.5-coder-1.5b\n"
                "  extra_args:\n"
                "  - --enable-auto-tool-choice\n"
                "  - --tool-call-parser hermes\n"
            ),
        ),
        warnings=["vram estimate: 9.4 GiB"],
    )
    transport = make_transport(fake)
    form_data = {
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "alias": "qwen2.5-coder-1.5b",
        "quantization": "none",
        "max_model_len": "",
        "gpu_mem": "",
        "port": "",
        "tool_parser": "hermes",
        "use_case": "code gen",
        "tags": "code, tools",
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        preview = await client.post("/developer/register", data=form_data)
        assert preview.status_code == 200
        assert "<pre" in preview.text
        assert "container_name: vibe-qwen2.5-coder-1.5b" in preview.text
        assert "--tool-call-parser" in preview.text
        assert "warning: vram estimate: 9.4 GiB" in preview.text
        assert "Confirm and reload registry" in preview.text
        # The confirm round-trips the controller-filled hidden values, as the
        # browser does (IMP-04.22).
        hidden = extract_hidden(preview.text)
        assert hidden["port"] == "8011"
        confirm = await client.post(
            "/developer/register/confirm", data=hidden, follow_redirects=False
        )
        assert confirm.status_code == 303
        assert confirm.headers["location"] == "/models"
        followed = await client.get("/models")
    assert "alias qwen2.5-coder-1.5b added" in followed.text
    assert [spec.alias for spec in fake.propose_calls] == [
        "qwen2.5-coder-1.5b",
        "qwen2.5-coder-1.5b",
    ]
    assert [spec.alias for spec in fake.write_calls] == ["qwen2.5-coder-1.5b"]
    assert fake.write_calls[0].tool_parser == "hermes"
    assert list(fake.write_calls[0].tags) == ["code", "tools"]
    assert fake.reload_calls == 1


async def test_register_propose_errors_rerender_form() -> None:
    fake = FakeControllerClient()
    fake.propose_response = control_pb2.ProposeRegistrationResponse(
        errors=["alias 'bad!' must match ^[a-z0-9][a-z0-9.-]*$", "host_port 8020 already in use"]
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/developer/register", data={"repo": "org/repo", "alias": "bad!"}
        )
    assert response.status_code == 200
    assert "host_port 8020 already in use" in response.text
    assert 'name="alias"' in response.text  # the form re-renders
    assert "Confirm and reload registry" not in response.text
    assert fake.write_calls == []
    assert fake.reload_calls == 0


async def test_register_form_parse_errors_rerender_form() -> None:
    """IMP-04.18: "80a0"/"0.7x" re-render with messages instead of coercing."""
    fake = FakeControllerClient()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/developer/register",
            data={
                "repo": "org/repo",
                "alias": "x",
                "max_model_len": "80a0",
                "gpu_mem": "0.7x",
            },
        )
    assert response.status_code == 200
    assert "max model len must be a whole number" in response.text
    assert "GPU mem fraction must be a number" in response.text
    assert 'name="alias"' in response.text  # the form re-renders
    assert "Confirm and reload registry" not in response.text
    assert fake.propose_calls == []
    assert fake.write_calls == []


async def test_register_form_prefills_repo_and_suggests_alias() -> None:
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/developer/register", params={"repo": "org/my-model"}
        )
    assert response.status_code == 200
    assert 'value="org/my-model"' in response.text
    assert 'value="my-model"' in response.text


async def test_register_confirm_detects_drift_from_preview() -> None:
    """IMP-04.22: alias/port cannot drift between preview and write."""
    fake = FakeControllerClient()
    fake.propose_response = control_pb2.ProposeRegistrationResponse(
        model=control_pb2.ModelInfo(
            alias="qwen",
            hf_repo="org/repo",
            quantization="none",
            max_model_len=32768,
            gpu_memory_utilization=0.6,
            host_port=8011,
        ),
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        preview = await client.post(
            "/developer/register",
            data={"repo": "org/repo", "alias": "qwen", "quantization": "none"},
        )
        hidden = extract_hidden(preview.text)
        # The controller now proposes a different port — confirm must refuse.
        fake.propose_response = control_pb2.ProposeRegistrationResponse(
            model=control_pb2.ModelInfo(
                alias="qwen",
                hf_repo="org/repo",
                quantization="none",
                max_model_len=32768,
                gpu_memory_utilization=0.6,
                host_port=8022,
            ),
        )
        confirm = await client.post("/developer/register/confirm", data=hidden)
    assert confirm.status_code == 200
    assert "changed since the preview" in confirm.text
    assert "8022" in confirm.text
    assert fake.write_calls == []


async def test_register_confirm_write_failure_rerenders_with_reload() -> None:
    fake = FakeControllerClient()
    fake.propose_response = control_pb2.ProposeRegistrationResponse(
        model=control_pb2.ModelInfo(alias="x", hf_repo="org/repo", quantization="none"),
    )
    fake.write_response = control_pb2.WriteRegistrationResponse(
        ok=False, errors=["host_port 8020 already in use"]
    )
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/developer/register/confirm",
            data={"repo": "org/repo", "alias": "x", "quantization": "none"},
        )
    assert response.status_code == 200
    assert "host_port 8020 already in use" in response.text
    assert "Reload registry" in response.text
    assert fake.reload_calls == 0


async def test_register_confirm_reload_failure_shows_reload_button() -> None:
    """IMP-04.17: write-succeeds-then-reload-fails offers the recovery button."""
    fake = FakeControllerClient()
    fake.propose_response = control_pb2.ProposeRegistrationResponse(
        model=control_pb2.ModelInfo(alias="x", hf_repo="org/repo", quantization="none"),
    )
    fake.reload_errors = ["invalid yaml: duplicate alias"]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/developer/register/confirm",
            data={"repo": "org/repo", "alias": "x", "quantization": "none"},
        )
        assert response.status_code == 200
        assert "invalid yaml: duplicate alias" in response.text
        assert "Reload registry" in response.text
        assert fake.reload_calls == 1
        # The standalone button recovers on retry.
        fake.reload_errors = []
        retry = await client.post("/developer/reload", follow_redirects=False)
        assert retry.status_code == 303
        followed = await client.get("/developer")
    assert "registry reloaded" in followed.text
    assert fake.reload_calls == 2


async def test_reload_registry_button_flashes_outcome() -> None:
    fake = FakeControllerClient()
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/developer/reload", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/developer"
        followed = await client.get("/developer")
    assert "registry reloaded" in followed.text
    assert fake.reload_calls == 1


async def test_models_page_download_button_only_for_uncached() -> None:
    fake = FakeControllerClient()
    fake.infos = [
        make_info("cached-one"),
        control_pb2.ModelInfo(
            alias="uncached",
            display_name="Uncached",
            hf_repo="org/uncached",
            quantization="none",
            max_model_len=4096,
            host_port=0,
            state=control_pb2.AVAILABLE,
            cached_locally=False,
        ),
        control_pb2.ModelInfo(
            alias="loading",
            display_name="Loading",
            hf_repo="org/loading",
            quantization="none",
            max_model_len=4096,
            host_port=0,
            state=control_pb2.LOADING,
            cached_locally=False,
        ),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/models")
    text = response.text
    assert response.status_code == 200
    assert text.count('name="repo"') == 1
    assert 'name="repo" value="org/uncached"' in text
    assert 'name="repo" value="org/loading"' not in text
    assert 'name="repo" value="org/cached-one"' not in text
    assert 'aria-label="Download uncached"' in text


async def test_developer_controller_down_banner_not_500() -> None:
    fake = FakeControllerClient()
    fake.status_error = ControllerUnavailable("down")
    fake.downloads_error = ControllerUnavailable("down")
    fake.cache_error = ControllerUnavailable("down")
    fake.list_error = ControllerUnavailable("down")
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/developer")
    assert response.status_code == 200
    assert "controller unreachable" in response.text.lower()
    assert "<script" not in response.text


# -- session store (IMP-04.9) --------------------------------------------------


def test_truncate_history_drops_oldest_turns_at_budget() -> None:
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "a" * 400},  # ~100 tokens
        {"role": "assistant", "content": "b" * 800},  # ~200 tokens
        {"role": "user", "content": "c" * 400},  # ~100 tokens
    ]
    kept, dropped = truncate_history(messages, 300)
    assert dropped == 2
    assert [turn["role"] for turn in kept] == ["system", "user"]
    assert kept[-1]["content"] == "c" * 400
    # System turns are never dropped.
    only_system = [{"role": "system", "content": "x" * 4000}]
    kept, dropped = truncate_history(only_system, 100)
    assert dropped == 0
    assert kept == only_system


def test_session_store_sweeps_by_last_seen_not_created_at() -> None:
    store = SessionStore()
    session = store.get_or_create(None)
    # Created long ago but used recently: survives the sweep.
    session.created_at = time.time() - 2 * SESSION_TTL_SECONDS
    session.last_seen = time.time()
    again = store.get_or_create(session.session_id)
    assert again.session_id == session.session_id
    # Idle past the TTL: swept on the next access.
    session.last_seen = time.time() - SESSION_TTL_SECONDS - 1
    fresh = store.get_or_create(session.session_id)
    assert fresh.session_id != session.session_id


def test_session_store_lru_cap_evicts_least_recently_used() -> None:
    store = SessionStore()
    ids = [store.get_or_create(None).session_id for _ in range(MAX_SESSIONS)]
    # Touch the first session so it becomes most-recently-used.
    assert store.get_or_create(ids[0]).session_id == ids[0]
    store.get_or_create(None)  # one over the cap → evicts the LRU (ids[1])
    assert len(store._sessions) == MAX_SESSIONS
    assert ids[0] in store._sessions
    assert ids[1] not in store._sessions


def test_chat_session_turn_cap_drops_oldest_non_system() -> None:
    now = time.time()
    session = ChatSession("s", now, now)
    for i in range(MAX_TURNS + 10):
        session.append({"role": "user" if i % 2 else "assistant", "content": str(i)})
    assert len(session.messages) == MAX_TURNS
    assert session.messages[0]["content"] == "10"


# -- lifespan ---------------------------------------------------------------


async def test_lifespan_closes_http_client_and_controller() -> None:
    fake = FakeControllerClient()
    app = make_app(fake)
    async with app.router.lifespan_context(app):
        pass
    assert app.state.http.is_closed
    assert fake.closed
