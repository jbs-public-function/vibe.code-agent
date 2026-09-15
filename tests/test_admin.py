"""Admin UI tests (Phase 3).

The controller is a duck-typed `FakeControllerClient` implementing the full
client contract; requests go through `httpx.ASGITransport` against the
composed FastAPI app. Streaming assertions drive the ASGI app directly with a
`send` recorder because `ASGITransport` buffers the chunked body.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast, override

import httpx
import pytest

from vibe.admin.app import build_app
from vibe.client import ControllerClient, ControllerUnavailable
from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2
from vibe.jobs import JobInProgress

CSP = "default-src 'self'; script-src 'none'; style-src 'self'"


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
    `load_script`/`unload_script` mix `JobEvent`s with floats (sleep seconds).
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
        self.logs_calls: list[tuple[int | None, bool]] = []
        self.logs_error: Exception | None = None
        self.job_events_map: dict[str, list[control_pb2.JobEvent]] = {}
        self.job_events_error: Exception | None = None
        self.load_script: list[control_pb2.JobEvent | float] = []
        self.load_error: Exception | None = None
        self.load_calls: list[tuple[str, bool]] = []
        self.load_consumed: list[control_pb2.JobEvent] = []
        self.unload_script: list[control_pb2.JobEvent | float] = []
        self.unload_error: Exception | None = None
        self.unload_consumed: list[control_pb2.JobEvent] = []
        self.chat_chunks: list[control_pb2.ChatChunk] = []
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

    # Added by the Phase 3 controller work with exactly this signature.
    async def job_events(self, job_id: str) -> AsyncIterator[control_pb2.JobEvent]:
        if self.job_events_error is not None:
            raise self.job_events_error
        if job_id not in self.job_events_map:
            raise KeyError(job_id)
        for event in self.job_events_map[job_id]:
            yield event

    @override
    async def logs(
        self, tail: int | None = None, follow: bool = False
    ) -> AsyncIterator[control_pb2.LogLine]:
        self.logs_calls.append((tail, follow))
        if self.logs_error is not None:
            raise self.logs_error
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
        if self.chat_error is not None:
            # Simulates the controller dying mid-stream: chunks already
            # yielded reach the page, then the stream errors out.
            for chunk in self.chat_chunks:
                yield chunk
            raise self.chat_error
        for chunk in self.chat_chunks:
            yield chunk

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
    return fake


def make_app(fake: FakeControllerClient) -> Any:
    return build_app(Settings(vllm_image_tag="v0.11.2"), fake)


def make_transport(fake: FakeControllerClient) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=make_app(fake))


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
]


@pytest.mark.parametrize("path", ADMIN_PAGES)
async def test_pages_are_script_free_and_carry_security_headers(path: str) -> None:
    transport = make_transport(populated_fake())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 200
    assert "<script" not in response.text
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
    assert 'href="http://test/jobs/j-ok"' in text
    assert "http://localhost:8080/v1" in text


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


async def test_chat_page_shows_no_model_loaded_when_not_ready() -> None:
    fake = FakeControllerClient()
    fake.status_value = control_pb2.Status(active_alias="", state=control_pb2.AVAILABLE)
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/chat")
    assert response.status_code == 200
    assert "no model loaded" in response.text.lower()
    assert 'href="http://test/models"' in response.text


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


async def test_logs_stream_is_chunked_text_plain() -> None:
    fake = FakeControllerClient()
    fake.log_lines = [
        control_pb2.LogLine(ts_unix_ms=int(time.time() * 1000), line="streamed line"),
    ]
    transport = make_transport(fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/logs/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "streamed line" in response.text
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
        assert "450 / 1,000" in running.text
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
        confirm = await client.post(
            "/developer/register/confirm", data=form_data, follow_redirects=False
        )
        assert confirm.status_code == 303
        assert confirm.headers["location"] == "/models"
        followed = await client.get("/models")
    assert "alias qwen2.5-coder-1.5b added" in followed.text
    assert [spec.alias for spec in fake.propose_calls] == ["qwen2.5-coder-1.5b"]
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
