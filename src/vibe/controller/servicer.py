"""gRPC `vibe.v1.ModelController` servicer — the control plane (Phases 1, 4).

Wraps the Phase-0 `JobRunner`, `docker_ctl`, and `gpu` modules in a
`grpc.aio` service (FR-02.1–FR-02.8), plus the Phase-4 download/cache
managers (M4.1/M4.3) and registry registration/reload RPCs (M4.4). Every
blocking Docker call runs through `asyncio.to_thread` (NFR-02.3); long job
streams emit a debug heartbeat at least every 10 s (NFR-02.5); a client
cancelling `LoadModel` or `UnloadModel` never aborts the job (FR-02.11) —
the job runs as an independent task and the RPC stream merely subscribes
to its fan-out queue. Downloads follow the same pattern: they live in the
`DownloadManager`, never in the RPC stream (FR-04.1).

This module never imports `docker` or `pynvml` (XR-5): the test fakes are
plain duck-typed stand-ins.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any, cast

import grpc
import httpx
from pydantic import ValidationError

from vibe.config import Settings
from vibe.controller.cache import CacheScanner
from vibe.controller.downloads import DownloadManager
from vibe.controller.logbuffer import RingBuffer
from vibe.docker_ctl import DockerCtl, DockerError
from vibe.gen.vibe.v1 import control_pb2, control_pb2_grpc
from vibe.gpu import GpuStats
from vibe.jobs import JobEvent, JobInProgress, JobRunner, JobState
from vibe.registry import ModelSpec, Registry

# Job streams emit a debug heartbeat at least this often (NFR-02.5).
HEARTBEAT_SECONDS = 10.0
HEARTBEAT_MESSAGE = "waiting for the job to make progress"

# Per-job fan-out queue capacity. Writes beyond it are dropped so a
# disconnected client can never make the queue grow unbounded (FR-02.11);
# dropped events are still recorded in the shared ring buffer.
FANOUT_CAPACITY = 200

# Bounded per-job event history for `GetJobEvents` (FR-01.4): keep the last
# `JOB_HISTORY_JOBS` jobs and at most `JOB_EVENT_HISTORY` events per job, so
# memory stays flat no matter how long the controller runs (NFR-01.5).
JOB_HISTORY_JOBS = 8
JOB_EVENT_HISTORY = 1000

# Docker `timestamps=True` lines look like
# "2026-09-10T12:34:56.789123456Z message".
_DOCKER_LOG_TS = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(.*)$"
)

# End marker for the (blocking) docker logs iterator, mirroring jobs.py.
_SENTINEL = object()

_TERMINAL_STATES = frozenset({JobState.READY, JobState.ERROR, JobState.AVAILABLE})

_STATE_TO_PROTO: dict[JobState, control_pb2.ModelState] = {
    JobState.AVAILABLE: control_pb2.AVAILABLE,
    JobState.LOADING: control_pb2.LOADING,
    JobState.READY: control_pb2.READY,
    JobState.UNLOADING: control_pb2.UNLOADING,
    JobState.ERROR: control_pb2.ERROR,
}


def model_state_from_job_state(job_state: JobState | str) -> control_pb2.ModelState:
    """Map a `jobs.JobState` (or its string value) to the proto `ModelState`.

    The proto enum members are plain ints at runtime. Raises `ValueError` for
    unknown states.
    """
    state = job_state if isinstance(job_state, JobState) else JobState(job_state)
    return _STATE_TO_PROTO[state]


def job_kind_to_proto(kind: str) -> control_pb2.JobKind:
    """Map a job kind string to the proto `JobKind`; `ValueError` otherwise."""
    if kind == "load":
        return control_pb2.LOAD
    if kind == "unload":
        return control_pb2.UNLOAD
    raise ValueError(f"unknown job kind: {kind!r}")


def to_proto(event: JobEvent, job_id: str) -> control_pb2.JobEvent:
    """Map a `jobs.JobEvent` to the proto `JobEvent`."""
    return control_pb2.JobEvent(
        job_id=job_id,
        ts_unix_ms=int(event.ts.timestamp() * 1000),
        level=event.level,
        message=event.message,
        state=(
            model_state_from_job_state(event.state)
            if event.state is not None
            else control_pb2.MODEL_STATE_UNSPECIFIED
        ),
        progress=event.progress if event.progress is not None else 0.0,
        terminal=event.state in _TERMINAL_STATES,
    )


def _parse_log_line(line: str) -> control_pb2.LogLine:
    """Split a docker-timestamped line into `LogLine`; fall back to now."""
    match = _DOCKER_LOG_TS.match(line)
    if match:
        try:
            ts = datetime.fromisoformat(match.group(1))
        except ValueError:
            pass  # malformed timestamp — fall through to the current time
        else:
            return control_pb2.LogLine(
                ts_unix_ms=int(ts.timestamp() * 1000),
                line=match.group(2),
            )
    now = datetime.now(UTC)
    return control_pb2.LogLine(ts_unix_ms=int(now.timestamp() * 1000), line=line)


def _event_log_line(event: JobEvent) -> control_pb2.LogLine:
    """Render a buffered job event as a log line (StreamLogs fallback)."""
    return control_pb2.LogLine(
        ts_unix_ms=int(event.ts.timestamp() * 1000),
        line=f"[{event.level}] {event.message}",
    )


def _container_status_to_proto(status: Any) -> control_pb2.ModelState:
    """Map a docker container status string to the proto `ModelState`."""
    if not status:
        return control_pb2.MODEL_STATE_UNSPECIFIED
    if "running" in str(status).lower():
        return control_pb2.READY
    return control_pb2.ERROR


def _registration_spec_to_dict(spec: control_pb2.RegistrationSpec) -> dict[str, Any]:
    """Map the proto `RegistrationSpec` to a propose/write dict (FR-04.5).

    Empty strings and zero numbers mean "auto"/unset and are omitted so the
    registry-side defaults apply.
    """
    data: dict[str, Any] = {}
    if spec.hf_repo:
        data["hf_repo"] = spec.hf_repo
    if spec.alias:
        data["alias"] = spec.alias
    if spec.quantization:
        data["quantization"] = spec.quantization
    if spec.max_model_len:
        data["max_model_len"] = int(spec.max_model_len)
    if spec.gpu_memory_utilization:
        data["gpu_memory_utilization"] = float(spec.gpu_memory_utilization)
    if spec.host_port:
        data["host_port"] = int(spec.host_port)
    if spec.tool_parser:
        data["tool_parser"] = spec.tool_parser
    if spec.use_case:
        data["use_case"] = spec.use_case
    if spec.tags:
        data["tags"] = list(spec.tags)
    if spec.display_name:
        data["display_name"] = spec.display_name
    return data


class ModelControllerServicer(control_pb2_grpc.ModelControllerServicer):
    """Server-side implementation of `vibe.v1.ModelController`."""

    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        runner: JobRunner,
        docker: DockerCtl,
        gpu_fn: Callable[[], GpuStats],
        buffer: RingBuffer,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.runner = runner
        self.docker = docker
        self.gpu_fn = gpu_fn
        self.buffer = buffer
        self.downloads = DownloadManager(settings)
        self.cache = CacheScanner(settings.hf_home)
        self.last_error = ""
        self._job_events: OrderedDict[str, deque[JobEvent]] = OrderedDict()

    # -- unary queries -----------------------------------------------------

    async def ListModels(self, request, context):
        """Registry snapshot with per-model state and cache presence (M1.2)."""
        try:
            managed = await asyncio.to_thread(self.docker.find_managed)
        except DockerError:
            managed = []  # Docker unreachable — report states as AVAILABLE
        active_alias: str | None = None
        active_running = False
        if managed:
            try:
                info = await asyncio.to_thread(
                    self.docker.container_info, managed[0]
                )
            except DockerError:
                info = {}
            active_alias = info.get("alias")
            active_running = "running" in str(info.get("status", "")).lower()

        job = self.runner.current_job()

        specs = self.registry.all()
        cached_ids = await asyncio.to_thread(self._cached_repo_ids)

        response = control_pb2.ListModelsResponse()
        for spec in specs:
            state = control_pb2.AVAILABLE
            if active_alias is not None and spec.alias == active_alias:
                # `info["alias"]` is the container's `vibe.alias` label.
                state = control_pb2.READY if active_running else control_pb2.ERROR
            if job is not None and job.alias == spec.alias:
                state = (
                    control_pb2.LOADING
                    if job.kind == "load"
                    else control_pb2.UNLOADING
                )
            model = response.models.add()
            model.alias = spec.alias
            model.display_name = spec.display_name
            model.hf_repo = spec.hf_repo
            model.quantization = spec.quantization
            model.max_model_len = spec.max_model_len
            model.gpu_memory_utilization = spec.gpu_memory_utilization
            model.host_port = spec.host_port
            model.tags.extend(spec.tags)
            model.use_case = spec.use_case
            model.state = state
            model.cached_locally = spec.hf_repo in cached_ids
        return response

    def _cached_repo_ids(self) -> set[str]:
        """HF-hub repo ids with a local snapshot (M4.3, FR-04.9).

        Fed by the same `CacheScanner` used by `ListCache` so the two
        views never disagree; scan errors degrade to an empty set.
        """
        return {repo.repo_id for repo in self.cache.scan()}

    async def GetStatus(self, request, context):
        """Active model, container, in-flight job and last error (FR-02.2)."""
        try:
            status = await self.runner.status()
        except DockerError:
            status = {}  # Docker unreachable — degrade to an empty snapshot

        container_name = status.get("container")
        container_id = ""
        if container_name:
            try:
                managed = await asyncio.to_thread(self.docker.find_managed)
                for container in managed:
                    info = await asyncio.to_thread(
                        self.docker.container_info, container
                    )
                    if info.get("name") == container_name:
                        container_id = str(info.get("id") or "")
                        break
            except DockerError:
                container_id = ""  # Docker unreachable — degrade gracefully

        started_at = status.get("started_at")
        started_at_unix = 0
        if started_at:
            try:
                started_at_unix = int(
                    datetime.fromisoformat(str(started_at)).timestamp()
                )
            except ValueError:
                started_at_unix = 0  # unparseable container timestamp

        host_port = status.get("host_port") or 0
        try:
            host_port = int(host_port)
        except (TypeError, ValueError):
            host_port = 0

        current_job = None
        job = self.runner.current_job()
        if job is not None:
            current_job = control_pb2.Job(
                id=job.id,
                kind=job_kind_to_proto(job.kind),
                alias=job.alias or "",
                started_at_unix=int(job.started_at.timestamp()),
                done=False,
                ok=False,
            )

        return control_pb2.Status(
            active_alias=status.get("active_alias") or "",
            state=_container_status_to_proto(status.get("state")),
            container_id=container_id,
            host_port=host_port,
            started_at_unix=started_at_unix,
            current_job=current_job,
            last_error=self.last_error,
        )

    async def GetGpuStats(self, request, context):
        """GPU memory/utilization snapshot (FR-02.7).

        GPU stats are best-effort: when NVML and `nvidia-smi` are both
        unavailable (e.g. no GPU present), return zeros rather than failing.
        """
        try:
            stats = await asyncio.to_thread(self.gpu_fn)
        except RuntimeError:
            stats = GpuStats(used_mib=0, total_mib=0, utilization_pct=0, processes=())
        result = control_pb2.GpuStats(
            used_mib=stats.used_mib,
            total_mib=stats.total_mib,
            utilization_pct=stats.utilization_pct,
        )
        for process in stats.processes:
            entry = result.processes.add()
            entry.pid = process.pid
            entry.name = process.name
            entry.used_mib = process.used_mib
        return result

    # -- job streams -------------------------------------------------------

    async def LoadModel(self, request, context):
        """Stream a load job until a terminal event (FR-02.3)."""
        alias = request.alias
        try:
            self.registry.get(alias)
        except KeyError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"unknown alias '{alias}'"
            )
            return

        def start() -> AsyncIterator[JobEvent]:
            return self.runner.load(
                alias,
                force=request.force,
                keep_failed=request.keep_failed,
                timeout_s=request.timeout_s or None,
            )

        async for event in self._stream_job(alias, context, start):
            yield event

    async def UnloadModel(self, request, context):
        """Stream an unload job; idempotent via the runner (FR-02.4)."""
        async for event in self._stream_job(None, context, self.runner.unload):
            yield event

    @staticmethod
    def _offer(queue: asyncio.Queue, item: tuple) -> None:
        """Push one fan-out item, dropping it when the queue is full.

        The queue is bounded so a slow or disconnected consumer can never
        leak memory (FR-02.11); dropped events still land in the ring buffer.
        """
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def _record_job_event(self, job_id: str, event: JobEvent) -> None:
        """Append one event to the per-job history for `GetJobEvents`.

        History is bounded (NFR-01.5): `JOB_EVENT_HISTORY` events per job,
        `JOB_HISTORY_JOBS` jobs in total with the oldest job evicted first.
        Events without a job id are dropped.
        """
        if not job_id:
            return
        self._job_events.setdefault(job_id, deque(maxlen=JOB_EVENT_HISTORY)).append(
            event
        )
        while len(self._job_events) > JOB_HISTORY_JOBS:
            self._job_events.popitem(last=False)

    async def _pump_job(
        self, queue: asyncio.Queue, events: AsyncIterator[JobEvent]
    ) -> None:
        """Consume one runner job generator into the fan-out queue.

        Runs as its own task so the job survives client disconnects
        (FR-02.11). Events are also recorded in the shared ring buffer and
        the per-job history (`GetJobEvents`), and terminal events update
        `self.last_error`.
        """
        job_id = ""
        terminal_seen = False
        try:
            async for event in events:
                job = self.runner.current_job()
                if job is not None:
                    job_id = job.id
                self.buffer.append(event)
                self._record_job_event(job_id, event)
                if event.state in _TERMINAL_STATES:
                    terminal_seen = True
                    self._update_last_error(event)
                self._offer(queue, ("event", event, job_id))
        except (JobInProgress, KeyError) as exc:
            # Raised by the runner's acquire/alias lookup before any event.
            self._offer(queue, ("startup_error", exc))
        except asyncio.CancelledError:
            raise  # server shutdown — never mask task cancellation
        except Exception as exc:  # noqa: BLE001
            # An error that escaped the runner's own ERROR-event handling:
            # surface it as a terminal ERROR event instead of dying silently.
            print(f"vibe controller: job stream failed: {exc}", file=sys.stderr)
            if not terminal_seen:
                event = JobEvent(
                    datetime.now(UTC),
                    "error",
                    f"unexpected error: {exc}",
                    state=JobState.ERROR,
                )
                self.buffer.append(event)
                self._record_job_event(job_id, event)
                self._update_last_error(event)
                self._offer(queue, ("event", event, job_id))

    def _update_last_error(self, event: JobEvent) -> None:
        """Track the last terminal error for `GetStatus` (cleared on success)."""
        if event.state is JobState.ERROR:
            self.last_error = event.message
        else:
            self.last_error = ""

    async def _abort_in_progress(self, context: Any) -> None:
        """Abort with FAILED_PRECONDITION naming the in-flight job (FR-02.3)."""
        job = self.runner.current_job()
        job_id = getattr(job, "id", "") or ""
        if job_id:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"a job is already in flight (job {job_id})",
            )
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION, "a job is already in flight"
        )

    async def _stream_job(
        self,
        alias: str | None,
        context: Any,
        start: Callable[[], AsyncIterator[JobEvent]],
    ) -> AsyncIterator[control_pb2.JobEvent]:
        """Shared streaming core for LoadModel/UnloadModel (FR-02.3/FR-02.4).

        The job generator runs in a detached task (FR-02.11); this stream
        subscribes to its fan-out queue, sends heartbeats during long waits
        (NFR-02.5) and ends on the first terminal event or when the
        generator finishes.
        """
        queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=FANOUT_CAPACITY)
        try:
            events = start()
        except JobInProgress:
            await self._abort_in_progress(context)
            return
        except KeyError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"unknown alias '{alias}'"
            )
            return

        task = asyncio.create_task(self._pump_job(queue, events))
        job_id = ""
        while True:
            if task.done() and queue.empty():
                # The pump finished without a terminal event (e.g. the
                # idempotent no-op unload): the stream simply ends here.
                exception = task.exception()
                if exception is not None:
                    print(
                        f"vibe controller: job pump crashed: {exception}",
                        file=sys.stderr,
                    )
                    yield control_pb2.JobEvent(
                        job_id=job_id,
                        ts_unix_ms=int(datetime.now(UTC).timestamp() * 1000),
                        level="error",
                        message=f"job failed: {exception}",
                        state=control_pb2.ERROR,
                        progress=0.0,
                        terminal=True,
                    )
                return
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=HEARTBEAT_SECONDS
                )
            except TimeoutError:
                if context.cancelled():
                    return  # FR-02.11: client gone — the job keeps running
                current = self.runner.current_job()
                if current is not None:
                    job_id = current.id
                yield control_pb2.JobEvent(
                    job_id=job_id,
                    ts_unix_ms=int(datetime.now(UTC).timestamp() * 1000),
                    level="debug",
                    message=HEARTBEAT_MESSAGE,
                )
                continue
            if item[0] == "startup_error":
                exc = item[1]
                if isinstance(exc, JobInProgress):
                    await self._abort_in_progress(context)
                    return
                # KeyError: the alias vanished between the upfront lookup and
                # the runner's own lookup — same status as the upfront check.
                await context.abort(
                    grpc.StatusCode.NOT_FOUND, f"unknown alias '{alias}'"
                )
                return
            event: JobEvent = item[1]
            job_id = item[2]
            yield to_proto(event, job_id)
            if event.state in _TERMINAL_STATES:
                return

    # -- logs ---------------------------------------------------------------

    async def StreamLogs(self, request, context):
        """Container logs with a ring-buffer fallback (FR-02.5)."""
        try:
            managed = await asyncio.to_thread(self.docker.find_managed)
        except DockerError:
            managed = []
        if managed:
            try:
                # The docker logs iterator blocks on reads (NFR-02.3):
                # create it off-loop, then pull each line via a worker thread.
                it = await asyncio.to_thread(
                    self.docker.logs,
                    managed[0],
                    request.tail or None,
                    request.follow,
                )
                while True:
                    line = await asyncio.to_thread(next, it, _SENTINEL)
                    if line is _SENTINEL:
                        return
                    yield _parse_log_line(str(line))
            except DockerError:
                pass  # daemon went away mid-stream — replay the ring buffer
        # No container (or Docker unreachable): replay buffered job events,
        # oldest first, so the tail reads chronologically.
        for event in reversed(self.buffer.recent(request.tail or 100)):
            yield _event_log_line(event)

    # -- chat ---------------------------------------------------------------

    async def Chat(self, request, context):
        """Proxy the active vLLM's `/v1/chat/completions` stream (FR-02.6)."""
        alias = request.alias
        try:
            self.registry.get(alias)
        except KeyError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"unknown alias '{alias}'"
            )
            return

        container = None
        host_port = None
        try:
            managed = await asyncio.to_thread(self.docker.find_managed)
        except DockerError:
            managed = []
        for candidate in managed:
            try:
                info = await asyncio.to_thread(
                    self.docker.container_info, candidate
                )
            except DockerError:
                continue
            # `info["alias"]` is the container's `vibe.alias` label.
            if info.get("alias") != alias:
                continue
            if "running" not in str(info.get("status", "")).lower():
                continue
            container = candidate
            host_port = info.get("host_port")
            break
        if container is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"no model loaded — run vibe load {alias} first",
            )
            return
        try:
            host_port = int(cast(Any, host_port))
        except (TypeError, ValueError):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"no host port reported for {alias} — run vibe load {alias} first",
            )
            return

        payload: dict[str, Any] = {
            "model": alias,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "stream": True,
        }
        if request.HasField("temperature"):
            payload["temperature"] = request.temperature
        if request.HasField("max_tokens"):
            payload["max_tokens"] = request.max_tokens

        url = f"http://127.0.0.1:{host_port}/v1/chat/completions"
        timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                try:
                    if response.status_code != 200:
                        body = (await response.aread()).decode(
                            "utf-8", "replace"
                        )[:500]
                        await context.abort(
                            grpc.StatusCode.UNAVAILABLE,
                            f"model server returned {response.status_code}: {body}",
                        )
                        return
                    async for raw in response.aiter_lines():
                        line = raw.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            return
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue  # keep-alive or malformed SSE line
                        if not isinstance(obj, dict):
                            continue
                        choices = cast(list[Any], obj.get("choices") or [])
                        if not choices:
                            continue
                        choice = cast(dict[str, Any], choices[0])
                        delta = (
                            cast(dict[str, Any], choice.get("delta") or {}).get(
                                "content"
                            )
                            or ""
                        )
                        finish_reason = choice.get("finish_reason") or ""
                        usage = cast(dict[str, Any], obj.get("usage") or {})
                        yield control_pb2.ChatChunk(
                            delta=delta,
                            finish_reason=finish_reason,
                            prompt_tokens=int(
                                cast(Any, usage.get("prompt_tokens")) or 0
                            ),
                            completion_tokens=int(
                                cast(Any, usage.get("completion_tokens")) or 0
                            ),
                        )
                finally:
                    # Closing the response aborts the upstream vLLM
                    # request when this RPC is cancelled (FR-02.11).
                    # `aclose` is idempotent, so this composes with the
                    # context manager's own close on exit.
                    await response.aclose()
        except httpx.HTTPError as exc:
            await context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"cannot reach the model server at {url}: {exc}",
            )

    # -- job event replay ---------------------------------------------------

    async def GetJobEvents(self, request, context):
        """Replay the buffered events for one job, oldest first (FR-01.4).

        Replay-and-return only: the admin job page polls every 2 s and each
        call is a fresh snapshot of the bounded per-job history (NFR-01.5).
        """
        events = self._job_events.get(request.job_id)
        if events is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"unknown job '{request.job_id}'"
            )
            return
        for event in events:
            yield to_proto(event, request.job_id)

    # -- downloads (Phase 4, M4.1) -------------------------------------------

    async def ListDownloads(self, request, context):
        """In-flight plus the last 20 completed downloads (FR-04.2)."""
        response = control_pb2.ListDownloadsResponse()
        for handle in self.downloads.list():
            info = response.downloads.add()
            info.id = handle.id
            info.hf_repo = handle.hf_repo
            info.include.extend(handle.include)
            info.progress = handle.progress
            info.bytes_done = handle.bytes_done
            info.bytes_total = handle.bytes_total
            info.done = handle.done
            info.ok = handle.ok
            info.error = handle.error
        return response

    async def DownloadModel(self, request, context):
        """Stream one download's progress until its terminal event (FR-04.1).

        The download lives in the `DownloadManager`, not this RPC: a client
        disconnect never aborts it (FR-02.11 semantics), and a duplicate
        request for an in-flight repo attaches to the same download.
        """
        try:
            handle = self.downloads.start(
                request.hf_repo,
                list(request.include) or None,
                request.revision or None,
            )
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return
        async for progress in self.downloads.events(handle):
            yield control_pb2.DownloadProgress(
                id=progress.id,
                progress=progress.progress,
                bytes_done=progress.bytes_done,
                bytes_total=progress.bytes_total,
                current_file=progress.current_file,
                terminal=progress.terminal,
                error=progress.error,
            )

    async def CancelDownload(self, request, context):
        """Cancel one download cooperatively; partial files stay (FR-04.2)."""
        return control_pb2.CancelDownloadResponse(
            cancelled=self.downloads.cancel(request.id)
        )

    # -- HF cache (Phase 4, M4.3) -------------------------------------------

    async def ListCache(self, request, context):
        """Cached repos, sizes, registry aliases, and disk space (FR-04.4)."""
        repos = await asyncio.to_thread(self.cache.list, self.registry)
        total, free = await asyncio.to_thread(self.cache.disk_usage)
        response = control_pb2.ListCacheResponse(
            disk_total_bytes=total, disk_free_bytes=free
        )
        for repo in repos:
            entry = response.entries.add()
            entry.repo_id = repo.repo_id
            entry.size_bytes = repo.size_bytes
            entry.last_modified = repo.last_modified
            entry.aliases.extend(repo.aliases)
        return response

    async def RemoveCached(self, request, context):
        """Remove one repo's cache dir; idempotent (FR-04.4)."""
        removed, error = await asyncio.to_thread(self.cache.remove, request.repo_id)
        return control_pb2.RemoveCachedResponse(removed=removed, error=error)

    # -- registry management (Phase 4, M4.4) ---------------------------------

    async def ReloadRegistry(self, request, context):
        """Re-read models.yaml; errors leave the live registry untouched
        (FR-04.6, validate-before-swap)."""
        errors = await asyncio.to_thread(self.registry.reload)
        return control_pb2.ReloadRegistryResponse(errors=list(errors))

    async def ProposeRegistration(self, request, context):
        """Dry-run a registration: fill defaults, list warnings/errors
        (FR-04.5)."""
        data = _registration_spec_to_dict(request.spec)
        return await asyncio.to_thread(self._propose, data)

    def _propose(self, data: dict[str, Any]) -> control_pb2.ProposeRegistrationResponse:
        """Blocking half of `ProposeRegistration` (runs in a worker thread)."""
        entry, warnings, errors = self.registry.propose(data, self.cache)
        response = control_pb2.ProposeRegistrationResponse(
            warnings=list(warnings), errors=list(errors)
        )
        try:
            spec = ModelSpec.model_validate(entry)
        except ValidationError:
            return response  # incomplete entry — the errors carry the details
        model = response.model
        model.alias = spec.alias
        model.display_name = spec.display_name
        model.hf_repo = spec.hf_repo
        model.quantization = spec.quantization
        model.max_model_len = spec.max_model_len
        model.gpu_memory_utilization = spec.gpu_memory_utilization
        model.host_port = spec.host_port
        model.tags.extend(spec.tags)
        model.use_case = spec.use_case
        model.state = control_pb2.AVAILABLE
        model.cached_locally = self.cache.is_cached(spec.hf_repo)
        return response

    async def WriteRegistration(self, request, context):
        """Append a validated entry to models.yaml, then reload it (FR-04.5/
        FR-04.6) so the controller sees the new alias immediately.

        Defaults are filled exactly like `ProposeRegistration` (the proto
        spec carries no container_name/port fields), then the completed
        entry is validated against the on-disk registry and appended
        atomically.
        """
        data = _registration_spec_to_dict(request.spec)
        entry, warnings, errors = await asyncio.to_thread(
            self.registry.propose, data, self.cache
        )
        if not errors:
            errors = await asyncio.to_thread(self.registry.write_entry, entry)
        if not errors:
            errors.extend(await asyncio.to_thread(self.registry.reload))
        return control_pb2.WriteRegistrationResponse(
            ok=not errors, errors=list(errors)
        )
