"""Streamed, cancellable long-running operations (FR-00.4 semantics, XR-4)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from vibe.config import Settings
from vibe.docker_ctl import (
    ALIAS_LABEL,
    ContainerExited,
    DockerCtl,
    DockerError,
    ReadinessTimeout,
)
from vibe.gpu import GpuStats, get_gpu_stats
from vibe.registry import ModelSpec, Registry

# Marks the end of the (blocking) docker logs iterator in `JobRunner.logs`.
_SENTINEL = object()


class JobState(str, Enum):
    """Job lifecycle states (ADR-5)."""

    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    READY = "READY"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"


@dataclass
class Job:
    """A long-running operation tracked by the runner (XR-4)."""

    id: str
    kind: str  # "load" | "unload"
    alias: str | None
    state: JobState
    started_at: datetime


@dataclass
class JobEvent:
    """One streamed progress event of a job."""

    ts: datetime
    level: str  # "info" | "warn" | "error"
    message: str
    state: JobState | None = None
    progress: float | None = None  # reserved for download jobs (Phase 4)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation; timestamps as ISO 8601 strings."""
        return {
            "ts": self.ts.isoformat(),
            "level": self.level,
            "message": self.message,
            "state": self.state.value if self.state is not None else None,
            "progress": self.progress,
        }


class JobInProgress(RuntimeError):
    """A job is already running; a second concurrent one is rejected."""

    def __init__(self, message: str = "a job is already running") -> None:
        super().__init__(message)


def _now() -> datetime:
    """Current UTC timestamp for every job and event timestamp."""
    return datetime.now(UTC)


class JobRunner:
    """Runs streamed jobs with a single in-flight guard (M0.4)."""

    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        docker: DockerCtl | None = None,
        get_gpu_stats: Callable[[], GpuStats] = get_gpu_stats,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.docker = docker if docker is not None else DockerCtl(settings)
        self.get_gpu_stats = get_gpu_stats
        self._job: Job | None = None

    # -- in-flight job -----------------------------------------------------

    def current_job(self) -> Job | None:
        """Return the running job, or None."""
        return self._job

    def _acquire(self, kind: str, alias: str | None) -> Job:
        """Register a new job, rejecting a second concurrent one."""
        current = self._job
        if current is not None:
            raise JobInProgress(
                f"job {current.id} ({current.kind} {current.alias or ''}) is already running"
            )
        job = Job(
            id=uuid.uuid4().hex[:8],
            kind=kind,
            alias=alias,
            state=JobState.LOADING if kind == "load" else JobState.UNLOADING,
            started_at=_now(),
        )
        self._job = job
        return job

    # -- jobs --------------------------------------------------------------

    async def load(
        self,
        alias: str,
        force: bool = False,
        keep_failed: bool = False,
        timeout_s: int | None = None,
    ) -> AsyncIterator[JobEvent]:
        """Stream a load job: purge, start, wait for readiness (FR-00.5).

        Raises KeyError for an unknown alias (the CLI maps it to exit code 3).
        """
        spec: ModelSpec = self.registry.get(alias)
        job = self._acquire("load", alias)
        timeout_s = timeout_s or self.settings.vibe_load_timeout
        container: Any = None  # docker_ctl exports no container type (XR-5)
        outcome = JobState.ERROR
        try:
            if not force:
                managed = await asyncio.to_thread(self.docker.find_managed)
                for existing in managed:
                    if existing.labels.get(ALIAS_LABEL) != alias:
                        continue
                    info = await asyncio.to_thread(
                        self.docker.container_info, existing
                    )
                    # docker status strings: "running" | "exited" | "created" | ...
                    if "running" not in str(info.get("status", "")).lower():
                        continue
                    yield JobEvent(
                        _now(),
                        "info",
                        f"{alias} is already READY — use --force to reload",
                        state=JobState.READY,
                    )
                    outcome = JobState.READY
                    return

            yield JobEvent(
                _now(),
                "info",
                "purging managed and legacy containers",
                state=JobState.UNLOADING,
            )
            purged = await asyncio.to_thread(self.docker.purge_all)
            if purged:
                yield JobEvent(
                    _now(),
                    "info",
                    f"removed {len(purged)} container(s): {', '.join(purged)}",
                )
            else:
                yield JobEvent(_now(), "info", "no containers to purge")

            yield JobEvent(
                _now(),
                "info",
                f"starting {spec.alias} ({spec.hf_repo})",
                state=JobState.LOADING,
            )
            container = await asyncio.to_thread(self.docker.start, spec)
            yield JobEvent(
                _now(),
                "info",
                f"container {container.id[:12]} running — waiting for readiness"
                f" (up to {timeout_s}s)",
            )
            await self.docker.wait_ready(spec, timeout_s)
            yield JobEvent(
                _now(),
                "info",
                f"{spec.alias} READY at http://127.0.0.1:{spec.host_port}/v1",
                state=JobState.READY,
            )
            outcome = JobState.READY
        except (ContainerExited, ReadinessTimeout) as exc:
            message = str(exc)
            last_logs = getattr(exc, "last_logs", None)
            if last_logs:
                message = f"{message}\n{last_logs}"
            if not keep_failed and container is not None:
                try:
                    await asyncio.to_thread(self.docker.stop, container)
                except Exception:  # noqa: BLE001
                    # IMP-02.8: best-effort cleanup — an SDK APIError (or
                    # anything else) during `stop` must never mask the real
                    # cause of the failure.
                    pass
            yield JobEvent(_now(), "error", message, state=JobState.ERROR)
            raise
        except DockerError as exc:
            yield JobEvent(_now(), "error", str(exc), state=JobState.ERROR)
            raise
        except Exception as exc:
            yield JobEvent(
                _now(), "error", f"unexpected error: {exc}", state=JobState.ERROR
            )
            raise
        finally:
            job.state = outcome
            self._job = None

    async def unload(self) -> AsyncIterator[JobEvent]:
        """Stream an unload job: purge all managed/legacy containers (FR-00.6)."""
        job = self._acquire("unload", None)
        outcome = JobState.ERROR
        try:
            yield JobEvent(
                _now(),
                "info",
                "purging managed and legacy containers",
                state=JobState.UNLOADING,
            )
            purged = await asyncio.to_thread(self.docker.purge_all)
            if purged:
                yield JobEvent(
                    _now(),
                    "info",
                    f"removed {len(purged)} container(s): {', '.join(purged)}",
                )
                yield JobEvent(
                    _now(), "info", "all models unloaded", state=JobState.AVAILABLE
                )
            else:
                # IMP-02.7: the no-op unload must still terminate — with
                # state=AVAILABLE so clients can tell "done" from
                # "disconnected" (FR-02.4).
                yield JobEvent(
                    _now(),
                    "info",
                    "nothing to unload — no managed or legacy containers found",
                    state=JobState.AVAILABLE,
                )
            outcome = JobState.AVAILABLE
        except DockerError as exc:
            yield JobEvent(_now(), "error", str(exc), state=JobState.ERROR)
            raise
        finally:
            job.state = outcome
            self._job = None

    # -- queries -----------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """Snapshot of the active container and GPU state (FR-00.7)."""
        managed = await asyncio.to_thread(self.docker.find_managed)
        active: dict[str, Any] = {
            "active_alias": None,
            "state": None,
            "container": None,
            "started_at": None,
            "host_port": None,
            "image": None,
        }
        if managed:
            info = await asyncio.to_thread(self.docker.container_info, managed[0])
            started_at = info.get("started_at")
            if started_at is not None:
                started_at = (
                    started_at.isoformat()
                    if isinstance(started_at, datetime)
                    else str(started_at)
                )
            host_port = info.get("host_port")
            active = {
                "active_alias": info.get("alias"),
                "state": info.get("status"),
                "container": info.get("name"),
                "started_at": started_at,
                "host_port": int(host_port) if host_port is not None else None,
                "image": info.get("image"),
            }

        gpu: dict[str, Any] | None = None
        gpu_error: str | None = None
        try:
            stats = await asyncio.to_thread(self.get_gpu_stats)
        except RuntimeError as exc:
            gpu_error = str(exc)  # no NVML and no nvidia-smi available
        else:
            gpu = {
                "used_mib": stats.used_mib,
                "total_mib": stats.total_mib,
                "utilization_pct": stats.utilization_pct,
                "processes": [
                    {"pid": p.pid, "name": p.name, "used_mib": p.used_mib}
                    for p in stats.processes
                ],
            }
        return {**active, "gpu": gpu, "gpu_error": gpu_error}

    async def logs(
        self, tail: int | None = None, follow: bool = False
    ) -> AsyncIterator[str]:
        """Stream the active managed container's logs (FR-00.8)."""
        managed = await asyncio.to_thread(self.docker.find_managed)
        if not managed:
            raise RuntimeError(
                "no managed container running — load a model first"
                " (`vibe load <alias>`)"
            )
        # The docker logs iterator blocks on reads: create it off the event
        # loop, then pull each line through a worker thread.
        it = await asyncio.to_thread(self.docker.logs, managed[0], tail, follow)
        while True:
            line = await asyncio.to_thread(next, it, _SENTINEL)
            if line is _SENTINEL:
                break
            # The sentinel check above guarantees a log line, not the sentinel.
            yield cast(str, line)
