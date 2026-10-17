"""Swap-on-request coordinator (FR-03.6).

When `VIBE_SWAP_ON_REQUEST=true` the gateway may ask the controller to load a
model mid-request. Exactly one load job may be in flight:

- the first request for an alias starts the job,
- concurrent requests for the *same* alias attach to it and await completion,
- requests for a *different* alias are rejected immediately (`SwapConflict`).

Attach waits are bounded by `timeout_s`; a timed-out caller raises
`SwapTimeout` but the background job keeps running (its completion is not
cancelled by a client's 504). The job is tracked by a per-job `asyncio.Event`
plus a result holder, so waiters capture the event at attach time and always
read the matching result even if a new job starts afterwards.

A `JobInProgress` from the controller means a CLI/admin-initiated job owns the
queue (IMP-03.7): for a load of the *same* alias the gateway attaches by
polling `GetStatus` until READY/ERROR; for anything else it reports a swap
conflict. The in-flight load stream itself is bounded by
`load_timeout_s + _LOAD_STREAM_MARGIN` so a hung stream cannot wedge the
coordinator forever (IMP-03.5).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from vibe import logging as vlog
from vibe.client import ControllerClient
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import state_name
from vibe.jobs import JobInProgress

logger = vlog.get_logger("vibe.gateway")

# Extra headroom on top of VIBE_LOAD_TIMEOUT before a load stream is
# declared hung (IMP-03.5). Module-level so tests can shrink it.
_LOAD_STREAM_MARGIN = 30.0

# Poll interval while attached to an externally started load (IMP-03.7).
_ATTACH_POLL_SECONDS = 1.0


class SwapConflict(RuntimeError):
    """A different alias is already being loaded."""


class SwapFailed(RuntimeError):
    """The load job failed (terminal ERROR, controller error, or post-check)."""


class SwapTimeout(RuntimeError):
    """The attach wait exceeded the configured swap timeout."""


@dataclass
class _Job:
    """Per-load completion state; waiters read it only after the event fires."""

    alias: str
    event: asyncio.Event
    port: int | None = None
    error: SwapFailed | SwapConflict | None = None
    # IMP-03.6: the driving task is retained on the job so it can never be
    # garbage-collected mid-flight and could be cancelled on shutdown.
    task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)


class SwapCoordinator:
    """Coordinates swap-on-request loads with attach/await semantics."""

    def __init__(
        self,
        client: ControllerClient,
        timeout_s: float,
        load_timeout_s: float = 600.0,
    ) -> None:
        self._client = client
        self._timeout_s = timeout_s
        self._load_timeout_s = load_timeout_s
        self._job: _Job | None = None

    @property
    def in_flight_alias(self) -> str | None:
        """The alias currently being loaded, or `None`."""
        return self._job.alias if self._job is not None else None

    async def ensure_ready(self, alias: str) -> int:
        """Load `alias` (or attach to its in-flight load); return its port.

        Raises `SwapConflict` when a different alias is loading, `SwapTimeout`
        when the wait exceeds `timeout_s`, and `SwapFailed` when the load
        itself fails.
        """
        if self._job is not None:
            if self._job.alias != alias:
                raise SwapConflict(
                    f"swap to '{self._job.alias}' is in progress; "
                    f"cannot load '{alias}'"
                )
            job = self._job
        else:
            job = _Job(alias=alias, event=asyncio.Event())
            self._job = job
            # No await between the None check above and this task start, so
            # racing callers can never both create a job (single-threaded
            # event loop).
            job.task = asyncio.create_task(self._run(job))
        await self._await_completion(job)
        if job.error is not None:
            raise job.error
        port = job.port
        if port is None:
            raise SwapFailed(f"swap to '{alias}' completed without a port")
        return port

    async def _await_completion(self, job: _Job) -> None:
        try:
            await asyncio.wait_for(job.event.wait(), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            # Only this caller gives up; the background job keeps running.
            raise SwapTimeout(f"swap timed out after {self._timeout_s}s") from None

    async def _run(self, job: _Job) -> None:
        """Drive one load job to completion and publish its result."""
        logger.info("swap start", extra={"alias": job.alias})
        try:
            terminal_state: int | None = None
            terminal_message = ""
            try:
                # IMP-03.5: a load stream that never terminates must not
                # leave `self._job` set forever — expire it past the load
                # timeout and let the `finally` clear the slot.
                async with asyncio.timeout(self._load_timeout_s + _LOAD_STREAM_MARGIN):
                    async for event in self._client.load(job.alias):
                        if event.terminal:
                            terminal_state = event.state
                            terminal_message = event.message
                            break
            except TimeoutError:
                job.error = SwapFailed(
                    f"swap to '{job.alias}' failed: load stream did not finish "
                    f"within {self._load_timeout_s + _LOAD_STREAM_MARGIN:.0f}s"
                )
                return
            except JobInProgress as exc:
                # The controller's single in-flight guard rejected us: a
                # CLI/admin job owns the queue (IMP-03.7).
                await self._attach_or_conflict(job, exc)
                return
            except Exception as exc:
                # Covers KeyError, ControllerUnavailable and RuntimeError
                # from the controller client, plus any unexpected transport
                # failure.
                job.error = SwapFailed(f"swap to '{job.alias}' failed: {exc}")
                return
            if terminal_state != control_pb2.READY:
                if terminal_state is None:
                    detail = "load stream ended without a terminal event"
                elif terminal_state == control_pb2.ERROR and terminal_message:
                    detail = f"load ended in ERROR: {terminal_message}"
                else:
                    detail = f"load ended in {state_name(terminal_state)}"
                job.error = SwapFailed(f"swap to '{job.alias}' failed: {detail}")
                return
            try:
                status = await self._client.status()
            except Exception as exc:
                job.error = SwapFailed(
                    f"swap to '{job.alias}' failed: status check failed: {exc}"
                )
                return
            if not (
                status.active_alias == job.alias
                and status.state == control_pb2.READY
                and status.host_port
            ):
                job.error = SwapFailed(
                    f"swap to '{job.alias}' failed: controller reports "
                    f"active={status.active_alias or None} in "
                    f"{state_name(status.state)}"
                )
                return
            job.port = int(status.host_port)
            logger.info(
                "swap finished",
                extra={"alias": job.alias, "upstream": f"127.0.0.1:{job.port}"},
            )
        finally:
            if job.error is not None:
                logger.info(
                    "swap failed",
                    extra={"alias": job.alias, "reason": str(job.error)},
                )
            job.event.set()  # publish the result (or failure) to waiters
            if self._job is job:
                self._job = None  # a later request may start a fresh load

    async def _attach_or_conflict(self, job: _Job, exc: Exception) -> None:
        """Handle `JobInProgress` from the controller (IMP-03.7).

        When the in-flight job is a load of *this* alias (started by the CLI
        or the admin UI), attach by polling `GetStatus` until it reaches
        READY or ERROR. Otherwise the gateway cannot start its own load and
        answers with a swap conflict.
        """
        try:
            status = await self._client.status()
        except Exception:
            job.error = SwapFailed(f"swap to '{job.alias}' failed: {exc}")
            return
        current = status.current_job
        if (
            current is not None
            and current.kind == control_pb2.LOAD
            and current.alias == job.alias
        ):
            await self._poll_until_ready(job)
            return
        other = current.alias if current is not None else None
        job.error = SwapConflict(
            f"swap to '{other or 'another model'}' is in progress; "
            f"cannot load '{job.alias}'"
        )

    async def _poll_until_ready(self, job: _Job) -> None:
        """Poll the controller while an externally started load runs."""
        deadline = time.monotonic() + self._load_timeout_s + _LOAD_STREAM_MARGIN
        while time.monotonic() < deadline:
            await asyncio.sleep(_ATTACH_POLL_SECONDS)
            try:
                status = await self._client.status()
            except Exception as exc:
                job.error = SwapFailed(
                    f"swap to '{job.alias}' failed: status check failed: {exc}"
                )
                return
            if (
                status.state == control_pb2.READY
                and status.active_alias == job.alias
                and status.host_port
            ):
                job.port = int(status.host_port)
                logger.info(
                    "swap finished",
                    extra={"alias": job.alias, "upstream": f"127.0.0.1:{job.port}"},
                )
                return
            if status.state == control_pb2.ERROR:
                detail = f": {status.last_error}" if status.last_error else ""
                job.error = SwapFailed(
                    f"swap to '{job.alias}' failed: load ended in ERROR{detail}"
                )
                return
            current = status.current_job
            if current is None or current.terminal:
                detail = (
                    current.last_error
                    if current is not None and current.last_error
                    else "load finished without READY"
                )
                job.error = SwapFailed(f"swap to '{job.alias}' failed: {detail}")
                return
            # Still LOADING/UNLOADING: keep polling.
        job.error = SwapFailed(
            f"swap to '{job.alias}' failed: timed out waiting for the in-flight load"
        )
