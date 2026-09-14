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
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from vibe.client import ControllerClient
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import state_name


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
    error: SwapFailed | None = None


class SwapCoordinator:
    """Coordinates swap-on-request loads with attach/await semantics."""

    def __init__(self, client: ControllerClient, timeout_s: float) -> None:
        self._client = client
        self._timeout_s = timeout_s
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
            asyncio.create_task(self._run(job))
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
        try:
            terminal_state: int | None = None
            terminal_message = ""
            try:
                async for event in self._client.load(job.alias):
                    if event.terminal:
                        terminal_state = event.state
                        terminal_message = event.message
                        break
            except Exception as exc:
                # Covers KeyError, jobs.JobInProgress (a RuntimeError
                # subclass), ControllerUnavailable and RuntimeError from the
                # controller client, plus any unexpected transport failure.
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
        finally:
            job.event.set()  # publish the result (or failure) to waiters
            if self._job is job:
                self._job = None  # a later request may start a fresh load
