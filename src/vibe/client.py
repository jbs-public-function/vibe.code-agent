"""gRPC client for the vibe model controller (FR-02.10).

Gateway and admin import *only* this module for controller access (02-grpc-server:
"Interfaces"). All `AioRpcError`s are translated into the Python exceptions the
CLI expects:

- `UNAVAILABLE`           → `ControllerUnavailable` (or `UpstreamError` for
  `Chat` when the model server itself answered with an error, IMP-02.25)
- `NOT_FOUND`             → `KeyError` (unknown alias or job id)
- `FAILED_PRECONDITION`   → `jobs.JobInProgress` (`NoModelLoaded` for `Chat`,
  IMP-02.25)
- anything else           → `RuntimeError` with the server's details
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Sequence
from typing import NoReturn, Self

import grpc

from vibe.config import Settings
from vibe.gen.vibe.v1 import control_pb2, control_pb2_grpc
from vibe.jobs import JobInProgress


class ControllerUnavailable(RuntimeError):
    """The controller is not reachable at the configured target (FR-02.10)."""


class NoModelLoaded(RuntimeError):
    """`Chat`: no running container serves the requested alias (IMP-02.25)."""

    def __init__(
        self,
        message: str = "no model loaded — run `vibe load <alias>` first",
    ) -> None:
        super().__init__(message)


class UpstreamError(RuntimeError):
    """`Chat`: the model server answered with an error status (IMP-02.25)."""


class ControllerClient:
    """Typed async wrapper around the `ModelController` gRPC service."""

    def __init__(
        self, settings: Settings | None = None, target: str | None = None
    ) -> None:
        """Create a client; `target` defaults to `settings.controller_target`.

        The channel is created eagerly but `grpc.aio.insecure_channel` does not
        connect until the first RPC, so constructing the client is cheap.
        """
        if settings is None:
            settings = Settings()
        self.target = target if target else settings.controller_target
        self._channel: grpc.aio.Channel = grpc.aio.insecure_channel(self.target)
        self._stub = control_pb2_grpc.ModelControllerStub(self._channel)

    async def close(self) -> None:
        """Close the underlying channel; safe to call more than once."""
        await self._channel.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- error translation -------------------------------------------------

    # -- error translation -------------------------------------------------

    def _raise_translated(
        self,
        exc: grpc.aio.AioRpcError,
        *,
        alias: str | None = None,
        failed_precondition: type[Exception] = JobInProgress,
        unavailable_with: str | None = None,
    ) -> NoReturn:
        """Convert an `AioRpcError` into the Python exception the CLI expects.

        IMP-02.25: translation is per-RPC, not global-by-code. Chat passes
        `failed_precondition=NoModelLoaded` (its FAILED_PRECONDITION means
        "no model loaded") and `unavailable_with="model server returned"`
        (an UNAVAILABLE whose details start with that prefix means the
        vLLM upstream answered with an error, not that the controller is
        down).
        """
        code = exc.code()
        details = exc.details() or ""
        if code is grpc.StatusCode.UNAVAILABLE:
            if unavailable_with and details.startswith(unavailable_with):
                raise UpstreamError(details) from exc
            message = (
                f"controller unreachable at {self.target} — "
                "start it with `vibe serve --only controller`"
            )
            if details:
                message += f" ({details})"
            raise ControllerUnavailable(message) from exc
        if code is grpc.StatusCode.NOT_FOUND:
            raise KeyError(alias if alias is not None else details) from exc
        if code is grpc.StatusCode.FAILED_PRECONDITION:
            if details:
                raise failed_precondition(details) from exc
            raise failed_precondition() from exc
        raise RuntimeError(details or str(exc)) from exc

    async def _unary[T](
        self, call: Awaitable[T], *, alias: str | None = None
    ) -> T:
        """Await a unary RPC, translating `AioRpcError`."""
        try:
            return await call
        except grpc.aio.AioRpcError as exc:
            self._raise_translated(exc, alias=alias)

    async def _iter_stream[T](
        self, stream: AsyncIterable[T], *, alias: str | None = None, **translate
    ) -> AsyncIterator[T]:
        """Iterate a server stream, translating `AioRpcError`.

        Streaming RPCs surface their error status when the stream is first
        iterated (creating the stream never raises, IMP-02.26), so callers
        wrap the iteration. Extra keywords are forwarded to
        `_raise_translated` for per-RPC mappings.
        """
        try:
            async for item in stream:
                yield item
        except grpc.aio.AioRpcError as exc:
            self._raise_translated(exc, alias=alias, **translate)

    # -- RPCs --------------------------------------------------------------

    async def list_models(self) -> list[control_pb2.ModelInfo]:
        """Return every registered model with its current state."""
        response = await self._unary(
            self._stub.ListModels(control_pb2.ListModelsRequest())
        )
        return list(response.models)

    async def status(self) -> control_pb2.Status:
        """Return the active model snapshot (`Status`)."""
        return await self._unary(
            self._stub.GetStatus(control_pb2.GetStatusRequest())
        )

    async def load(
        self,
        alias: str,
        force: bool = False,
        keep_failed: bool = False,
        timeout_s: int | None = None,
    ) -> AsyncIterator[control_pb2.JobEvent]:
        """Load a model, streaming `JobEvent`s until a terminal state.

        Raises `KeyError(alias)` for an unknown alias and `JobInProgress` when
        another job is already running (FR-02.3).
        """
        request = control_pb2.LoadModelRequest(
            alias=alias,
            force=force,
            keep_failed=keep_failed,
            timeout_s=timeout_s or 0,  # 0 = use VIBE_LOAD_TIMEOUT
        )
        # IMP-02.26: creating the stub stream never raises — errors surface
        # on iteration, where `_iter_stream` translates them.
        stream = self._stub.LoadModel(request)
        async for event in self._iter_stream(stream, alias=alias):
            yield event

    async def unload(self) -> AsyncIterator[control_pb2.JobEvent]:
        """Unload every model, streaming `JobEvent`s (idempotent, FR-02.4)."""
        stream = self._stub.UnloadModel(control_pb2.UnloadModelRequest())
        async for event in self._iter_stream(stream):
            yield event

    async def logs(
        self, tail: int | None = None, follow: bool = False
    ) -> AsyncIterator[control_pb2.LogLine]:
        """Stream the active container's logs; fall back to the ring buffer."""
        request = control_pb2.StreamLogsRequest(tail=tail or 0, follow=follow)
        stream = self._stub.StreamLogs(request)
        async for line in self._iter_stream(stream):
            yield line

    async def job_events(
        self, job_id: str, *, follow: bool = False
    ) -> AsyncIterator[control_pb2.JobEvent]:
        """Replay buffered events for a job the caller did not start (Phase 3).

        The controller keeps a bounded per-job history, so this is a snapshot
        of the events recorded so far, oldest first. With `follow` the stream
        tails live events after the replay until the terminal event
        (IMP-01.7). Raises `KeyError` for an unknown job id (NOT_FOUND).
        """
        request = control_pb2.GetJobEventsRequest(job_id=job_id, follow=follow)
        stream = self._stub.GetJobEvents(request)
        async for event in self._iter_stream(stream):
            yield event

    async def get_job(self, job_id: str) -> control_pb2.Job:
        """One job's summary; `KeyError` once it was evicted from history
        (IMP-01.2)."""
        return await self._unary(
            self._stub.GetJob(control_pb2.GetJobRequest(job_id=job_id))
        )

    async def gpu(self) -> control_pb2.GpuStats:
        """Return current GPU memory/utilization/process stats (FR-02.7)."""
        return await self._unary(
            self._stub.GetGpuStats(control_pb2.GetGpuStatsRequest())
        )

    async def chat(
        self,
        alias: str,
        messages: Sequence[control_pb2.ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[control_pb2.ChatChunk]:
        """Stream chat completions from the active model (FR-02.6).

        IMP-02.25: Chat's error translation differs from the other RPCs —
        FAILED_PRECONDITION means "no model loaded" (`NoModelLoaded`) and
        UNAVAILABLE details beginning with "model server returned" mean the
        vLLM upstream answered with an error (`UpstreamError`).
        """
        request = control_pb2.ChatRequest(alias=alias, messages=list(messages))
        if temperature is not None:
            request.temperature = temperature
        if max_tokens is not None:
            request.max_tokens = max_tokens
        stream = self._stub.Chat(request)
        async for chunk in self._iter_stream(
            stream,
            alias=alias,
            failed_precondition=NoModelLoaded,
            unavailable_with="model server returned",
        ):
            yield chunk

    # -- downloads (Phase 4, FR-04.1/FR-04.2) -------------------------------

    async def list_downloads(self) -> list[control_pb2.DownloadInfo]:
        """In-flight and recently completed downloads (last 20)."""
        response = await self._unary(
            self._stub.ListDownloads(control_pb2.ListDownloadsRequest())
        )
        return list(response.downloads)

    async def download(
        self,
        hf_repo: str,
        include: Sequence[str] | None = None,
        revision: str | None = None,
    ) -> AsyncIterator[control_pb2.DownloadProgress]:
        """Download an HF repo, streaming `DownloadProgress` until the
        terminal event (FR-04.1)."""
        request = control_pb2.DownloadModelRequest(
            hf_repo=hf_repo,
            include=list(include or []),
            revision=revision or "",
        )
        stream = self._stub.DownloadModel(request)
        async for progress in self._iter_stream(stream):
            yield progress

    async def cancel_download(self, download_id: str) -> bool:
        """Request cooperative cancellation of one download (FR-04.2)."""
        response = await self._unary(
            self._stub.CancelDownload(
                control_pb2.CancelDownloadRequest(id=download_id)
            )
        )
        return response.cancelled

    # -- HF cache (Phase 4, FR-04.4) -----------------------------------------

    async def list_cache(self) -> control_pb2.ListCacheResponse:
        """Cached repos, sizes, alias refs, and disk space."""
        return await self._unary(
            self._stub.ListCache(control_pb2.ListCacheRequest())
        )

    async def remove_cached(self, repo_id: str) -> tuple[bool, str]:
        """Remove one repo's cache dir; `(removed, error)`; idempotent."""
        response = await self._unary(
            self._stub.RemoveCached(control_pb2.RemoveCachedRequest(repo_id=repo_id))
        )
        return response.removed, response.error

    # -- registry management (Phase 4, FR-04.5/FR-04.6) ----------------------

    async def reload_registry(self) -> list[str]:
        """Hot-reload models.yaml; `[]` on success, errors otherwise."""
        response = await self._unary(
            self._stub.ReloadRegistry(control_pb2.ReloadRegistryRequest())
        )
        return list(response.errors)

    async def propose_registration(
        self, spec: control_pb2.RegistrationSpec
    ) -> control_pb2.ProposeRegistrationResponse:
        """Dry-run a registration: defaults filled, warnings/errors listed."""
        return await self._unary(
            self._stub.ProposeRegistration(
                control_pb2.ProposeRegistrationRequest(spec=spec)
            )
        )

    async def write_registration(
        self, spec: control_pb2.RegistrationSpec
    ) -> control_pb2.WriteRegistrationResponse:
        """Append a validated entry to models.yaml and reload it."""
        return await self._unary(
            self._stub.WriteRegistration(
                control_pb2.WriteRegistrationRequest(spec=spec)
            )
        )
