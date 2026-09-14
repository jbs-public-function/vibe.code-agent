"""vibe CLI — thin gRPC client for the model controller (FR-00.3..FR-00.8, FR-02.10)."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any

import typer
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from vibe.client import ControllerClient, ControllerUnavailable
from vibe.config import Settings
from vibe.errors import EXIT_ALIAS, EXIT_CONFIG, EXIT_OK, EXIT_RUNTIME
from vibe.gen.vibe.v1 import control_pb2
from vibe.jobs import JobEvent, JobInProgress
from vibe.registry import Registry, RegistryError
from vibe.serve import run_serve

app = typer.Typer(name="vibe", help="Local LLM dev environment — model controller CLI")

# Short display names for the ModelState enum (proto values are ints).
_STATE_NAMES: dict[int, str] = {
    control_pb2.AVAILABLE: "AVAILABLE",
    control_pb2.LOADING: "LOADING",
    control_pb2.READY: "READY",
    control_pb2.UNLOADING: "UNLOADING",
    control_pb2.ERROR: "ERROR",
}


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async command body.

    The async logic lives in plain async functions (unit-testable with
    pytest-asyncio); Typer commands are sync wrappers because not every
    Typer version awaits coroutine commands.
    """
    return asyncio.run(coro)


# -- helpers ---------------------------------------------------------------


def _load_settings() -> Settings:
    """Load settings, printing every validation error (client commands need
    only the controller target, not the local registry)."""
    try:
        return Settings()
    except ValidationError as exc:
        print("config error:", file=sys.stderr)
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            print(f"  {loc}: {err['msg']}" if loc else f"  {err['msg']}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG) from exc


def _load_config() -> tuple[Settings, Registry]:
    """Load settings and the registry, printing every validation error (FR-00.4)."""
    settings = _load_settings()
    try:
        registry = Registry.load(Registry.default_path())
    except RegistryError as exc:
        print("config error:", file=sys.stderr)
        for line in exc.errors:
            print(f"  {line}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG) from exc
    return settings, registry


def _print_event(event: JobEvent) -> None:
    """Print one job event as `HH:MM:SS  [STATE-or-level]  message`."""
    ts = event.ts.astimezone().strftime("%H:%M:%S")
    tag = event.state.value if event.state else event.level.upper()
    line = f"{ts}  [{tag}]  {event.message}"
    if event.level == "error":
        print(line, file=sys.stderr)
    else:
        print(line)


def _print_proto_event(event: control_pb2.JobEvent) -> None:
    """Print one gRPC job event like `_print_event` does for `jobs.JobEvent`."""
    ts = datetime.fromtimestamp(event.ts_unix_ms / 1000, UTC).astimezone()
    if event.state:
        tag = control_pb2.ModelState.Name(event.state)
    else:
        tag = event.level.upper()
    line = f"{ts.strftime('%H:%M:%S')}  [{tag}]  {event.message}"
    if event.level == "error":
        print(line, file=sys.stderr)
    else:
        print(line)


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print a left-aligned, column-padded table."""
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows])
        for i in range(len(headers))
    ]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


# -- commands --------------------------------------------------------------


@app.command("list")
def list_cmd() -> None:
    """List registered models and the state of their container (FR-00.3)."""
    _run(_list())


async def _list() -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        models = await client.list_models()
    except ControllerUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc

    headers = ("ALIAS", "DISPLAY NAME", "REPO", "PORT", "STATE")
    rows = [
        (
            model.alias,
            model.display_name,
            model.hf_repo,
            str(model.host_port) if model.host_port else "-",
            _STATE_NAMES.get(model.state, "-"),
        )
        for model in models
    ]
    _table(headers, rows)


@app.command()
def validate() -> None:
    """Validate configuration and the registry; exit 0 on success, 2 on error (FR-00.4)."""
    _settings, registry = _load_config()
    print(f"configuration OK — {len(registry)} model(s) registered in {registry.path}")


@app.command()
def load(
    alias: str,
    force: bool = typer.Option(False, "--force", help="Reload even if already READY."),
    keep_failed: bool = typer.Option(
        False, "--keep-failed", help="Keep a failed container for log inspection."
    ),
    timeout: int | None = typer.Option(
        None, "--timeout", "-t", help="Readiness timeout in seconds."
    ),
) -> None:
    """Load a model: purge old containers, start, and wait for readiness (FR-00.5)."""
    _run(_load(alias, force=force, keep_failed=keep_failed, timeout=timeout))


async def _load(
    alias: str, force: bool, keep_failed: bool, timeout: int | None
) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    final_state: int | None = None
    try:
        async for event in client.load(
            alias, force=force, keep_failed=keep_failed, timeout_s=timeout
        ):
            _print_proto_event(event)
            if event.state:
                final_state = event.state
    except KeyError:
        print(
            f"unknown alias '{alias}' — run `vibe list` to see registered models",
            file=sys.stderr,
        )
        raise typer.Exit(EXIT_ALIAS) from None
    except JobInProgress as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    except ControllerUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    if final_state == control_pb2.ERROR:
        raise typer.Exit(EXIT_RUNTIME)


@app.command()
def unload() -> None:
    """Stop and remove all managed/legacy containers; idempotent (FR-00.6)."""
    _run(_unload())


async def _unload() -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    final_state: int | None = None
    try:
        async for event in client.unload():
            _print_proto_event(event)
            if event.state:
                final_state = event.state
    except JobInProgress as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    except ControllerUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    if final_state == control_pb2.ERROR:
        raise typer.Exit(EXIT_RUNTIME)


@app.command()
def status(
    json_: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Show the active model, container, endpoint, and GPU usage (FR-00.7)."""
    _run(_status(json_=json_))


async def _status(json_: bool) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        status = await client.status()
    except ControllerUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc

    if json_:
        print(
            json.dumps(
                MessageToDict(
                    status, always_print_fields_with_no_presence=True
                ),
                indent=2,
            )
        )
        return

    try:
        gpu = await client.gpu()
    except RuntimeError as exc:  # GPU stats are best-effort (see Phase 0 status)
        gpu = None
        print(f"warning: {exc}", file=sys.stderr)

    state = control_pb2.ModelState.Name(status.state) if status.state else "-"
    started = "-"
    if status.started_at_unix:
        started = datetime.fromtimestamp(
            status.started_at_unix, UTC
        ).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"Active model: {status.active_alias or '-'} ({state})")
    print(f"Container: {status.container_id or '-'}  started: {started}")
    if status.host_port:
        print(f"Endpoint: http://127.0.0.1:{status.host_port}/v1")
    else:
        print("Endpoint: -")
    if gpu is not None:
        print(f"GPU: {gpu.used_mib} / {gpu.total_mib} MiB ({gpu.utilization_pct}%)")
        for process in gpu.processes:
            print(f"  pid={process.pid}  {process.name}  {process.used_mib} MiB")
    if status.last_error:
        print(f"warning: {status.last_error}", file=sys.stderr)


@app.command()
def logs(
    tail: int = typer.Option(50, "--tail", help="Number of trailing log lines."),
    follow: bool = typer.Option(
        False, "--follow", help="Follow the log stream until interrupted."
    ),
) -> None:
    """Stream the active container's logs (FR-00.8)."""
    try:
        _run(_logs(tail=tail, follow=follow))
    except KeyboardInterrupt:
        if follow:
            raise typer.Exit(EXIT_OK) from None
        raise


async def _logs(tail: int, follow: bool) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        async for line in client.logs(tail=tail, follow=follow):
            print(line.line, flush=True)
    except ControllerUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc


@app.command()
def serve(
    only: str | None = typer.Option(
        None,
        "--only",
        help="Run a single component: controller or gateway (admin lands in Phase 3)",
    ),
) -> None:
    """Start the control plane: controller, gateway, and admin (FR-02.9)."""
    raise typer.Exit(run_serve(only))


if __name__ == "__main__":
    app()
