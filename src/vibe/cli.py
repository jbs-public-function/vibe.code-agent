"""vibe CLI — thin gRPC client for the model controller (FR-00.3..FR-00.8, FR-02.10, FR-04.3..FR-04.5, FR-04.8)."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Coroutine, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, cast

import typer
from google.protobuf import descriptor
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from pydantic import ValidationError

from vibe.client import (
    ControllerClient,
    ControllerUnavailable,
    JevReadOnly,
    JevSchemaNotFound,
)
from vibe.config import ConfigError, Settings, load_config
from vibe.errors import EXIT_ALIAS, EXIT_CONFIG, EXIT_OK, EXIT_RUNTIME
from vibe.gen.vibe.v1 import control_pb2
from vibe.jev import SchemaStore, StoreError, validate_schema_text
from vibe.jobs import JobEvent, JobInProgress
from vibe.registry import REGISTER_PORT_MAX, REGISTER_PORT_MIN, Registry
from vibe.serve import run_serve
from vibe.zedsettings import ZedEntry, zed_settings_json

app = typer.Typer(name="vibe", help="Local LLM dev environment — model controller CLI")

# Short display names for the ModelState enum (proto values are ints).
_STATE_NAMES: dict[int, str] = {
    control_pb2.AVAILABLE: "AVAILABLE",
    control_pb2.LOADING: "LOADING",
    control_pb2.READY: "READY",
    control_pb2.UNLOADING: "UNLOADING",
    control_pb2.ERROR: "ERROR",
}


class Quantization(str, Enum):
    """Allowed `--quantization` values for `vibe register` (FR-04.5).

    Typer 0.27.2 maps Enum options to its own choice type (its vendored click
    copy has no `click.Choice`), giving the same validation and error message.
    """

    awq = "awq"
    gptq = "gptq"
    none = "none"


class ToolParser(str, Enum):
    """Allowed `--tool-parser` values for `vibe register` (FR-04.5)."""

    hermes = "hermes"
    llama3_json = "llama3_json"


class Component(str, Enum):
    """Allowed `--only` values for `vibe serve` (IMP-03.17).

    Typer maps Enum options to its own choice type, so typos become usage
    errors (exit 2) and `--help` lists the choices.
    """

    controller = "controller"
    gateway = "gateway"
    admin = "admin"


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
    try:
        return load_config()
    except ConfigError as exc:
        print("config error:", file=sys.stderr)
        for line in exc.errors:
            print(f"  {line}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG) from exc


_INT64_TYPES = {
    descriptor.FieldDescriptor.TYPE_INT64,
    descriptor.FieldDescriptor.TYPE_UINT64,
    descriptor.FieldDescriptor.TYPE_SINT64,
    descriptor.FieldDescriptor.TYPE_FIXED64,
    descriptor.FieldDescriptor.TYPE_SFIXED64,
}


def _cast_int64s(desc: Any, data: dict[str, Any]) -> None:
    """Recursively cast 64-bit fields from proto3-JSON strings to `int`.

    `desc` is a field descriptor — `Descriptor` in pure-Python protobuf, the
    upb descriptor class under the C++ runtime — so it is typed loosely.
    """
    for field in desc.fields:
        name = field.name
        if field.type in _INT64_TYPES:
            value = data.get(name)
            if value is None:
                continue
            if field.is_repeated:
                data[name] = [int(item) for item in value]
            else:
                data[name] = int(value)
        elif (
            field.type == descriptor.FieldDescriptor.TYPE_MESSAGE
            and field.message_type is not None
        ):
            value = data.get(name)
            if isinstance(value, dict):
                _cast_int64s(field.message_type, value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _cast_int64s(field.message_type, item)


def to_json_dict(msg: Message) -> dict[str, Any]:
    """proto message → JSON dict in the CLI's stable `--json` shape (IMP-03.15).

    `--json` output is a contract (the laboratory reads it): keys are
    snake_case (`preserving_proto_field_name=True`), every field is present
    (`always_print_fields_with_no_presence=True`, so e.g. `active_alias`
    always exists), and 64-bit integers are real JSON numbers rather than
    proto3-JSON strings.
    """
    data = MessageToDict(
        msg,
        always_print_fields_with_no_presence=True,
        preserving_proto_field_name=True,
    )
    _cast_int64s(msg.DESCRIPTOR, data)
    return data


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
    """Print one gRPC job event like `_print_event` does for `jobs.JobEvent`.

    JEV-04.5: error events carry their `error_code` in brackets so a Jev-only
    failure is visible even though the terminal state stays READY; warning
    events render as `[WARN]` via their level.
    """
    ts = datetime.fromtimestamp(event.ts_unix_ms / 1000, UTC).astimezone()
    if event.state:
        tag = control_pb2.ModelState.Name(event.state)
    else:
        tag = event.level.upper()
    line = f"{ts.strftime('%H:%M:%S')}  [{tag}]  {event.message}"
    if event.error_code:
        line += f" [{event.error_code}]"
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


# -- Phase 4 helpers (FR-04.3..FR-04.5, FR-04.8) ---------------------------


def _exit_unavailable(exc: ControllerUnavailable, target: str) -> NoReturn:
    """Standard controller-down exit (FR-04.3, IMP-03.19).

    One message style everywhere: name the target and point at `vibe serve`.
    """
    print(
        f"controller unreachable at {target} — start it with `vibe serve`",
        file=sys.stderr,
    )
    raise typer.Exit(EXIT_RUNTIME) from exc


def _exit_runtime(exc: Exception) -> NoReturn:
    """Generic runtime-failure exit for the Phase 4 commands (FR-04.3)."""
    print(f"error: {exc}", file=sys.stderr)
    raise typer.Exit(EXIT_RUNTIME) from exc


def _fmt_size(size_bytes: int) -> str:
    """Bytes as a compact GiB/MiB label."""
    gib = size_bytes / 1024**3
    if gib >= 1:
        return f"{gib:.1f} GiB"
    mib = size_bytes / 1024**2
    if mib >= 1:
        return f"{mib:.1f} MiB"
    return f"{size_bytes} B"


def _download_state(info: control_pb2.DownloadInfo) -> str:
    """Human-readable state for one download row (FR-04.3)."""
    if info.done:
        if info.ok:
            return "DONE"
        return f"ERROR: {info.error}" if info.error else "ERROR"
    return "DOWNLOADING"


def _print_download_event(event: control_pb2.DownloadProgress, *, is_tty: bool) -> None:
    """Render one download event: a `\r`-line on a tty, per-event lines otherwise."""
    done = f"{event.bytes_done}/{event.bytes_total} bytes ({event.progress * 100:.0f}%)"
    status = f"{event.current_file}  {done}" if event.current_file else done
    if is_tty:
        print(f"\r{status}", end="\r", flush=True)
    else:
        print(status)


# -- Jev helpers (improvements-2 doc 04, J3) --------------------------------


def _jev_exit_code(final_state: int | None, error_code: str) -> int:
    """Exit code for a finished load/attach stream (README taxonomy, JEV-04.4).

    Keyed off the terminal event's `error_code`, never the state:
    `jev_schema_not_found` ⇒ 3, `jev_schema_invalid` ⇒ 2, any other code
    (`jev_probe_failed`, `jev_unsupported_model`, …) ⇒ 1. Without a code an
    ERROR terminal state keeps its historical exit 1; `jev_not_active` is an
    info event and exits 0.
    """
    if error_code == "jev_schema_not_found":
        return EXIT_ALIAS
    if error_code == "jev_schema_invalid":
        return EXIT_CONFIG
    if error_code:
        return EXIT_RUNTIME
    if final_state == control_pb2.ERROR:
        return EXIT_RUNTIME
    return EXIT_OK


def _check_schema_name(name: str) -> None:
    """Client-side store-name pre-check so bad names fail before any RPC."""
    try:
        SchemaStore.check_name(name)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG) from exc


def _stdin_schema_text() -> str:
    """Read schema text from stdin, refusing an interactive TTY (JEV-04.2)."""
    isatty = getattr(sys.stdin, "isatty", None)
    if isatty is not None and isatty():
        print(
            "error: refusing to read the schema from a TTY — pipe the JSON "
            "or pass a file path",
            file=sys.stderr,
        )
        raise typer.Exit(EXIT_CONFIG)
    return sys.stdin.read()


def _read_schema_text(path: str) -> str:
    """Read and locally validate one schema file (or stdin) — JEV-04.2.

    Every problem (unreadable file, non-UTF-8, > 64 KiB, invalid schema)
    exits 2 *before* any RPC, printing errors and warnings so the common
    mistake fails in milliseconds without a controller.
    """
    if path == "-":
        text = _stdin_schema_text()
    else:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            print(
                f"error: {path} is not valid UTF-8 (byte offset {exc.start})",
                file=sys.stderr,
            )
            raise typer.Exit(EXIT_CONFIG) from exc
        except OSError as exc:
            print(
                f"error: cannot read {path}: {exc.strerror or exc}",
                file=sys.stderr,
            )
            raise typer.Exit(EXIT_CONFIG) from exc
    result = validate_schema_text(text)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    if not result.ok:
        raise typer.Exit(EXIT_CONFIG)
    return text


def _resolve_load_jev(
    *,
    jev: bool,
    no_jev: bool,
    include_json: str | None,
    schema: str | None,
    save_as: str | None,
) -> tuple[int, control_pb2.JevSpec | None]:
    """Check the `load` Jev option conflicts (JEV-04.1) and build the args.

    `--include-json`/`--schema` imply `--jev`; `--no-jev` conflicts with all
    of them; `--save-as` is only valid with `--include-json`. Every usage
    error exits 2 before any RPC.
    """
    if jev and no_jev:
        print("error: --jev and --no-jev are mutually exclusive", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG)
    if no_jev and (include_json is not None or schema is not None or save_as is not None):
        print(
            "error: --no-jev cannot be combined with --include-json, --schema "
            "or --save-as",
            file=sys.stderr,
        )
        raise typer.Exit(EXIT_CONFIG)
    if include_json is not None and schema is not None:
        print(
            "error: --include-json and --schema are mutually exclusive",
            file=sys.stderr,
        )
        raise typer.Exit(EXIT_CONFIG)
    if save_as is not None and include_json is None:
        print("error: --save-as requires --include-json", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG)
    if no_jev:
        return control_pb2.JEV_OFF, None
    if not (jev or include_json is not None or schema is not None):
        return control_pb2.JEV_MODE_UNSPECIFIED, None
    if schema is not None:
        _check_schema_name(schema)
    if save_as is not None:
        _check_schema_name(save_as)
    spec = control_pb2.JevSpec()
    if include_json is not None:
        spec.schema_json = _read_schema_text(include_json)  # exits 2, no RPC
    if schema is not None:
        spec.schema_name = schema
    if save_as is not None:
        spec.save_as = save_as
    return control_pb2.JEV_ON, spec


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
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    finally:
        await client.close()

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
    jev: bool = typer.Option(
        False,
        "--jev",
        help="Attach Jev; the schema is picked by precedence "
        "(model default, else 'default').",
        rich_help_panel="Jev",
    ),
    no_jev: bool = typer.Option(
        False,
        "--no-jev",
        help="Detach Jev / load plainly (same-alias READY fast path).",
        rich_help_panel="Jev",
    ),
    include_json: str | None = typer.Option(
        None,
        "--include-json",
        help="Attach Jev with this schema file ('-' = stdin).",
        rich_help_panel="Jev",
    ),
    schema: str | None = typer.Option(
        None,
        "--schema",
        help="Attach Jev with a stored/built-in schema by name.",
        rich_help_panel="Jev",
    ),
    save_as: str | None = typer.Option(
        None,
        "--save-as",
        help="Persist the --include-json schema under this name after a "
        "successful probe.",
        rich_help_panel="Jev",
    ),
) -> None:
    """Load a model: purge old containers, start, and wait for readiness (FR-00.5)."""
    jev_mode, jev_spec = _resolve_load_jev(
        jev=jev, no_jev=no_jev, include_json=include_json, schema=schema, save_as=save_as
    )
    _run(
        _load(
            alias,
            force=force,
            keep_failed=keep_failed,
            timeout=timeout,
            jev_mode=jev_mode,
            jev=jev_spec,
        )
    )


async def _load(
    alias: str,
    force: bool,
    keep_failed: bool,
    timeout: int | None,
    jev_mode: int = control_pb2.JEV_MODE_UNSPECIFIED,
    jev: control_pb2.JevSpec | None = None,
) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    final_state: int | None = None
    final_error_code: str = ""
    saw_container_start = False
    saw_attach = False
    try:
        async for event in client.load(
            alias,
            force=force,
            keep_failed=keep_failed,
            timeout_s=timeout,
            jev_mode=jev_mode,
            jev=jev,
        ):
            _print_proto_event(event)
            if event.state:
                final_state = event.state
            if event.state in (control_pb2.LOADING, control_pb2.UNLOADING):
                saw_container_start = True
            if event.error_code:
                final_error_code = event.error_code
            if event.message.startswith("jev: attached"):
                saw_attach = True
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
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    finally:
        await client.close()
    exit_code = _jev_exit_code(final_state, final_error_code)
    if exit_code != EXIT_OK:
        raise typer.Exit(exit_code)
    # JEV-04.3: an attach that never saw a container start took the fast path.
    if jev_mode == control_pb2.JEV_ON and saw_attach and not saw_container_start:
        print("attached without restart")


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
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    finally:
        await client.close()
    if final_state == control_pb2.ERROR:
        raise typer.Exit(EXIT_RUNTIME)


@app.command()
def status(
    json_: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Show the active model, container, endpoint, and GPU usage (FR-00.7)."""
    _run(_status(json_=json_))


@app.command("jobs")
def jobs_cmd(
    job_id: str = typer.Argument(..., help="Job id from a load/unload stream."),
) -> None:
    """Show one job's kind, alias, and terminal state (IMP-01.2)."""
    _run(_jobs(job_id))


async def _jobs(job_id: str) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        job = await client.get_job(job_id)
    except KeyError:
        print(
            f"unknown job '{job_id}' — the controller keeps only recent jobs",
            file=sys.stderr,
        )
        raise typer.Exit(EXIT_RUNTIME) from None
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    finally:
        await client.close()
    kind = control_pb2.JobKind.Name(job.kind)
    print(
        f"job {job.id}: kind={kind} alias={job.alias or '-'} "
        f"terminal={job.terminal} last_error={job.last_error or '-'}"
    )


async def _status(json_: bool) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    gpu: control_pb2.GpuStats | None = None
    try:
        status = await client.status()
        if not json_:
            try:
                gpu = await client.gpu()
            except RuntimeError as exc:  # GPU stats are best-effort (Phase 0)
                print(f"warning: {exc}", file=sys.stderr)
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    finally:
        await client.close()

    if json_:
        data = to_json_dict(status)
        # JEV-04.6: `MessageToDict` omits unset message fields, so surface
        # the `jev` key explicitly — `null` when inactive, the populated
        # JevInfo dict otherwise (int64s cast to real ints by to_json_dict).
        data["jev"] = to_json_dict(status.jev) if status.HasField("jev") else None
        print(json.dumps(data, indent=2))
        return

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
    if status.HasField("jev"):
        info = status.jev
        stale = " (stale — re-attach to refresh)" if info.stale else ""
        print(f"Jev: {info.schema_name} ({info.source}, sha {info.sha256[:12]}){stale}")
    else:
        print("Jev: off")
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
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME) from exc
    finally:
        await client.close()


@app.command()
def serve(
    only: Component | None = typer.Option(
        None,
        "--only",
        help="Run a single component: controller, gateway, or admin",
    ),
) -> None:
    """Start the control plane: controller, gateway, and admin (FR-02.9)."""
    raise typer.Exit(run_serve(only.value if only is not None else None))


@app.command()
def download(
    hf_repo: str,
    include: list[str] | None = typer.Option(
        None,
        "--include",
        help="Glob pattern to download (repeatable); omit for the whole repo.",
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="Repo revision (branch, tag or commit) to pin."
    ),
) -> None:
    """Download an HF repo with live progress; exit 0/1 (FR-04.3)."""
    _run(_download(hf_repo, include=include, revision=revision))


async def _download(
    hf_repo: str, include: list[str] | None, revision: str | None
) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    is_tty = sys.stderr.isatty()
    terminal_seen = False
    try:
        # IMP-01.4: when the RPC attaches to an in-flight download, show its
        # current file from `DownloadInfo` so the user sees where it stands.
        for info in await client.list_downloads():
            if info.done or info.hf_repo.lower() != hf_repo.strip().lower():
                continue
            current = f" ({info.current_file})" if info.current_file else ""
            print(f"attaching to in-flight download{current}")
            break
        async for event in client.download(
            hf_repo, include=include, revision=revision
        ):
            _print_download_event(event, is_tty=is_tty)
            if not event.terminal:
                continue
            terminal_seen = True
            if event.error:
                if is_tty:
                    print()  # finish the progress line before the error
                print(f"download failed: {event.error}", file=sys.stderr)
                raise typer.Exit(EXIT_RUNTIME)
            if is_tty:
                print()  # finish the progress line
            print(f"downloaded {hf_repo}")
            return
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except typer.Exit:
        # IMP-03.16: `typer.Exit` subclasses `RuntimeError`; re-raise it so
        # the generic handler below does not print a spurious `error:` line.
        raise
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if not terminal_seen:
        print("error: download stream ended without a terminal event", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME)


@app.command()
def downloads() -> None:
    """List in-flight and recent downloads (FR-04.3)."""
    _run(_downloads())


async def _downloads() -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        infos = await client.list_downloads()
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if not infos:
        print("no downloads")
        return
    headers = ("ID", "REPO", "PROGRESS", "STATE")
    rows = [
        (info.id, info.hf_repo, f"{info.progress * 100:.0f}%", _download_state(info))
        for info in infos
    ]
    _table(headers, rows)


cache_app = typer.Typer(help="Inspect and manage the HF model cache (FR-04.4).")
app.add_typer(cache_app, name="cache")


@cache_app.command("list")
def cache_list(
    json_: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List cached HF repos with size, last-modified and alias links (FR-04.4)."""
    _run(_cache_list(json_=json_))


async def _cache_list(json_: bool) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        cache = await client.list_cache()
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if json_:
        print(
            json.dumps(
                [to_json_dict(entry) for entry in cache.entries],
                indent=2,
            )
        )
        return
    if not cache.entries:
        print("cache is empty")
        return
    headers = ("REPO", "SIZE", "MODIFIED", "ALIASES")
    rows = [
        (
            entry.repo_id,
            _fmt_size(entry.size_bytes),
            entry.last_modified or "-",
            ", ".join(entry.aliases) or "-",
        )
        for entry in cache.entries
    ]
    _table(headers, rows)


@cache_app.command("rm")
def cache_rm(repo: str) -> None:
    """Remove a cached HF repo after confirmation (FR-04.4)."""
    _run(_cache_rm(repo))


async def _cache_rm(repo: str) -> None:
    if not typer.confirm(f"remove {repo} from the HF cache?"):
        raise typer.Exit(EXIT_OK)
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        removed, error = await client.remove_cached(repo)
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if not removed:
        print(f"not removed: {error or repo}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME)
    print(f"removed {repo}")


@cache_app.command("du")
def cache_du() -> None:
    """Print total and free space on the HF_HOME filesystem (FR-04.4)."""
    _run(_cache_du())


async def _cache_du() -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        cache = await client.list_cache()
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    print(
        f"total: {_fmt_size(cache.disk_total_bytes)} / "
        f"free: {_fmt_size(cache.disk_free_bytes)}"
    )


jev_app = typer.Typer(
    help="Manage Jev: schema-constrained model output (improvements-2 doc 04)."
)
app.add_typer(jev_app, name="jev")


@jev_app.command("status")
def jev_status(
    json_: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Show the active Jev schema: name, source, sha, stale, applied-at."""
    _run(_jev_status(json_=json_))


async def _jev_status(json_: bool) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        active = await client.get_active_jev()
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if json_:
        payload: dict[str, Any] = {"active": False, "info": None}
        if active.HasField("info"):
            payload = {"active": True, "info": to_json_dict(active.info)}
        print(json.dumps(payload, indent=2))
        return
    if not active.HasField("info"):
        print("Jev: off")
        return
    info = active.info
    applied = "-"
    if info.applied_at_unix_ms:
        applied = (
            datetime.fromtimestamp(info.applied_at_unix_ms / 1000, UTC)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
    print(f"Jev: on — schema {info.schema_name!r} ({info.source})")
    print(f"  sha256: {info.sha256 or '-'}")
    print(f"  applied: {applied}")
    if info.title:
        print(f"  title: {info.title}")
    if info.stale:
        print("  stale: the schema was edited after activation — re-attach to refresh")


@jev_app.command("off")
def jev_off() -> None:
    """Detach the active Jev schema; idempotent, never touches the container."""
    _run(_jev_off())


async def _jev_off() -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        try:
            status = await client.status()
        except ControllerUnavailable as exc:
            _exit_unavailable(exc, client.target)
        except Exception as exc:
            _exit_runtime(exc)
        alias = status.active_alias
        if not alias or status.state != control_pb2.READY:
            # Jev state only exists while its container is READY (purge and
            # unload clear it), so there is nothing to detach — JEV-02.12(d).
            print("jev: not active — nothing to detach")
            return
        final_state: int | None = None
        final_error_code = ""
        try:
            async for event in client.load(alias, jev_mode=control_pb2.JEV_OFF):
                _print_proto_event(event)
                if event.state:
                    final_state = event.state
                if event.error_code:
                    final_error_code = event.error_code
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
            _exit_unavailable(exc, client.target)
        except Exception as exc:
            _exit_runtime(exc)
    finally:
        await client.close()
    exit_code = _jev_exit_code(final_state, final_error_code)
    if exit_code != EXIT_OK:
        raise typer.Exit(exit_code)


schema_app = typer.Typer(
    help="List, inspect, edit and validate stored Jev schemas."
)
jev_app.add_typer(schema_app, name="schema")


@schema_app.command("list")
def schema_list(
    json_: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List every visible schema (built-in, user, override) with status."""
    _run(_schema_list(json_=json_))


async def _schema_list(json_: bool) -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        entries = await client.list_jev_schemas()
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if json_:
        print(json.dumps([to_json_dict(entry) for entry in entries], indent=2))
        return
    if not entries:
        print("no Jev schemas")
        return
    headers = ("NAME", "SOURCE", "SHA12", "SIZE", "ACTIVE")
    rows = [
        (
            entry.name,
            entry.source,
            entry.sha256[:12],
            _fmt_size(entry.size_bytes),
            "*" if entry.active else "",
        )
        for entry in entries
    ]
    _table(headers, rows)


@schema_app.command("show")
def schema_show(
    name: str,
    raw: bool = typer.Option(
        False, "--raw", help="Print the exact stored bytes (no pretty-printing)."
    ),
) -> None:
    """Pretty-print one stored schema (or its exact bytes with --raw)."""
    _run(_schema_show(name, raw=raw))


async def _schema_show(name: str, raw: bool) -> None:
    _check_schema_name(name)
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        schema = await client.get_jev_schema(name)
    except JevSchemaNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_ALIAS) from exc
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if raw:
        sys.stdout.write(schema.schema_json)
        if not schema.schema_json.endswith("\n"):
            sys.stdout.write("\n")
        return
    try:
        pretty = json.dumps(json.loads(schema.schema_json), indent=2)
    except json.JSONDecodeError as exc:
        print(f"error: stored schema is not valid JSON: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG) from exc
    print(pretty)


@schema_app.command("validate")
def schema_validate(
    file: str | None = typer.Argument(None, help="Schema file to validate ('-' = stdin)."),
    name: str | None = typer.Option(
        None, "--name", help="Validate a stored schema by name (needs the controller)."
    ),
) -> None:
    """Validate a schema file offline, or a stored schema via the controller."""
    _run(_schema_validate(file=file, name=name))


async def _schema_validate(file: str | None, name: str | None) -> None:
    if (file is None) == (name is None):
        print(
            "error: pass a schema FILE ('-' for stdin) or --name NAME",
            file=sys.stderr,
        )
        raise typer.Exit(EXIT_CONFIG)
    if file is not None:
        # JEV-04.8: fully offline — the controller may be down.
        _read_schema_text(file)  # prints warnings; exits 2 on any error
        print("schema is valid")
        return
    assert name is not None
    _check_schema_name(name)
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        schema = await client.get_jev_schema(name)
        validation = await client.validate_jev_schema(schema.schema_json)
    except JevSchemaNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_ALIAS) from exc
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    for warning in validation.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in validation.errors:
        print(f"error: {error}", file=sys.stderr)
    if not validation.ok:
        raise typer.Exit(EXIT_CONFIG)
    print(f"schema '{name}' is valid (sha {validation.sha256[:12]})")


@schema_app.command("put")
def schema_put(name: str, file: str) -> None:
    """Create or update a user schema (overriding a built-in is allowed)."""
    _run(_schema_put(name, file))


async def _schema_put(name: str, file: str) -> None:
    _check_schema_name(name)
    text = _read_schema_text(file)  # local validation first, no RPC on failure
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        response = await client.put_jev_schema(name, text)
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if response.error:
        print(f"error: {response.error}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG)
    for warning in response.validation.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in response.validation.errors:
        print(f"error: {error}", file=sys.stderr)
    if not response.validation.ok:
        raise typer.Exit(EXIT_CONFIG)
    print(
        f"stored schema '{name}' (sha {response.validation.sha256[:12]}, "
        f"{_fmt_size(response.validation.size_bytes)})"
    )


@schema_app.command("rm")
def schema_rm(name: str) -> None:
    """Delete a user schema (or remove an override of a built-in)."""
    _run(_schema_rm(name))


async def _schema_rm(name: str) -> None:
    _check_schema_name(name)
    settings = _load_settings()
    client = ControllerClient(settings)
    was_active = False
    try:
        try:
            active = await client.get_active_jev()
            was_active = active.HasField("info") and active.info.schema_name == name
        except ControllerUnavailable as exc:
            _exit_unavailable(exc, client.target)
        except Exception as exc:
            _exit_runtime(exc)
        try:
            response = await client.delete_jev_schema(name)
        except JevReadOnly as exc:
            # JEV-04.9: built-in-only names are read-only — exit 1.
            print(f"error: {exc}", file=sys.stderr)
            raise typer.Exit(EXIT_RUNTIME) from exc
        except JevSchemaNotFound as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise typer.Exit(EXIT_ALIAS) from exc
        except ControllerUnavailable as exc:
            _exit_unavailable(exc, client.target)
        except Exception as exc:
            _exit_runtime(exc)
    finally:
        await client.close()
    if not response.ok:
        print(f"error: {response.error or 'delete failed'}", file=sys.stderr)
        raise typer.Exit(EXIT_RUNTIME)
    if response.reverted_to:
        print(f"deleted override of '{name}' — reverted to built-in")
    else:
        print(f"deleted schema '{name}'")
    if was_active:
        # JEV-04.9: deleting the active schema is allowed; the running model
        # keeps its own copy of the body.
        print(
            f"note: '{name}' is currently active — the running model keeps "
            "its copy until a new schema is attached"
        )


@schema_app.command("export")
def schema_export(
    name: str,
    output: str | None = typer.Option(
        None, "-o", "--output", help="Write to FILE instead of stdout."
    ),
) -> None:
    """Write a stored schema's JSON to stdout or a file."""
    _run(_schema_export(name, output))


async def _schema_export(name: str, output: str | None) -> None:
    _check_schema_name(name)
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        schema = await client.get_jev_schema(name)
    except JevSchemaNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(EXIT_ALIAS) from exc
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    if output is not None:
        try:
            Path(output).write_text(schema.schema_json, encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {output}: {exc}", file=sys.stderr)
            raise typer.Exit(EXIT_RUNTIME) from exc
        print(f"exported '{name}' to {output}")
        return
    sys.stdout.write(schema.schema_json)
    if not schema.schema_json.endswith("\n"):
        sys.stdout.write("\n")


@app.command()
def register(
    hf_repo: str,
    alias: str = typer.Option(..., "--alias", help="Registry alias for the new entry."),
    quantization: Quantization | None = typer.Option(
        None, "--quantization", help="Weights quantization format."
    ),
    max_model_len: int | None = typer.Option(
        None, "--max-model-len", help="Context length cap (0/omitted = auto)."
    ),
    gpu_mem: float | None = typer.Option(
        None, "--gpu-mem", help="GPU memory fraction in (0, 1] (0/omitted = auto)."
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help=f"Host port (0/omitted = next free in {REGISTER_PORT_MIN}-{REGISTER_PORT_MAX}).",
    ),
    tool_parser: ToolParser | None = typer.Option(
        None, "--tool-parser", help="Tool-call parser for automatic tool choice."
    ),
    use_case: str | None = typer.Option(None, "--use-case", help="One-line description."),
    tags: list[str] | None = typer.Option(None, "--tags", help="Free-form tag (repeatable)."),
) -> None:
    """Register a downloaded repo in models.yaml after validation (FR-04.5)."""
    _run(
        _register(
            hf_repo,
            alias=alias,
            quantization=quantization,
            max_model_len=max_model_len,
            gpu_mem=gpu_mem,
            port=port,
            tool_parser=tool_parser,
            use_case=use_case,
            tags=tags,
        )
    )


async def _register(
    hf_repo: str,
    alias: str,
    quantization: Quantization | None,
    max_model_len: int | None,
    gpu_mem: float | None,
    port: int | None,
    tool_parser: ToolParser | None,
    use_case: str | None,
    tags: list[str] | None,
) -> None:
    settings = _load_settings()
    spec = control_pb2.RegistrationSpec(
        hf_repo=hf_repo,
        alias=alias,
        quantization=quantization.value if quantization else "",
        max_model_len=max_model_len or 0,
        gpu_memory_utilization=gpu_mem or 0.0,
        host_port=port or 0,
        tool_parser=tool_parser.value if tool_parser else "",
        use_case=use_case or "",
        tags=tags or [],
    )
    client = ControllerClient(settings)
    try:
        proposal = await client.propose_registration(spec)
        for warning in proposal.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if proposal.errors:
            for error in proposal.errors:
                print(f"error: {error}", file=sys.stderr)
            raise typer.Exit(EXIT_CONFIG)
        write = await client.write_registration(spec)
        if not write.ok:
            for error in write.errors:
                print(f"error: {error}", file=sys.stderr)
            raise typer.Exit(EXIT_CONFIG)
        errors = await client.reload_registry()
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            raise typer.Exit(EXIT_CONFIG)
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except typer.Exit:
        # IMP-03.16: `typer.Exit` subclasses `RuntimeError`; re-raise it so
        # the generic handler below does not print a spurious `error:` line.
        raise
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    model = proposal.model
    print(f"registered {model.alias} — http://127.0.0.1:{model.host_port}/v1")
    print("models.yaml updated")


@app.command("zed-settings")
def zed_settings() -> None:
    """Print the Zed openai_compatible block for the live registry (FR-04.8)."""
    _run(_zed_settings())


async def _zed_settings() -> None:
    settings = _load_settings()
    client = ControllerClient(settings)
    try:
        models = await client.list_models()
    except ControllerUnavailable as exc:
        _exit_unavailable(exc, client.target)
    except Exception as exc:
        _exit_runtime(exc)
    finally:
        await client.close()
    print(zed_settings_json(cast(Sequence[ZedEntry], models)))


if __name__ == "__main__":
    app()
