"""vibe CLI — thin gRPC client for the model controller (FR-00.3..FR-00.8, FR-02.10, FR-04.3..FR-04.5, FR-04.8)."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Coroutine, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any, NoReturn, cast

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


# -- Phase 4 helpers (FR-04.3..FR-04.5, FR-04.8) ---------------------------


def _exit_unavailable(exc: ControllerUnavailable) -> NoReturn:
    """Standard controller-down exit for the Phase 4 commands (FR-04.3)."""
    print("controller unreachable — start it with `vibe serve`", file=sys.stderr)
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
    """Render one download event: a `\\r`-line on a tty, per-event lines otherwise."""
    done = f"{event.bytes_done}/{event.bytes_total} bytes ({event.progress * 100:.0f}%)"
    status = f"{event.current_file}  {done}" if event.current_file else done
    if is_tty:
        print(f"\r{status}", end="\r", flush=True)
    else:
        print(status)


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
        help="Run a single component: controller, gateway, or admin",
    ),
) -> None:
    """Start the control plane: controller, gateway, and admin (FR-02.9)."""
    raise typer.Exit(run_serve(only))


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
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
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
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
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
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
    if json_:
        print(
            json.dumps(
                [
                    MessageToDict(entry, always_print_fields_with_no_presence=True)
                    for entry in cache.entries
                ],
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
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
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
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
    print(
        f"total: {_fmt_size(cache.disk_total_bytes)} / "
        f"free: {_fmt_size(cache.disk_free_bytes)}"
    )


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
        None, "--port", help="Host port (0/omitted = next free in 8010-8099)."
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
    except ControllerUnavailable as exc:
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
    for warning in proposal.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if proposal.errors:
        for error in proposal.errors:
            print(f"error: {error}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG)
    try:
        write = await client.write_registration(spec)
    except ControllerUnavailable as exc:
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
    if not write.ok:
        for error in write.errors:
            print(f"error: {error}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG)
    try:
        errors = await client.reload_registry()
    except ControllerUnavailable as exc:
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise typer.Exit(EXIT_CONFIG)
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
        _exit_unavailable(exc)
    except Exception as exc:
        _exit_runtime(exc)
    print(zed_settings_json(cast(Sequence[ZedEntry], models)))


if __name__ == "__main__":
    app()
