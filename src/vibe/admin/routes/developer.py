"""Developer tools: downloads, HF cache and guided registration (FR-04.7, NFR-04.5).

`GET /developer` shows the Phase 4 sections: the new-download form plus a
table of in-flight/recent downloads, the HF cache browser with a disk-usage
line, the read-only Zed settings block generated from the registry, and the
entry point to the guided registration flow. `POST /developer/downloads`
starts a download stream in a background task and 303-redirects to its
progress page, which auto-refreshes every 2 s until the download finishes.
`GET /developer/register` pre-fills the registration form from a cached repo;
`POST /developer/register` proposes the entry and shows the YAML preview;
`POST /developer/register/confirm` writes it to `models.yaml` and reloads the
registry. Zero JavaScript (NFR-04.5/NFR-01.1); every controller call goes
through the shared `grpc()` deadline helper (NFR-01.5).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import cast

import yaml
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from vibe.admin.routes import (
    GRPC_TIMEOUT,
    client_of,
    grpc,
    parse_form,
    redirect,
    redirect_with_flash,
    render,
)
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2
from vibe.zedsettings import ZedEntry, zed_settings_json

router = APIRouter()

# Defaults for the register form; the controller fills 0/"" as "auto" (FR-04.5).
_QUANTIZATION_OPTIONS = ("auto", "awq", "gptq", "none")
_TOOL_PARSER_OPTIONS = ("auto", "hermes", "llama3_json")


# -- formatting helpers (FR-04.7b/c/d) --------------------------------------


def _fmt_size(size_bytes: int) -> str:
    """Bytes as a compact GiB/MiB label."""
    gib = size_bytes / 1024**3
    if gib >= 1:
        return f"{gib:.1f} GiB"
    mib = size_bytes / 1024**2
    if mib >= 1:
        return f"{mib:.1f} MiB"
    return f"{size_bytes} B"


def _suggest_alias(repo: str) -> str:
    """A default alias from the repo's last path segment (FR-04.7d)."""
    return repo.rstrip("/").split("/")[-1].lower()


def _register_form(form: dict[str, str], repo: str = "") -> dict[str, str]:
    """Pre-filled register form values; missing keys fall back to defaults."""
    repo = repo or form.get("repo", "").strip()
    return {
        "repo": repo,
        "alias": form.get("alias", "").strip() or _suggest_alias(repo),
        "quantization": form.get("quantization", "").strip() or "auto",
        "max_model_len": form.get("max_model_len", "").strip(),
        "gpu_mem": form.get("gpu_mem", "").strip(),
        "port": form.get("port", "").strip(),
        "tool_parser": form.get("tool_parser", "").strip() or "auto",
        "use_case": form.get("use_case", "").strip(),
        "tags": form.get("tags", "").strip(),
    }


def _spec_from_form(form: dict[str, str]) -> control_pb2.RegistrationSpec:
    """Build a `RegistrationSpec`; blank/auto values stay 0/"" (auto upstream)."""

    def _int(key: str) -> int:
        try:
            return int(form.get(key, "").strip())
        except ValueError:
            return 0

    def _float(key: str) -> float:
        try:
            return float(form.get(key, "").strip())
        except ValueError:
            return 0.0

    quantization = form.get("quantization", "").strip()
    if quantization == "auto":
        quantization = ""
    tool_parser = form.get("tool_parser", "").strip()
    if tool_parser == "auto":
        tool_parser = ""
    tags = [tag.strip() for tag in form.get("tags", "").split(",") if tag.strip()]
    return control_pb2.RegistrationSpec(
        hf_repo=form.get("repo", "").strip(),
        alias=form.get("alias", "").strip(),
        quantization=quantization,
        max_model_len=_int("max_model_len"),
        gpu_memory_utilization=_float("gpu_mem"),
        host_port=_int("port"),
        tool_parser=tool_parser,
        use_case=form.get("use_case", "").strip(),
        tags=tags,
    )


def _preview_yaml(
    entry: control_pb2.RegistrationSpec | control_pb2.ModelInfo, tool_parser: str
) -> str:
    """An illustrative `models.yaml` snippet for the register preview (FR-04.7d)."""
    extra_args = (
        ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
        if tool_parser
        else []
    )
    data = {
        "models": [
            {
                "alias": entry.alias,
                "display_name": entry.display_name or entry.alias,
                "hf_repo": entry.hf_repo,
                "quantization": entry.quantization or "none",
                "max_model_len": entry.max_model_len,
                "gpu_memory_utilization": entry.gpu_memory_utilization,
                "container_name": f"vibe-{entry.alias}",
                "host_port": entry.host_port,
                "extra_args": extra_args,
                "use_case": entry.use_case,
                "tags": list(entry.tags),
            }
        ]
    }
    return yaml.safe_dump(data, sort_keys=False)


# -- downloads (FR-04.7a) ---------------------------------------------------


async def _start_download(
    request: Request,
    repo: str,
    include: list[str],
    revision: str | None,
) -> str:
    """Start a download stream and hand it to a background task (FR-04.7a).

    Like `start_background_job`, but for `DownloadProgress` streams: the first
    event carries the download id and the rest of the stream is drained by a
    task held in `app.state.background_tasks`. The latest event and the start
    time are recorded on the app state so the progress page can show the
    current file and elapsed time.
    """
    client = client_of(request)
    agen = client.download(repo, include=include, revision=revision)
    first = await asyncio.wait_for(anext(agen), timeout=GRPC_TIMEOUT)
    download_id = first.id
    if not hasattr(request.app.state, "download_live"):
        request.app.state.download_live = {}
    if not hasattr(request.app.state, "download_started"):
        request.app.state.download_started = {}
    live = cast(dict[str, control_pb2.DownloadProgress], request.app.state.download_live)
    started = cast(dict[str, float], request.app.state.download_started)
    live[download_id] = first
    started[download_id] = time.time()

    async def drain() -> None:
        try:
            async for event in agen:
                live[download_id] = event
        except Exception:
            pass  # a background download must never raise into the loop

    tasks = cast(set[asyncio.Task[None]], request.app.state.background_tasks)
    task = asyncio.create_task(drain())
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return download_id


@router.get("/developer", name="developer")
async def developer_page(request: Request) -> Response:
    """Downloads, HF cache, disk usage, register entry and Zed block (FR-04.7)."""
    client = client_of(request)
    status = await grpc(client.status())
    downloads = await grpc(client.list_downloads())
    cache = await grpc(client.list_cache())
    models = await grpc(client.list_models())
    controller_down = any(value is None for value in (status, downloads, cache))
    zed_json: str | None = None
    if models is not None:
        zed_json = zed_settings_json(cast(Sequence[ZedEntry], models))
    cache_rows = (
        [
            {
                "repo_id": entry.repo_id,
                "size": _fmt_size(entry.size_bytes),
                "last_modified": entry.last_modified or "—",
                "aliases": list(entry.aliases),
            }
            for entry in cache.entries
        ]
        if cache is not None
        else []
    )
    return render(
        request,
        "developer.html",
        {
            "page": "developer",
            "controller_down": controller_down,
            "active_alias": status.active_alias if status is not None else "",
            "downloads": downloads if downloads is not None else [],
            "cache_entries": cache_rows,
            "disk_total": _fmt_size(cache.disk_total_bytes) if cache is not None else "—",
            "disk_free": _fmt_size(cache.disk_free_bytes) if cache is not None else "—",
            "zed_json": zed_json,
        },
    )


@router.post("/developer/downloads", name="developer_downloads_post")
async def developer_downloads_post(request: Request) -> Response:
    """Start a download in the background and follow its progress page (FR-04.7a)."""
    form = await parse_form(request)
    repo = form.get("repo", "").strip()
    if not repo:
        return redirect_with_flash("/developer", "enter an HF repo to download")
    include = [
        line.strip() for line in form.get("include", "").splitlines() if line.strip()
    ]
    revision = form.get("revision", "").strip() or None
    try:
        download_id = await _start_download(request, repo, include, revision)
    except (asyncio.TimeoutError, ControllerUnavailable):
        return redirect_with_flash(
            "/developer", "controller unreachable — try again in a moment"
        )
    except RuntimeError as exc:
        return redirect_with_flash("/developer", str(exc))
    except StopAsyncIteration:
        return redirect_with_flash("/developer", "controller reported no download events")
    if not download_id:
        return redirect_with_flash("/developer", "controller did not report a download id")
    return redirect(f"/developer/downloads/{download_id}")


@router.get("/developer/downloads/{download_id}", name="developer_download")
async def download_page(request: Request, download_id: str) -> Response:
    """Progress page for one download, refreshing every 2 s until done (FR-04.7a)."""
    client = client_of(request)
    status = await grpc(client.status())
    downloads = await grpc(client.list_downloads())
    controller_down = status is None or downloads is None
    info = next(
        (download for download in (downloads or []) if download.id == download_id),
        None,
    )
    if info is None:
        return render(
            request,
            "download.html",
            {
                "page": "developer",
                "controller_down": controller_down,
                "active_alias": status.active_alias if status is not None else "",
                "not_found": True,
                "download_id": download_id,
            },
        )
    live = cast(
        dict[str, control_pb2.DownloadProgress],
        getattr(request.app.state, "download_live", {}),
    )
    latest = live.get(download_id) if live else None
    started = cast(
        dict[str, float], getattr(request.app.state, "download_started", {})
    ).get(download_id)
    pct = round(min(max(info.progress, 0.0), 1.0) * 100)
    return render(
        request,
        "download.html",
        {
            "page": "developer",
            "controller_down": controller_down,
            "active_alias": status.active_alias if status is not None else "",
            "not_found": False,
            "download_id": download_id,
            "info": info,
            "progress_pct": f"{pct}",
            "progress_class": f"bar-{min(100, pct // 5 * 5)}",
            "current_file": latest.current_file if latest is not None else "",
            "elapsed_s": time.time() - started if started is not None else None,
            "refresh": not info.done,
        },
    )


@router.post(
    "/developer/downloads/{download_id}/cancel", name="developer_download_cancel"
)
async def download_cancel(request: Request, download_id: str) -> Response:
    """Stop a download cooperatively and reload its page (FR-04.2, FR-04.7a)."""
    client = client_of(request)
    cancelled = await grpc(client.cancel_download(download_id))
    target = f"/developer/downloads/{download_id}"
    if cancelled is None:
        return redirect_with_flash(
            target, "controller unreachable — try again in a moment"
        )
    if cancelled:
        return redirect_with_flash(target, "download cancelled")
    return redirect_with_flash(target, "download could not be cancelled")


# -- HF cache (FR-04.7b/c) --------------------------------------------------


@router.get("/developer/cache/{repo:path}/confirm", name="developer_cache_confirm")
async def cache_confirm(request: Request, repo: str) -> Response:
    """Confirmation page before deleting a cached repo (FR-04.7b)."""
    client = client_of(request)
    status = await grpc(client.status())
    return render(
        request,
        "cache_rm_confirm.html",
        {
            "page": "developer",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
            "repo": repo,
        },
    )


@router.post("/developer/cache/{repo:path}/remove", name="developer_cache_remove")
async def cache_remove(request: Request, repo: str) -> Response:
    """Delete a cached repo and flash the outcome (FR-04.7b)."""
    client = client_of(request)
    result = await grpc(client.remove_cached(repo))
    if result is None:
        return redirect_with_flash(
            "/developer", "controller unreachable — try again in a moment"
        )
    removed, error = result
    if removed:
        return redirect_with_flash("/developer", f"removed {repo} from the HF cache")
    return redirect_with_flash(
        "/developer", f"could not remove {repo}: {error or 'unknown error'}"
    )


# -- registration (FR-04.7d) ------------------------------------------------


async def _render_register(
    request: Request,
    *,
    form: dict[str, str],
    errors: list[str],
    warnings: list[str],
    preview: str | None,
) -> Response:
    """Render the register form/preview page with the usual chrome (FR-04.7d)."""
    client = client_of(request)
    status = await grpc(client.status())
    return render(
        request,
        "register_preview.html",
        {
            "page": "developer",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
            "form": form,
            "errors": errors,
            "warnings": warnings,
            "preview": preview,
        },
    )


@router.get("/developer/register", name="developer_register_form")
async def register_form(request: Request, repo: str = "") -> Response:
    """Pre-filled registration form for a cached repo (FR-04.7d)."""
    return await _render_register(
        request,
        form=_register_form({}, repo),
        errors=[],
        warnings=[],
        preview=None,
    )


@router.post("/developer/register", name="developer_register")
async def register_post(request: Request) -> Response:
    """Propose a registration and show the YAML preview (FR-04.7d)."""
    client = client_of(request)
    form = await parse_form(request)
    if not form.get("repo", "").strip():
        return redirect_with_flash("/developer/register", "enter an HF repo to register")
    spec = _spec_from_form(form)
    proposal = await grpc(client.propose_registration(spec))
    if proposal is None:
        return redirect_with_flash(
            "/developer", "controller unreachable — try again in a moment"
        )
    filled = _register_form(form)
    if proposal.errors:
        return await _render_register(
            request,
            form=filled,
            errors=list(proposal.errors),
            warnings=list(proposal.warnings),
            preview=None,
        )
    preview = _preview_yaml(proposal.model, spec.tool_parser)
    return await _render_register(
        request,
        form=filled,
        errors=[],
        warnings=list(proposal.warnings),
        preview=preview,
    )


@router.post("/developer/register/confirm", name="developer_register_confirm")
async def register_confirm(request: Request) -> Response:
    """Write the registration and reload the registry (FR-04.7d)."""
    client = client_of(request)
    form = await parse_form(request)
    spec = _spec_from_form(form)
    filled = _register_form(form)
    write = await grpc(client.write_registration(spec))
    if write is None:
        return redirect_with_flash(
            "/developer", "controller unreachable — try again in a moment"
        )
    if not write.ok:
        proposal = await grpc(client.propose_registration(spec))
        preview = (
            _preview_yaml(proposal.model, spec.tool_parser)
            if proposal is not None
            else None
        )
        return await _render_register(
            request,
            form=filled,
            errors=list(write.errors),
            warnings=[],
            preview=preview,
        )
    errors = await grpc(client.reload_registry())
    if errors is None:
        return redirect_with_flash(
            "/developer", "controller unreachable — try again in a moment"
        )
    if errors:
        proposal = await grpc(client.propose_registration(spec))
        preview = (
            _preview_yaml(proposal.model, spec.tool_parser)
            if proposal is not None
            else None
        )
        return await _render_register(
            request, form=filled, errors=list(errors), warnings=[], preview=preview
        )
    return redirect_with_flash(
        "/models", f"alias {spec.alias} added — models.yaml updated"
    )
