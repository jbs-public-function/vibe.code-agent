"""Developer tools: downloads, HF cache and guided registration (FR-04.7, NFR-04.5).

`GET /developer` shows the Phase 4 sections: the new-download form plus a
table of in-flight/recent downloads, the HF cache browser with a disk-usage
line, the read-only Zed settings block generated from the registry, and the
entry point to the guided registration flow. `POST /developer/downloads`
starts a download stream in a background task and 303-redirects to its
progress page, which auto-refreshes every 2 s until the download finishes.
`GET /developer/register` pre-fills the registration form from a cached repo;
`POST /developer/register` proposes the entry and shows the controller's
`yaml_preview` verbatim (IMP-04.19); `POST /developer/register/confirm`
re-proposes, diffs against the previewed values, writes to `models.yaml` and
reloads the registry (IMP-04.22). `POST /developer/reload` is the standalone
"Reload registry" recovery button (IMP-04.17). Zero JavaScript
(NFR-04.5/NFR-01.1); every controller call goes through the shared `grpc()`
deadline helper (NFR-01.5).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from contextlib import suppress
from typing import cast

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from vibe.admin.routes import (
    GRPC_TIMEOUT,
    aclose,
    client_of,
    grpc_flash,
    parse_form,
    redirect,
    redirect_with_flash,
    render,
    safe_grpc,
)
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2
from vibe.zedsettings import ZedEntry, zed_settings_json

router = APIRouter()

# Defaults for the register form; the controller fills 0/"" as "auto" (FR-04.5).
# IMP-04.23: passed into the template — never hardcoded there a second time.
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


def _spec_from_form(
    form: dict[str, str],
) -> tuple[control_pb2.RegistrationSpec, list[str]]:
    """Build a `RegistrationSpec`; blank/auto values stay 0/"" (auto upstream).

    IMP-04.18: unparseable numeric fields (`"80a0"`, `"0.7x"`) produce an
    error message instead of being silently coerced to 0.
    """
    errors: list[str] = []

    def _int(key: str, label: str) -> int:
        raw = form.get(key, "").strip()
        if not raw:
            return 0
        try:
            return int(raw)
        except ValueError:
            errors.append(f"{label} must be a whole number, got {raw!r}")
            return 0

    def _float(key: str, label: str) -> float:
        raw = form.get(key, "").strip()
        if not raw:
            return 0.0
        try:
            return float(raw)
        except ValueError:
            errors.append(f"{label} must be a number, got {raw!r}")
            return 0.0

    quantization = form.get("quantization", "").strip()
    if quantization == "auto":
        quantization = ""
    tool_parser = form.get("tool_parser", "").strip()
    if tool_parser == "auto":
        tool_parser = ""
    tags = [tag.strip() for tag in form.get("tags", "").split(",") if tag.strip()]
    spec = control_pb2.RegistrationSpec(
        hf_repo=form.get("repo", "").strip(),
        alias=form.get("alias", "").strip(),
        quantization=quantization,
        max_model_len=_int("max_model_len", "max model len"),
        gpu_memory_utilization=_float("gpu_mem", "GPU mem fraction"),
        host_port=_int("port", "port"),
        tool_parser=tool_parser,
        use_case=form.get("use_case", "").strip(),
        tags=tags,
    )
    return spec, errors


def _confirm_values(model: control_pb2.ModelInfo, tool_parser: str) -> dict[str, str]:
    """The controller-filled values hidden on the preview page (IMP-04.22).

    The confirm POST re-proposes these and diffs against them, so alias/port
    cannot drift between what was previewed and what gets written.
    """
    return {
        "repo": model.hf_repo,
        "alias": model.alias,
        "quantization": model.quantization or "auto",
        "max_model_len": str(model.max_model_len) if model.max_model_len else "",
        "gpu_mem": f"{model.gpu_memory_utilization:g}"
        if model.gpu_memory_utilization
        else "",
        "port": str(model.host_port) if model.host_port else "",
        "tool_parser": tool_parser or "auto",
        "use_case": model.use_case,
        "tags": ", ".join(model.tags),
    }


def _diff_preview(
    previewed: dict[str, str], proposed: control_pb2.ModelInfo, tool_parser: str
) -> list[str]:
    """Drift between the previewed values and a fresh proposal (IMP-04.22)."""
    fresh = _confirm_values(proposed, tool_parser)
    drift: list[str] = []
    for key in ("repo", "alias", "port", "quantization", "max_model_len", "gpu_mem"):
        if fresh[key] != previewed.get(key, ""):
            drift.append(
                f"{key} changed since the preview "
                f"({previewed.get(key, '') or 'auto'} → {fresh[key] or 'auto'}) — "
                "review the preview and confirm again"
            )
    return drift


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
    task held in `app.state.background_tasks`. Progress is read from the
    controller's `ListDownloads` on every page render — no admin-side
    side channels (IMP-04.20).
    """
    client = client_of(request)
    agen = client.download(repo, include=include, revision=revision)
    try:
        first = await asyncio.wait_for(anext(agen), timeout=GRPC_TIMEOUT)
    except asyncio.TimeoutError:
        with suppress(Exception):
            await aclose(agen)  # IMP-04.15: release the stream on deadline
        raise
    download_id = first.id

    async def drain() -> None:
        try:
            async for _ in agen:
                pass
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
    # IMP-04.16: the four independent calls run concurrently under the shared
    # 3 s budget instead of 4 × 3 s back-to-back.
    (status, status_error), (downloads, downloads_error), (cache, cache_error), (
        models,
        models_error,
    ) = await asyncio.gather(
        safe_grpc(client.status()),
        safe_grpc(client.list_downloads()),
        safe_grpc(client.list_cache()),
        safe_grpc(client.list_models()),
    )
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
        error=status_error or downloads_error or cache_error or models_error,
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
    (status, status_error), (downloads, downloads_error) = await asyncio.gather(
        safe_grpc(client.status()), safe_grpc(client.list_downloads())
    )
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
            error=status_error or downloads_error,
        )
    # IMP-04.20: current file and timestamps come straight from `DownloadInfo`;
    # the admin keeps no `download_live`/`download_started` side channels.
    elapsed_s = (
        time.time() - info.started_at_unix_ms / 1000.0
        if info.started_at_unix_ms
        else None
    )
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
            "current_file": info.current_file,
            "elapsed_s": elapsed_s,
            "bytes_done_label": _fmt_size(info.bytes_done),
            "bytes_total_label": _fmt_size(info.bytes_total),
            "refresh": not info.done,
        },
        error=status_error or downloads_error,
    )


@router.post(
    "/developer/downloads/{download_id}/cancel", name="developer_download_cancel"
)
async def download_cancel(request: Request, download_id: str) -> Response:
    """Stop a download cooperatively and reload its page (FR-04.2, FR-04.7a)."""
    client = client_of(request)
    target = f"/developer/downloads/{download_id}"
    try:
        cancelled, error = await grpc_flash(client.cancel_download(download_id))
    except KeyError:
        # IMP-04.21: an unknown id is a user-visible outcome, not a 500.
        return redirect_with_flash(target, "unknown download")
    if cancelled is None:
        return redirect_with_flash(
            target, error or "controller unreachable — try again in a moment"
        )
    if cancelled:
        return redirect_with_flash(target, "download cancelled")
    return redirect_with_flash(target, "download could not be cancelled")


# -- HF cache (FR-04.7b/c) --------------------------------------------------


@router.get("/developer/cache/{repo:path}/confirm", name="developer_cache_confirm")
async def cache_confirm(request: Request, repo: str) -> Response:
    """Confirmation page before deleting a cached repo (FR-04.7b)."""
    client = client_of(request)
    status, error = await safe_grpc(client.status())
    return render(
        request,
        "cache_rm_confirm.html",
        {
            "page": "developer",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
            "repo": repo,
        },
        error=error,
    )


@router.post("/developer/cache/{repo:path}/remove", name="developer_cache_remove")
async def cache_remove(request: Request, repo: str) -> Response:
    """Delete a cached repo and flash the outcome (FR-04.7b)."""
    client = client_of(request)
    try:
        result, error = await grpc_flash(client.remove_cached(repo))
    except KeyError:
        # IMP-04.21: an unknown repo is a user-visible outcome, not a 500.
        return redirect_with_flash("/developer", "unknown repo")
    if result is None:
        return redirect_with_flash(
            "/developer", error or "controller unreachable — try again in a moment"
        )
    removed, remove_error = result
    if removed:
        return redirect_with_flash("/developer", f"removed {repo} from the HF cache")
    return redirect_with_flash(
        "/developer", f"could not remove {repo}: {remove_error or 'unknown error'}"
    )


# -- registration (FR-04.7d) ------------------------------------------------


async def _render_register(
    request: Request,
    *,
    form: dict[str, str],
    confirm: dict[str, str] | None,
    errors: list[str],
    warnings: list[str],
    preview: str | None,
) -> Response:
    """Render the register form/preview page with the usual chrome (FR-04.7d)."""
    client = client_of(request)
    status, error = await safe_grpc(client.status())
    return render(
        request,
        "register_preview.html",
        {
            "page": "developer",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
            "form": form,
            "confirm": confirm,
            "errors": errors,
            "warnings": warnings,
            "preview": preview,
            # IMP-04.23: options live in one place (this module).
            "quantization_options": _QUANTIZATION_OPTIONS,
            "tool_parser_options": _TOOL_PARSER_OPTIONS,
        },
        error=error,
    )


@router.get("/developer/register", name="developer_register_form")
async def register_form(request: Request, repo: str = "") -> Response:
    """Pre-filled registration form for a cached repo (FR-04.7d)."""
    return await _render_register(
        request,
        form=_register_form({}, repo),
        confirm=None,
        errors=[],
        warnings=[],
        preview=None,
    )


@router.post("/developer/register", name="developer_register")
async def register_post(request: Request) -> Response:
    """Propose a registration and show the controller's YAML preview (FR-04.7d)."""
    client = client_of(request)
    form = await parse_form(request)
    if not form.get("repo", "").strip():
        return redirect_with_flash("/developer/register", "enter an HF repo to register")
    filled = _register_form(form)
    spec, parse_errors = _spec_from_form(form)
    if parse_errors:
        # IMP-04.18: re-render the form with the messages; no proposal is made.
        return await _render_register(
            request,
            form=filled,
            confirm=None,
            errors=parse_errors,
            warnings=[],
            preview=None,
        )
    proposal, proposal_error = await grpc_flash(client.propose_registration(spec))
    if proposal is None:
        return redirect_with_flash(
            "/developer",
            proposal_error or "controller unreachable — try again in a moment",
        )
    if proposal.errors:
        return await _render_register(
            request,
            form=filled,
            confirm=None,
            errors=list(proposal.errors),
            warnings=list(proposal.warnings),
            preview=None,
        )
    # IMP-04.19: the controller's serialization is authoritative (XR-3).
    return await _render_register(
        request,
        form=filled,
        confirm=_confirm_values(proposal.model, spec.tool_parser),
        errors=[],
        warnings=list(proposal.warnings),
        preview=proposal.model.yaml_preview,
    )


@router.post("/developer/register/confirm", name="developer_register_confirm")
async def register_confirm(request: Request) -> Response:
    """Write the registration and reload the registry (FR-04.7d).

    IMP-04.22: the spec is re-proposed first and diffed against the values
    previewed on the previous page, so alias/port cannot drift between
    preview and write.
    """
    client = client_of(request)
    form = await parse_form(request)
    previewed = _register_form(form)
    spec, parse_errors = _spec_from_form(form)
    if parse_errors:
        return await _render_register(
            request,
            form=previewed,
            confirm=dict(form),
            errors=parse_errors,
            warnings=[],
            preview=None,
        )
    proposal, proposal_error = await grpc_flash(client.propose_registration(spec))
    if proposal is None:
        return redirect_with_flash(
            "/developer",
            proposal_error or "controller unreachable — try again in a moment",
        )
    if proposal.errors:
        return await _render_register(
            request,
            form=previewed,
            confirm=dict(form),
            errors=list(proposal.errors),
            warnings=[],
            preview=proposal.model.yaml_preview,
        )
    drift = _diff_preview(dict(form), proposal.model, spec.tool_parser)
    if drift:
        return await _render_register(
            request,
            form=previewed,
            confirm=_confirm_values(proposal.model, spec.tool_parser),
            errors=drift,
            warnings=[],
            preview=proposal.model.yaml_preview,
        )
    write, write_error = await grpc_flash(client.write_registration(spec))
    if write is None:
        return redirect_with_flash(
            "/developer",
            write_error or "controller unreachable — try again in a moment",
        )
    if not write.ok:
        return await _render_register(
            request,
            form=previewed,
            confirm=_confirm_values(proposal.model, spec.tool_parser),
            errors=list(write.errors),
            warnings=[],
            preview=proposal.model.yaml_preview,
        )
    errors, reload_error = await grpc_flash(client.reload_registry())
    if errors is None:
        return redirect_with_flash(
            "/developer",
            reload_error or "controller unreachable — try again in a moment",
        )
    if errors:
        # IMP-04.17: the write succeeded but the reload failed — the error
        # page offers the standalone Reload registry recovery button.
        return await _render_register(
            request,
            form=previewed,
            confirm=_confirm_values(proposal.model, spec.tool_parser),
            errors=list(errors),
            warnings=[],
            preview=proposal.model.yaml_preview,
        )
    return redirect_with_flash(
        "/models", f"alias {spec.alias} added — models.yaml updated"
    )


@router.post("/developer/reload", name="developer_reload")
async def developer_reload(request: Request) -> Response:
    """Standalone registry hot-reload (IMP-04.17, FR-04.7d)."""
    client = client_of(request)
    errors, error = await grpc_flash(client.reload_registry())
    if errors is None:
        return redirect_with_flash(
            "/developer", error or "controller unreachable — try again in a moment"
        )
    if errors:
        return redirect_with_flash("/developer", f"reload failed: {errors[0]}")
    return redirect_with_flash("/developer", "registry reloaded")
