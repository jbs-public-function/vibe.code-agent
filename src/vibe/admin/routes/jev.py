"""Jev schema pages: list, editor, delete, launch, quick builder (doc 05).

Zero-JS admin (JEV-X.5): every interaction is a plain HTML form; POSTs
303-redirect (PRG) except validation failures, which re-render the editor
with the user's exact text preserved (JEV-05 design note). Controller access
goes through the shared wrappers (`safe_grpc`, `grpc`, `render`, …) so the
3 s deadline and the "controller unreachable" degradation behave like every
other page (JEV-05.2).

JEV-05.7: POST bodies for `/jev*` routes are capped at 128 KiB *before*
url-decoding (urlencoding inflates up to 3×: a 64 KiB schema plus fields).
The shared `parse_form` stays untouched — other routers keep today's
behaviour.
"""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qsl

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from starlette.responses import Response

from vibe.admin.jev_builder import build_decision_schema
from vibe.admin.routes import (
    client_of,
    fmt_ts,
    grpc,
    grpc_flash,
    redirect,
    redirect_with_flash,
    render,
    safe_grpc,
    start_background_job,
)
from vibe.client import ControllerUnavailable, JevReadOnly, JevSchemaNotFound
from vibe.gen.vibe.v1 import control_pb2
from vibe.jev import StoreError, validate_schema_text
from vibe.jev.store import SchemaStore
from vibe.jobs import JobInProgress

router = APIRouter()

JEV_POST_BODY_LIMIT = 128 * 1024  # JEV-05.7

SOURCE_LABELS = {
    "builtin": "built-in",
    "user": "user",
    "override": "overrides built-in",
}

# Minimal prefill for the empty editor (JEV-05.3); mirrors the yes-no preset.
MINIMAL_TEMPLATE = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MyDecision",
  "type": "object",
  "properties": {
    "decision_type": { "const": "choice" },
    "result": { "type": "string", "enum": ["YES", "NO"], "description": "The chosen action." },
    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
  },
  "required": ["decision_type", "result", "confidence"],
  "additionalProperties": false
}
"""

_TOO_LARGE_BODY = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    "<title>413 Request too large</title></head><body>"
    "<h1>413 Request too large</h1>"
    "<p>The form body exceeds the 128 KiB admin limit for Jev pages.</p>"
    "</body></html>"
)


# -- small helpers ----------------------------------------------------------


def _too_large() -> HTMLResponse:
    """413 (HTML) for POST bodies over the `/jev*` cap (JEV-05.7)."""
    return HTMLResponse(_TOO_LARGE_BODY, status_code=413)


async def _jev_form(request: Request) -> dict[str, str] | None:
    """Parse a `/jev*` POST body; `None` when it exceeds the byte cap.

    The cap is enforced before `parse_qsl` (JEV-05.7). `Content-Length` is
    honoured up front so an oversized body is never buffered; a missing or
    malformed length falls through to the measured check.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > JEV_POST_BODY_LIMIT:
                return None
        except ValueError:
            pass
    body = await request.body()
    if len(body) > JEV_POST_BODY_LIMIT:
        return None
    try:
        text = body.decode()
    except UnicodeDecodeError:
        return {}
    return dict(parse_qsl(text, keep_blank_values=True))


def _normalise(text: str) -> str:
    """CRLF → LF: browsers normalise textarea line endings on submit (doc 05)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _name_error(name: str) -> str | None:
    """Store-rule name problems as a message, or `None` (JEV-01.9)."""
    try:
        SchemaStore.check_name(name)
    except StoreError as exc:
        return str(exc)
    return None


def _invalid_path_name(name: str) -> bool:
    """Whether a path-param name is invalid (regex or reserved word)."""
    return _name_error(name) is not None


def _parse_float(raw: str, default: float) -> float | None:
    """`float(raw)` with the quick-builder default; `None` on garbage."""
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return None


async def _not_found(request: Request, name: str) -> Response:
    """404 page for a path name the store rules reject (JEV-05.9)."""
    client = client_of(request)
    status, status_error = await safe_grpc(client.status())
    response = render(
        request,
        "jev_not_found.html",
        {
            "page": "jev",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
            "name": name,
        },
        error=status_error,
    )
    response.status_code = 404
    return response


async def _render_editor(
    request: Request,
    *,
    name: str,
    name_editable: bool,
    schema_json: str,
    base_sha: str,
    errors: list[str],
    warnings: list[str],
    notice: str | None,
    conflict: bool,
    source: str,
    error: str | None = None,
) -> Response:
    """The editor page with the model select for the launch buttons."""
    client = client_of(request)
    (status, status_error), (infos, infos_error) = await asyncio.gather(
        safe_grpc(client.status()), safe_grpc(client.list_models())
    )
    return render(
        request,
        "jev_edit.html",
        {
            "page": "jev",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
            "infos": infos if infos is not None else [],
            "reasoning_models": [
                info.alias for info in (infos or []) if "reasoning" in info.tags
            ],
            "name": name,
            "name_editable": name_editable,
            "schema_json": schema_json,
            "base_sha": base_sha,
            "errors": errors,
            "warnings": warnings,
            "notice": notice,
            "conflict": conflict,
            "source": source,
            "builtin": source == "builtin",
        },
        error=error or status_error or infos_error,
    )


async def _launch(
    request: Request,
    alias: str,
    force: bool,
    *,
    schema_name: str = "",
    inline_json: str = "",
    flash_target: str = "/jev",
) -> Response:
    """`LoadModel(JEV_ON, …)` and follow the job (JEV-05.6).

    A `jev_*` error on the **first** event is flashed (redirect), never a
    500 — the controller reports Jev failures as events, not exceptions.
    """
    client = client_of(request)
    if schema_name:
        spec = control_pb2.JevSpec(schema_name=schema_name)
    else:
        spec = control_pb2.JevSpec(schema_json=inline_json)
    agen = client.load(alias, force=force, jev_mode=control_pb2.JEV_ON, jev=spec)
    try:
        job_id, first = await start_background_job(request, agen, "LOAD", alias)
    except JobInProgress:
        status = await grpc(client.status())
        existing = ""
        if status is not None and status.HasField("current_job"):
            existing = status.current_job.id
        return redirect_with_flash(
            f"/jobs/{existing}" if existing else "/jev", "a job is already running"
        )
    except KeyError:
        return redirect_with_flash("/jev", f"unknown alias: {alias}")
    except StopAsyncIteration:
        # IMP-04.15: an empty job stream must not become a 500.
        return redirect_with_flash("/jev", "controller reported no job events")
    except (asyncio.TimeoutError, ControllerUnavailable):
        return redirect_with_flash(
            "/jev", "controller unreachable — try again in a moment"
        )
    except RuntimeError as exc:
        return redirect_with_flash("/jev", str(exc))
    if first.error_code:
        # JEV-05.6: a Jev failure event is an outcome, not a 500.
        return redirect_with_flash(
            flash_target, f"launch failed: {first.message or first.error_code}"
        )
    if not job_id:
        # Defensive: the controller should always tag the first event.
        status = await grpc(client.status())
        if status is not None and status.HasField("current_job"):
            return redirect(f"/jobs/{status.current_job.id}")
        return redirect_with_flash("/jev", "controller did not report a job id")
    return redirect(f"/jobs/{job_id}")


async def _editor_post(
    request: Request,
    name: str,
    form: dict[str, str],
    schema_json: str,
    base_sha: str,
    *,
    name_editable: bool,
) -> Response:
    """The four editor actions (JEV-05.4..05.6); 200 re-render on failure."""
    client = client_of(request)
    alias = form.get("alias", "").strip()
    force = form.get("force") == "on"
    action = form.get("action", "save")

    async def rerender(
        errors: list[str],
        *,
        warnings: list[str] | None = None,
        notice: str | None = None,
        conflict: bool = False,
        text: str | None = None,
        sha: str | None = None,
    ) -> Response:
        return await _render_editor(
            request,
            name=name,
            name_editable=name_editable,
            schema_json=schema_json if text is None else text,
            base_sha=base_sha if sha is None else sha,
            errors=errors,
            warnings=warnings or [],
            notice=notice,
            conflict=conflict,
            source="",
        )

    if name_editable:
        problem = _name_error(name) if name else "enter a schema name"
        if problem:
            return await rerender([problem])

    if action in ("save", "save_launch") and base_sha:
        # JEV-05.17: cheap optimistic concurrency — refuse the save when the
        # stored schema changed since the form was rendered.
        try:
            current = await grpc(client.get_jev_schema(name))
        except (JevSchemaNotFound, RuntimeError):
            current = None
        if current is not None and current.sha256 != base_sha:
            return await rerender(
                [],
                conflict=True,
                text=current.schema_json,
                sha=current.sha256,
            )

    if action == "validate":
        result, result_error = await grpc_flash(client.validate_jev_schema(schema_json))
        if result is None:
            return await rerender(
                [result_error or "controller unreachable — try again in a moment"]
            )
        if result.ok:
            return await rerender(
                [],
                warnings=list(result.warnings),
                notice=(
                    f"✔ valid (sha {result.sha256[:12]}, {result.size_bytes} bytes)"
                ),
            )
        return await rerender(list(result.errors), warnings=list(result.warnings))

    if action == "save":
        result, result_error = await grpc_flash(client.put_jev_schema(name, schema_json))
        if result is None:
            return await rerender(
                [result_error or "controller unreachable — try again in a moment"]
            )
        if result.error:
            return await rerender([result.error])
        validation = result.validation
        if not validation.ok:
            return await rerender(
                list(validation.errors), warnings=list(validation.warnings)
            )
        message = f"saved {name!r}"
        active = await grpc(client.get_active_jev())
        if (
            active is not None
            and active.HasField("info")
            and active.info.schema_name == name
        ):
            # JEV-05.5: saving never changes live behaviour.
            message += " — active Jev is now stale; re-launch to apply"
        return redirect_with_flash(f"/jev/{name}", message)

    if action == "save_launch":
        if not alias:
            return await rerender(["pick a model to launch on"])
        result, result_error = await grpc_flash(client.put_jev_schema(name, schema_json))
        if result is None:
            return await rerender(
                [result_error or "controller unreachable — try again in a moment"]
            )
        if result.error:
            return await rerender([result.error])
        validation = result.validation
        if not validation.ok:
            return await rerender(
                list(validation.errors), warnings=list(validation.warnings)
            )
        return await _launch(request, alias, force, schema_name=name, flash_target=f"/jev/{name}")

    if action == "launch_unsaved":
        if not alias:
            return await rerender(["pick a model to launch on"])
        return await _launch(
            request, alias, force, inline_json=schema_json, flash_target=f"/jev/{name}"
        )

    # Unknown action: re-render with the input intact.
    return await rerender([f"unknown action: {action!r}"])


# -- routes -----------------------------------------------------------------


@router.get("/jev", name="jev")
async def jev_page(request: Request) -> Response:
    """Schema table + active-Jev panel + launch form (JEV-05.1/05.2)."""
    client = client_of(request)
    (schemas, schemas_error), (status, status_error), (
        active,
        active_error,
    ), (infos, infos_error) = await asyncio.gather(
        safe_grpc(client.list_jev_schemas()),
        safe_grpc(client.status()),
        safe_grpc(client.get_active_jev()),
        safe_grpc(client.list_models()),
    )
    controller_down = (
        schemas is None or status is None or active is None or infos is None
    )
    rows = [
        {
            "name": entry.name,
            "source": entry.source,
            "source_label": SOURCE_LABELS.get(entry.source, entry.source),
            "sha12": entry.sha256[:12],
            "size_bytes": entry.size_bytes,
            "mtime_label": fmt_ts(entry.mtime_unix_ms) if entry.mtime_unix_ms else "—",
            "active": entry.active,
        }
        for entry in (schemas or [])
    ]
    active_view = None
    if active is not None and active.HasField("info"):
        info = active.info
        active_view = {
            "name": info.schema_name,
            "source": info.source,
            "sha12": info.sha256[:12],
            "applied_label": (
                fmt_ts(info.applied_at_unix_ms) if info.applied_at_unix_ms else "—"
            ),
            "stale": info.stale,
        }
    return render(
        request,
        "jev.html",
        {
            "page": "jev",
            "controller_down": controller_down,
            "active_alias": status.active_alias if status is not None else "",
            "schema_rows": rows,
            "active": active_view,
            "infos": infos if infos is not None else [],
            "reasoning_models": [
                info.alias for info in (infos or []) if "reasoning" in info.tags
            ],
        },
        error=schemas_error or status_error or active_error or infos_error,
    )


@router.get("/jev/new", name="jev_new")
async def jev_new(request: Request, name: str = "") -> Response:
    """Empty editor; `?name=` prefills a copy of that schema (JEV-05.3)."""
    client = client_of(request)
    schema_json = MINIMAL_TEMPLATE
    base_sha = ""
    if name:
        try:
            schema = await grpc(client.get_jev_schema(name))
        except (JevSchemaNotFound, RuntimeError):
            schema = None
        if schema is not None:
            schema_json = schema.schema_json
            base_sha = schema.sha256
    return await _render_editor(
        request,
        name=name,
        name_editable=True,
        schema_json=schema_json,
        base_sha=base_sha,
        errors=[],
        warnings=[],
        notice=None,
        conflict=False,
        source="",
    )


@router.post("/jev/new", name="jev_new_post")
async def jev_new_post(request: Request) -> Response:
    """Create/save from the new-schema editor; name comes from the form."""
    form = await _jev_form(request)
    if form is None:
        return _too_large()
    name = form.get("name", "").strip()
    schema_json = _normalise(form.get("schema_json", ""))
    return await _editor_post(
        request,
        name,
        form,
        schema_json,
        form.get("base_sha", ""),
        name_editable=True,
    )


@router.post("/jev/launch", name="jev_launch")
async def jev_launch(request: Request) -> Response:
    """Launch an existing schema on a model (JEV-05.1)."""
    form = await _jev_form(request)
    if form is None:
        return _too_large()
    alias = form.get("alias", "").strip()
    schema_name = form.get("schema_name", "").strip()
    force = form.get("force") == "on"
    if not alias:
        return redirect_with_flash("/jev", "pick a model to launch")
    if not schema_name:
        return redirect_with_flash("/jev", "pick a schema to launch")
    problem = _name_error(schema_name)
    if problem:
        return redirect_with_flash("/jev", f"invalid schema: {problem}")
    return await _launch(request, alias, force, schema_name=schema_name)


@router.post("/jev/detach", name="jev_detach")
async def jev_detach(request: Request) -> Response:
    """`LoadModel(active_alias, JEV_OFF)` → the job page (JEV-05.1)."""
    form = await _jev_form(request)
    if form is None:
        return _too_large()
    client = client_of(request)
    status = await grpc(client.status())
    if status is None:
        return redirect_with_flash(
            "/jev", "controller unreachable — try again in a moment"
        )
    alias = status.active_alias
    if not alias:
        return redirect_with_flash("/jev", "no model loaded")
    agen = client.load(alias, jev_mode=control_pb2.JEV_OFF)
    try:
        job_id, _first = await start_background_job(request, agen, "LOAD", alias)
    except JobInProgress:
        return redirect_with_flash("/jev", "a job is already running")
    except KeyError:
        return redirect_with_flash("/jev", f"unknown alias: {alias}")
    except StopAsyncIteration:
        return redirect_with_flash("/jev", "controller reported no job events")
    except (asyncio.TimeoutError, ControllerUnavailable):
        return redirect_with_flash(
            "/jev", "controller unreachable — try again in a moment"
        )
    except RuntimeError as exc:
        return redirect_with_flash("/jev", str(exc))
    if not job_id:
        return redirect_with_flash("/jev", "controller did not report a job id")
    return redirect(f"/jobs/{job_id}")


@router.post("/jev/build", name="jev_build")
async def jev_build(request: Request) -> Response:
    """Quick builder: compose the decision schema, re-render the editor (P2)."""
    form = await _jev_form(request)
    if form is None:
        return _too_large()
    name = form.get("name", "").strip()
    base_sha = form.get("base_sha", "")
    name_editable = form.get("name_editable") == "on"
    previous = _normalise(form.get("schema_json", ""))
    decision_types = [
        kind
        for kind in ("choice", "boolean", "score")
        if form.get(f"decision_{kind}") == "on"
    ]
    options = [
        line.strip()
        for line in form.get("choice_options", "").splitlines()
        if line.strip()
    ]
    score_min = _parse_float(form.get("score_min", ""), 0.0)
    score_max = _parse_float(form.get("score_max", ""), 1.0)
    include_confidence = form.get("include_confidence") == "on"
    additional_properties = form.get("additional_properties") == "on"

    errors: list[str] = []
    if score_min is None or score_max is None:
        errors.append("score minimum and maximum must be numbers")
        score_min, score_max = 0.0, 1.0
    if not decision_types:
        errors.append("pick at least one decision type")
    if "choice" in decision_types and not options:
        errors.append("enter at least one choice option (one per line)")
    if "score" in decision_types and score_min > score_max:
        errors.append("score minimum must not exceed the maximum")

    if errors:
        return await _render_editor(
            request,
            name=name,
            name_editable=name_editable,
            schema_json=previous,
            base_sha=base_sha,
            errors=errors,
            warnings=[],
            notice=None,
            conflict=False,
            source="",
        )
    text = build_decision_schema(
        decision_types,
        options,
        score_min,
        score_max,
        include_confidence=include_confidence,
        additional_properties_false=additional_properties,
    )
    result = validate_schema_text(text)
    if not result.ok:
        # Defensive: the builder's output is covered by tests (E11).
        return await _render_editor(
            request,
            name=name,
            name_editable=name_editable,
            schema_json=previous,
            base_sha=base_sha,
            errors=list(result.errors),
            warnings=[],
            notice=None,
            conflict=False,
            source="",
        )
    return await _render_editor(
        request,
        name=name,
        name_editable=name_editable,
        schema_json=text,
        base_sha=base_sha,
        errors=[],
        warnings=[],
        notice="generated — tweak and save (nothing was stored)",
        conflict=False,
        source="",
    )


@router.get("/jev/{name}/delete", name="jev_delete_confirm")
async def jev_delete_confirm(request: Request, name: str) -> Response:
    """Confirmation page; overrides note the revert to the built-in (JEV-05.10)."""
    if _invalid_path_name(name):
        return await _not_found(request, name)
    client = client_of(request)
    schema = None
    missing = False
    fetch_error: str | None = None
    try:
        schema = await grpc(client.get_jev_schema(name))
    except JevSchemaNotFound:
        missing = True
    except RuntimeError as exc:
        fetch_error = str(exc)
    if missing:
        return await _not_found(request, name)
    status, status_error = await safe_grpc(client.status())
    source = schema.source if schema is not None else ""
    return render(
        request,
        "jev_delete_confirm.html",
        {
            "page": "jev",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
            "name": name,
            "source": source,
            "builtin": source == "builtin",
        },
        error=fetch_error or status_error,
    )


@router.post("/jev/{name}/delete", name="jev_delete")
async def jev_delete(request: Request, name: str) -> Response:
    """Delete a user schema; overrides revert to the built-in (JEV-05.9)."""
    if _invalid_path_name(name):
        return await _not_found(request, name)
    form = await _jev_form(request)
    if form is None:
        return _too_large()
    client = client_of(request)
    try:
        result = await grpc(client.delete_jev_schema(name))
    except JevReadOnly:
        # E9: a built-in-only schema cannot be deleted.
        return redirect_with_flash(
            "/jev", f"{name} is a built-in schema — built-ins cannot be deleted"
        )
    except JevSchemaNotFound:
        return redirect_with_flash("/jev", f"no schema named {name}")
    except RuntimeError as exc:
        return redirect_with_flash("/jev", str(exc))
    if result is None:
        return redirect_with_flash(
            "/jev", "controller unreachable — try again in a moment"
        )
    if not result.ok:
        return redirect_with_flash(
            "/jev", f"could not delete {name}: {result.error or 'unknown error'}"
        )
    if result.reverted_to == "builtin":
        return redirect_with_flash("/jev", f"deleted {name} — reverted to the built-in")
    return redirect_with_flash("/jev", f"deleted {name}")


@router.get("/jev/{name}", name="jev_edit")
async def jev_edit(request: Request, name: str) -> Response:
    """Editor for one schema; built-ins render read-only (JEV-05.9)."""
    if _invalid_path_name(name):
        return await _not_found(request, name)
    client = client_of(request)
    schema = None
    missing = False
    fetch_error: str | None = None
    try:
        schema = await grpc(client.get_jev_schema(name))
    except JevSchemaNotFound:
        missing = True
    except RuntimeError as exc:
        fetch_error = str(exc)
    if missing:
        return await _not_found(request, name)
    if schema is None:
        # Controller down (or a real error): empty editor; the banner explains.
        return await _render_editor(
            request,
            name=name,
            name_editable=False,
            schema_json="",
            base_sha="",
            errors=[],
            warnings=[],
            notice=None,
            conflict=False,
            source="",
            error=fetch_error,
        )
    return await _render_editor(
        request,
        name=name,
        name_editable=False,
        schema_json=schema.schema_json,
        base_sha=schema.sha256,
        errors=[],
        warnings=[],
        notice=None,
        conflict=False,
        source=schema.source,
        error=fetch_error,
    )


@router.post("/jev/{name}", name="jev_edit_post")
async def jev_edit_post(request: Request, name: str) -> Response:
    """The four editor actions on an existing schema (name from the path)."""
    if _invalid_path_name(name):
        return await _not_found(request, name)
    form = await _jev_form(request)
    if form is None:
        return _too_large()
    schema_json = _normalise(form.get("schema_json", ""))
    return await _editor_post(
        request,
        name,
        form,
        schema_json,
        form.get("base_sha", ""),
        name_editable=False,
    )
