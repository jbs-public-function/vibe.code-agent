"""Model registry and load/unload routes (FR-01.2, FR-01.3, NFR-01.5).

`GET /models` lists every registry model with its live state. `POST
/models/load` and `POST /models/unload` start the job stream in a background
asyncio task and 303-redirect to the job page; a running job or an unknown
alias redirects with a flash message instead.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from vibe.admin.routes import (
    client_of,
    grpc,
    parse_form,
    redirect,
    redirect_with_flash,
    render,
    start_background_job,
)
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2
from vibe.jobs import JobInProgress

router = APIRouter()


@router.get("/models", name="models")
async def models_page(request: Request) -> Response:
    """Registry table plus the load/unload forms (FR-01.2)."""
    client = client_of(request)
    infos = await grpc(client.list_models())
    status = await grpc(client.status())
    controller_down = infos is None or status is None
    # FR-04.9: uncached, non-loading models get a Download button that POSTs
    # into the /developer download flow.
    downloadable: set[str] = set()
    if infos is not None:
        downloadable = {
            info.alias
            for info in infos
            if not info.cached_locally and info.state != control_pb2.LOADING
        }
    return render(
        request,
        "models.html",
        {
            "page": "models",
            "controller_down": controller_down,
            "infos": infos if infos is not None else [],
            "active_alias": status.active_alias if status is not None else "",
            "downloadable": downloadable,
        },
    )


async def _start_job(
    request: Request, alias: str, force: bool, unload: bool
) -> Response:
    """Start a load/unload stream in the background; redirect to its job (FR-01.3)."""
    client = client_of(request)
    agen = client.unload() if unload else client.load(alias, force=force)
    kind = "UNLOAD" if unload else "LOAD"
    try:
        job_id = await start_background_job(request, agen, kind, alias if not unload else "")
    except JobInProgress:
        status = await grpc(client.status())
        existing = ""
        if status is not None and status.HasField("current_job"):
            existing = status.current_job.id
        return redirect_with_flash(
            f"/jobs/{existing}" if existing else "/models",
            "a job is already running",
        )
    except KeyError:
        return redirect_with_flash("/models", f"unknown alias: {alias}")
    except (asyncio.TimeoutError, ControllerUnavailable):
        return redirect_with_flash(
            "/models", "controller unreachable — try again in a moment"
        )
    except RuntimeError as exc:
        return redirect_with_flash("/models", str(exc))
    if not job_id:
        # Defensive: the controller should always tag the first event.
        status = await grpc(client.status())
        if status is not None and status.HasField("current_job"):
            return redirect(f"/jobs/{status.current_job.id}")
        return redirect_with_flash("/models", "controller did not report a job id")
    return redirect(f"/jobs/{job_id}")


@router.post("/models/load", name="models_load")
async def models_load(request: Request) -> Response:
    """Load a model in the background and follow its job (FR-01.3)."""
    form = await parse_form(request)
    alias = form.get("alias", "").strip()
    force = form.get("force") == "on"
    if not alias:
        return redirect_with_flash("/models", "pick a model to load")
    return await _start_job(request, alias, force, unload=False)


@router.post("/models/unload", name="models_unload")
async def models_unload(request: Request) -> Response:
    """Unload the active model in the background and follow its job (FR-01.3)."""
    return await _start_job(request, "", False, unload=True)
