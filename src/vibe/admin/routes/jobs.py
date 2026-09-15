"""Job progress page (FR-01.4, NFR-01.5).

`GET /jobs/<id>` re-asks the controller for the job's event log on every load
(a fresh gRPC call per page render). While the job is the controller's current
job the page auto-refreshes every 2 s; once the last event is terminal the
refresh stops. Terminal ERROR pages also show the last 50 container log lines.
"""

from __future__ import annotations

import time
from typing import cast

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from vibe.admin.routes import (
    client_of,
    collect,
    grpc,
    is_terminal,
    job_kind_name,
    render,
)
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import state_name

router = APIRouter()


@router.get("/jobs/{job_id}", name="job")
async def job_page(request: Request, job_id: str) -> Response:
    """Render a job's kind, alias, elapsed time, state and event log (FR-01.4)."""
    client = client_of(request)
    status = await grpc(client.status())
    controller_down = status is None
    try:
        events = await collect(client.job_events(job_id))
    except KeyError:
        # Unknown job id: a dedicated page, no refresh (FR-01.4).
        return render(
            request,
            "job.html",
            {
                "page": "jobs",
                "controller_down": controller_down,
                "active_alias": status.active_alias if status is not None else "",
                "not_found": True,
                "job_id": job_id,
            },
        )
    if events is None:
        controller_down = True
        events = []
    terminal = bool(events) and is_terminal(events[-1])
    is_current = bool(
        status is not None
        and status.HasField("current_job")
        and status.current_job.id == job_id
    )
    if is_current:
        current = status.current_job if status is not None else control_pb2.Job()
        kind = job_kind_name(current.kind)
        alias = current.alias
        started_s = float(current.started_at_unix) if current.started_at_unix else None
    else:
        known = cast(dict[str, tuple[str, str]], request.app.state.known_jobs)
        kind, alias = known.get(job_id, ("—", ""))
        started_s = None
    if started_s is None and events:
        started_s = events[0].ts_unix_ms / 1000.0
    elapsed_s = time.time() - started_s if started_s is not None else None
    error_view = terminal and events and events[-1].state == control_pb2.ERROR
    log_lines: list[str] = []
    if error_view:
        lines = await collect(client.logs(tail=50))
        if lines is not None:
            log_lines = [line.line for line in lines]
    return render(
        request,
        "job.html",
        {
            "page": "jobs",
            "controller_down": controller_down,
            "active_alias": status.active_alias if status is not None else "",
            "not_found": False,
            "job_id": job_id,
            "kind": kind,
            "alias": alias,
            "elapsed_s": elapsed_s,
            "latest_state": state_name(events[-1].state) if events else "UNKNOWN",
            "events": events,
            "terminal": terminal,
            "refresh": is_current and not terminal,
            "error_view": error_view,
            "log_lines": log_lines,
        },
    )
