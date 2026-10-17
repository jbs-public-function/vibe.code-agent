"""Job progress page (FR-01.4, NFR-01.5).

`GET /jobs/<id>` re-asks the controller for the job's event log on every load
(a fresh gRPC call per page render). While the job is the controller's current
job the page auto-refreshes every 2 s; once the last event is terminal the
refresh stops. Terminal ERROR pages also show the last 50 container log lines.

IMP-04.13: kind/alias come from the replayed `JobEvent`s (every event carries
them now) or from `GetJob` — the admin no longer keeps its own `known_jobs`
side table. IMP-04.14: the page keeps refreshing while the controller is down
(a restart may resume the job), and a non-terminal job that is no longer
current gets an "outcome unknown" note.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from vibe.admin.routes import (
    client_of,
    grpc,
    is_terminal,
    job_kind_name,
    render,
    safe_collect,
    safe_grpc,
)
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import state_name

router = APIRouter()


@router.get("/jobs/{job_id}", name="job")
async def job_page(request: Request, job_id: str) -> Response:
    """Render a job's kind, alias, elapsed time, state and event log (FR-01.4)."""
    client = client_of(request)
    status, status_error = await safe_grpc(client.status())
    controller_down = status is None
    try:
        events, events_error = await safe_collect(client.job_events(job_id))
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
            error=status_error,
        )
    error = status_error or events_error
    if events is None:
        controller_down = True
        events = []
    terminal = bool(events) and is_terminal(events[-1])
    is_current = bool(
        status is not None
        and status.HasField("current_job")
        and status.current_job.id == job_id
    )
    kind = control_pb2.JOB_KIND_UNSPECIFIED
    alias = ""
    started_s: float | None = None
    if is_current and status is not None:
        current = status.current_job
        kind = current.kind
        alias = current.alias
        started_s = float(current.started_at_unix) if current.started_at_unix else None
    # IMP-04.13: replayed events carry kind/alias (IMP-01.1); `GetJob` is the
    # fallback once the event buffer was purged.
    for event in events:
        if event.kind:
            kind = event.kind
            alias = event.alias
            break
    if kind == control_pb2.JOB_KIND_UNSPECIFIED:
        try:
            job = await grpc(client.get_job(job_id))
        except RuntimeError as exc:
            error = str(exc)
        except KeyError:
            job = None
        else:
            if job is not None:
                kind = job.kind
                alias = job.alias
                started_s = (
                    float(job.started_at_unix) if job.started_at_unix else None
                )
    if started_s is None and events:
        started_s = events[0].ts_unix_ms / 1000.0
    elapsed_s = time.time() - started_s if started_s is not None else None
    error_view = terminal and events and events[-1].state == control_pb2.ERROR
    log_lines: list[str] = []
    if error_view:
        lines = await safe_collect(client.logs(tail=50))
        if lines[0] is not None:
            log_lines = [line.line for line in lines[0]]
    # IMP-04.14: refresh while current *or* while the controller is down (the
    # job may have continued underneath); stop only at a terminal event.
    refresh = (is_current or controller_down) and not terminal
    outcome_unknown = (not controller_down) and (not terminal) and (not is_current)
    return render(
        request,
        "job.html",
        {
            "page": "jobs",
            "controller_down": controller_down,
            "active_alias": status.active_alias if status is not None else "",
            "not_found": False,
            "job_id": job_id,
            "kind": job_kind_name(kind),
            "alias": alias,
            "elapsed_s": elapsed_s,
            "latest_state": state_name(events[-1].state) if events else "UNKNOWN",
            "events": events,
            "terminal": terminal,
            "refresh": refresh,
            "outcome_unknown": outcome_unknown,
            "error_view": error_view,
            "log_lines": log_lines,
        },
        error=error,
    )
