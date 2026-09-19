"""Dashboard and health routes (FR-01.1, FR-01.9, NFR-01.5).

`GET /` renders the active-model snapshot, GPU stats, controller/gateway
health and the current job link, auto-refreshing every 5 s. `GET /health` is
the JSON liveness probe with a 3 s deadline on the controller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import cast

import httpx
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from vibe.admin.routes import GRPC_TIMEOUT, client_of, grpc, render, safe_grpc
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import state_name

router = APIRouter()

logger = logging.getLogger("vibe.admin")

# IMP-04.24: the probe must be cheap — the dashboard refreshes every 5 s.
PROBE_TIMEOUT_S = 0.5


async def _gateway_ok(request: Request) -> bool:
    """Probe the Phase-2 gateway's /health over plain HTTP (IMP-04.24).

    The URL comes from `app.state.gateway_health_url` (wildcard binds already
    mapped to loopback, IPv6 bracketed); tests inject `app.state.gateway_transport`
    so the probe never dials the real gateway. Any exception counts as down —
    `httpx.InvalidURL` is a `ValueError`, not an `HTTPError`.
    """
    url = cast(str, request.app.state.gateway_health_url)
    transport = getattr(request.app.state, "gateway_transport", None)
    try:
        if transport is None:
            client = cast(httpx.AsyncClient, request.app.state.http)
            response = await client.get(url)
        else:
            async with httpx.AsyncClient(
                transport=transport, timeout=PROBE_TIMEOUT_S
            ) as probe:
                response = await probe.get(url)
    except Exception as exc:
        logger.warning(
            "gateway probe failed", extra={"url": url, "error": str(exc)}
        )
        return False
    return response.status_code == 200


@router.get("/", name="dashboard")
async def dashboard(request: Request) -> Response:
    """Dashboard: status, GPU, health and the current job (FR-01.1)."""
    client = client_of(request)
    (status_result, status_error), (gpu_result, gpu_error), gateway_ok = (
        await asyncio.gather(
            safe_grpc(client.status()), safe_grpc(client.gpu()), _gateway_ok(request)
        )
    )
    controller_down = status_result is None or gpu_result is None
    status = status_result if status_result is not None else control_pb2.Status()
    gpu = gpu_result if gpu_result is not None else control_pb2.GpuStats()
    if gpu.total_mib:
        mem_pct = int(gpu.used_mib * 100 / gpu.total_mib)
    else:
        mem_pct = 0
    uptime_s: float | None = None
    if status.started_at_unix:
        uptime_s = time.time() - status.started_at_unix
    context = {
        "page": "dashboard",
        "controller_down": controller_down,
        "active_alias": status.active_alias,
        "state": "UNKNOWN" if controller_down else state_name(status.state),
        "container_id": status.container_id[:12] or "—",
        "host_port": status.host_port,
        "uptime_s": uptime_s,
        "current_job": status.current_job,
        "last_error": status.last_error,
        "gpu": gpu,
        "util_pct": gpu.utilization_pct,
        "mem_pct": mem_pct,
        # CSS class stepping (5 %) — inline `style="width"` would violate
        # the `style-src 'self'` CSP (NFR-01.1).
        "bar_class": f"bar-{min(100, mem_pct - mem_pct % 5)}",
        "gateway_ok": gateway_ok,
    }
    return render(request, "dashboard.html", context, error=status_error or gpu_error)


@router.get("/health", name="health")
async def health(request: Request) -> JSONResponse:
    """Liveness probe; controller status under the 3 s deadline (FR-01.9)."""
    client = client_of(request)
    error: str | None = None
    try:
        await asyncio.wait_for(client.status(), timeout=GRPC_TIMEOUT)
        controller = "ok"
    except (asyncio.TimeoutError, ControllerUnavailable):
        controller = "down"
    except RuntimeError as exc:
        # IMP-04.15: a real error is reported, not swallowed as "down".
        controller = "down"
        error = str(exc)
    payload: dict[str, str] = {"admin": "ok", "controller": controller}
    if error is not None:
        payload["error"] = error
    return JSONResponse(payload)
