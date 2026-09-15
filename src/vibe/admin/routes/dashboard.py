"""Dashboard and health routes (FR-01.1, FR-01.9, NFR-01.5).

`GET /` renders the active-model snapshot, GPU stats, controller/gateway
health and the current job link, auto-refreshing every 5 s. `GET /health` is
the JSON liveness probe with a 3 s deadline on the controller.
"""

from __future__ import annotations

import asyncio
import time
from typing import cast

import httpx
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from vibe.admin.routes import GRPC_TIMEOUT, client_of, grpc, render
from vibe.client import ControllerUnavailable
from vibe.gen.vibe.v1 import control_pb2
from vibe.gateway.models import state_name

router = APIRouter()


async def _gateway_ok(request: Request) -> bool:
    """Probe the Phase-2 gateway's /health over plain HTTP."""
    http = cast(httpx.AsyncClient, request.app.state.http)
    url = cast(str, request.app.state.gateway_health_url)
    try:
        response = await http.get(url)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


@router.get("/", name="dashboard")
async def dashboard(request: Request) -> Response:
    """Dashboard: status, GPU, health and the current job (FR-01.1)."""
    client = client_of(request)
    status_result, gpu_result, gateway_ok = await asyncio.gather(
        grpc(client.status()), grpc(client.gpu()), _gateway_ok(request)
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
    return render(request, "dashboard.html", context)


@router.get("/health", name="health")
async def health(request: Request) -> JSONResponse:
    """Liveness probe; controller status under the 3 s deadline (FR-01.9)."""
    client = client_of(request)
    try:
        await asyncio.wait_for(client.status(), timeout=GRPC_TIMEOUT)
        controller = "ok"
    except (asyncio.TimeoutError, ControllerUnavailable, RuntimeError):
        controller = "down"
    return JSONResponse({"admin": "ok", "controller": controller})
