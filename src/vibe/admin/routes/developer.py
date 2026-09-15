"""Developer placeholder page (FR-01.8).

Downloads and guided registration land here in Phase 4; for now the page just
links to the Phase 4 spec.
"""

from __future__ import annotations

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from vibe.admin.routes import client_of, grpc, render

router = APIRouter()


@router.get("/developer", name="developer")
async def developer(request: Request) -> Response:
    """Phase 3 placeholder linking to the Phase 4 spec (FR-01.8)."""
    client = client_of(request)
    status = await grpc(client.status())
    return render(
        request,
        "developer.html",
        {
            "page": "developer",
            "controller_down": status is None,
            "active_alias": status.active_alias if status is not None else "",
        },
    )
