"""vibe controller — the gRPC control plane (Phase 1).

`ModelControllerServicer` wraps the Phase-0 `JobRunner`, `docker_ctl`, and
`gpu` modules behind `vibe.v1.ModelController`; `build_server`/`run`/`main`
compose the `grpc.aio` server and its health service; `RingBuffer` holds
recent job events for `StreamLogs` replay.
"""

from .logbuffer import RingBuffer
from .server import build_server, main, run
from .servicer import ModelControllerServicer

__all__ = [
    "ModelControllerServicer",
    "RingBuffer",
    "build_server",
    "main",
    "run",
]
