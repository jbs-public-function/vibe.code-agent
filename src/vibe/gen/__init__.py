"""Generated gRPC code (ADR-5) — committed intentionally.

`control_pb2_grpc.py` imports its pb2 module as `vibe.v1.control_pb2` (the
proto package path), but the committed layout puts the generated modules under
`vibe.gen.vibe.v1`. Importing `vibe.gen` installs a `sys.modules` alias so
`vibe.v1` resolves to `vibe.gen.vibe.v1`, keeping both import paths bound to
the same module instances regardless of order.
"""

from __future__ import annotations

import importlib
import sys

_v1 = importlib.import_module("vibe.gen.vibe.v1")
sys.modules.setdefault("vibe.v1", _v1)
sys.modules.setdefault(
    "vibe.v1.control_pb2",
    importlib.import_module("vibe.gen.vibe.v1.control_pb2"),
)
