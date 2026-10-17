"""Controller-side Jev logic — re-export shim (doc 02 names this module).

The implementation lives in `vibe.jev.control` so `vibe.jobs` can import it
without triggering the controller package `__init__` (which imports the
servicer and would create a circular import). Doc references to
`vibe/controller/jev.py` resolve here.
"""

from __future__ import annotations

from vibe.jev.control import (
    PROBE_MAX_TOKENS,
    PROBE_PROMPT,
    PROBE_TIMEOUT_S,
    ActiveJev,
    JevLoadSpec,
    JevState,
    ResolvedJev,
    probe_jev,
    resolve_jev,
)

__all__ = [
    "PROBE_MAX_TOKENS",
    "PROBE_PROMPT",
    "PROBE_TIMEOUT_S",
    "ActiveJev",
    "JevLoadSpec",
    "JevState",
    "ResolvedJev",
    "probe_jev",
    "resolve_jev",
]
