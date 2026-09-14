"""Bounded in-memory event buffer for job log replay (ADR-10).

The controller records recent `jobs.JobEvent`s so `StreamLogs` has something
to show when no container is running (FR-02.5). All access happens on the
single asyncio event loop, so no locking is needed.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class RingBuffer:
    """Fixed-size buffer of the most recent items (newest at the end)."""

    def __init__(self, maxlen: int = 2000) -> None:
        self._items: deque[Any] = deque(maxlen=maxlen)

    def append(self, item: Any) -> None:
        """Append an item, silently evicting the oldest one when full."""
        self._items.append(item)

    def recent(self, n: int) -> list[Any]:
        """Return at most `n` items, newest first."""
        if n <= 0:
            return []
        return list(reversed(self._items))[:n]

    def clear(self) -> None:
        """Drop everything buffered so far."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
