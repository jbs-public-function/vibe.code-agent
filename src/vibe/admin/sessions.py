"""In-memory chat session store keyed by a browser cookie (FR-01.5).

Sessions are plain dicts of `{role, content}` turns, keyed by a uuid4-hex id
sent as the `vibe_session` cookie. The store is bounded (IMP-04.9):

- **TTL sweep**: sessions idle for more than 24 h are dropped lazily on
  access, measured from `last_seen` (not `created_at`) so a conversation kept
  alive by use never expires mid-flight.
- **LRU cap**: at most `MAX_SESSIONS` sessions; the least recently used one
  is evicted (every cookie-less `GET /chat` mints one, so an unbounded store
  would leak).
- **Turn cap**: at most `MAX_TURNS` turns per session; the oldest non-system
  turn is dropped first so chat templates never see consecutive `user` turns.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

SESSION_COOKIE = "vibe_session"

# FR-01.5: a chat session lives for 24 h. An int so Set-Cookie `Max-Age`
# serializes as a plain integer.
SESSION_TTL_SECONDS = 24 * 60 * 60

# IMP-04.9: store bounds.
MAX_SESSIONS = 256
MAX_TURNS = 200


@dataclass
class ChatSession:
    """One browser's conversation; turns are stored as plain dicts."""

    session_id: str
    created_at: float
    last_seen: float
    messages: list[dict[str, str]] = field(default_factory=list)

    def token_estimate(self) -> int:
        """Rough token count (~4 chars/token), for the context-length warning."""
        return sum(len(message["content"]) // 4 for message in self.messages)

    def append(self, message: dict[str, str]) -> None:
        """Append a turn, evicting the oldest non-system turn past the cap."""
        self.messages.append(message)
        while len(self.messages) > MAX_TURNS:
            for index, turn in enumerate(self.messages):
                if turn.get("role") != "system":
                    del self.messages[index]
                    break
            else:
                break  # all-system history: nothing safe to drop


class SessionStore:
    """Cookie-keyed, TTL/LRU/turn-bounded sessions (FR-01.5, IMP-04.9)."""

    def __init__(self) -> None:
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()

    def get_or_create(self, session_id: str | None) -> ChatSession:
        """Return the session for `session_id`, or mint a fresh one."""
        self._sweep()
        if session_id is not None:
            session = self._sessions.get(session_id)
            if session is not None:
                self.touch(session)
                return session
        new_id = uuid.uuid4().hex
        now = time.time()
        session = ChatSession(new_id, now, now)
        self._sessions[new_id] = session
        self._evict_lru()
        return session

    def touch(self, session: ChatSession) -> None:
        """Refresh a session's TTL and LRU position on use."""
        session.last_seen = time.time()
        self._sessions.move_to_end(session.session_id)

    def reset(self, session_id: str) -> None:
        """Drop a session entirely (FR-01.5 `POST /chat/reset`)."""
        self._sessions.pop(session_id, None)

    def _evict_lru(self) -> None:
        """Drop the least recently used session once the cap is exceeded."""
        while len(self._sessions) > MAX_SESSIONS:
            self._sessions.popitem(last=False)

    def _sweep(self) -> None:
        """Remove sessions idle for more than 24 h (TTL from `last_seen`)."""
        deadline = time.time() - SESSION_TTL_SECONDS
        for stale in [
            session_id
            for session_id, session in self._sessions.items()
            if session.last_seen < deadline
        ]:
            del self._sessions[stale]
