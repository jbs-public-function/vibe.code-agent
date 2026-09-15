"""In-memory chat session store keyed by a browser cookie (FR-01.5).

Sessions are plain dicts of `{role, content}` turns, keyed by a uuid4-hex id
sent as the `vibe_session` cookie. Entries older than 24 h are swept lazily on
access so the store cannot grow without bound.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

SESSION_COOKIE = "vibe_session"

# FR-01.5: a chat session lives for 24 h. An int so Set-Cookie `Max-Age`
# serializes as a plain integer.
SESSION_TTL_SECONDS = 24 * 60 * 60


@dataclass
class ChatSession:
    """One browser's conversation; turns are stored as plain dicts."""

    session_id: str
    created_at: float
    messages: list[dict[str, str]] = field(default_factory=list)

    def token_estimate(self) -> int:
        """Rough token count (~4 chars/token), for the context-length warning."""
        return sum(len(message["content"]) // 4 for message in self.messages)


class SessionStore:
    """Cookie-keyed sessions with a TTL sweep run on every access (FR-01.5)."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, session_id: str | None) -> ChatSession:
        """Return the session for `session_id`, or mint a fresh one."""
        self._sweep()
        if session_id is not None:
            session = self._sessions.get(session_id)
            if session is not None:
                return session
        new_id = uuid.uuid4().hex
        session = ChatSession(new_id, time.time())
        self._sessions[new_id] = session
        return session

    def reset(self, session_id: str) -> None:
        """Drop a session entirely (FR-01.5 `POST /chat/reset`)."""
        self._sessions.pop(session_id, None)

    def _sweep(self) -> None:
        """Remove sessions older than 24 h."""
        deadline = time.time() - SESSION_TTL_SECONDS
        for stale in [
            session_id
            for session_id, session in self._sessions.items()
            if session.created_at < deadline
        ]:
            del self._sessions[stale]
