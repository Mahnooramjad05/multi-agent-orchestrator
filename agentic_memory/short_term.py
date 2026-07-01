"""
Short-term memory: in-process conversation buffer with a max turn window.

Keyed by session id. Each session keeps only the last N turns (a "turn"
being one (role, content) pair); older turns are dropped as new ones arrive.
This is intentionally simple in-memory state -- it does not persist across
process restarts (long-term/episodic memory below handle persistence).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str
    content: str


class ShortTermMemory:
    """Per-session rolling window of the most recent conversation turns."""

    def __init__(self, max_turns: int = 12) -> None:
        self._max_turns = max_turns
        self._sessions: dict[str, deque[Turn]] = {}
        self._lock = threading.Lock()

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            buf = self._sessions.setdefault(session_id, deque(maxlen=self._max_turns))
            buf.append(Turn(role=role, content=content))

    def get_turns(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id, deque()))

    def as_text(self, session_id: str) -> str:
        turns = self.get_turns(session_id)
        return "\n".join(f"{t.role}: {t.content}" for t in turns)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()
