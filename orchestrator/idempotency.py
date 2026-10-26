"""
Idempotency keys for worker attempts.

Every worker attempt gets a key derived from (run id, subtask slot, attempt
number, worker, normalized description). Before a worker runs, the
orchestrator checks the store: if that exact attempt already completed (for
example because a run was resumed, or a node was retried after a crash), the
stored result is reused instead of executing the worker a second time.

Today's workers are pure (retrieval, arithmetic, text generation), so reuse
mainly keeps replays cheap and traceable. It becomes essential once a worker
has side effects -- sending an email, writing to a database, calling a paid
API -- which is why workers declare a `side_effects` flag in the registry.

The store is in-memory by default; pass a SQLite path (or set IDEMPOTENCY_DB)
to make it survive process restarts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from orchestrator.schemas import WorkerResult


def make_idempotency_key(run_id: str, subtask_id: str, attempt: int, worker: str, description: str) -> str:
    normalized = " ".join(description.lower().split())
    payload = json.dumps([run_id, subtask_id, attempt, worker, normalized])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class IdempotencyStore:
    """Maps idempotency keys to completed WorkerResults."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self._lock = threading.Lock()
        self._memory: dict[str, WorkerResult] = {}
        self._conn: Optional[sqlite3.Connection] = None
        if path:
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS completed_attempts "
                "(key TEXT PRIMARY KEY, result TEXT NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> Optional[WorkerResult]:
        with self._lock:
            if key in self._memory:
                return self._memory[key]
            if self._conn is None:
                return None
            row = self._conn.execute(
                "SELECT result FROM completed_attempts WHERE key = ?", (key,)
            ).fetchone()
        return WorkerResult.model_validate_json(row[0]) if row else None

    def put(self, key: str, result: WorkerResult) -> None:
        with self._lock:
            self._memory[key] = result
            if self._conn is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO completed_attempts (key, result) VALUES (?, ?)",
                    (key, result.model_dump_json()),
                )
                self._conn.commit()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
