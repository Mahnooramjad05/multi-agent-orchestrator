"""
In-process semantic cache.

Caches (prompt embedding -> response) pairs in memory and short-circuits a
new LLM call when an incoming prompt is cosine-similar enough to a
previously seen prompt (same or paraphrased request). Backed by plain numpy
arrays -- no Redis, no external service.

This is wired into the LLM-call wrapper in orchestrator.llm_call so that
repeated/similar prompts across the graph (and across a whole eval run)
skip the underlying LLM call entirely.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class CacheEntry:
    prompt: str
    embedding: np.ndarray
    response: str
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    """
    Small in-process semantic cache keyed by embedding cosine similarity.

    Not thread-safety-critical for this project (single-process demo/API),
    but guarded with a lock since FastAPI can serve concurrent requests.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        similarity_threshold: float = 0.97,
        max_entries: int = 500,
    ) -> None:
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0

    def get(self, prompt: str) -> Optional[str]:
        """Return a cached response if a sufficiently similar prompt exists."""
        if not prompt:
            return None
        query_vec = np.array(self._embed_fn(prompt), dtype=float)
        with self._lock:
            best_entry: Optional[CacheEntry] = None
            best_score = -1.0
            for entry in self._entries:
                score = _cosine_similarity(query_vec, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry
            if best_entry is not None and best_score >= self._threshold:
                best_entry.hits += 1
                self.hit_count += 1
                return best_entry.response
        self.miss_count += 1
        return None

    def put(self, prompt: str, response: str) -> None:
        """Store a new (prompt, response) pair, embedding the prompt now."""
        if not prompt:
            return
        embedding = np.array(self._embed_fn(prompt), dtype=float)
        with self._lock:
            self._entries.append(CacheEntry(prompt=prompt, embedding=embedding, response=response))
            if len(self._entries) > self._max_entries:
                # Evict oldest entries first (simple FIFO eviction).
                self._entries = self._entries[-self._max_entries :]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self.hit_count = 0
        self.miss_count = 0

    def stats(self) -> dict:
        with self._lock:
            size = len(self._entries)
        return {
            "size": size,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "threshold": self._threshold,
        }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
