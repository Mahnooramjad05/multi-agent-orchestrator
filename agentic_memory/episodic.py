"""
Episodic memory: structured log of past full orchestrator runs (goal, plan,
outcome, score), stored as JSON Lines on disk and retrievable by similarity
over the goal embedding.

Each call to `log_episode()` appends one JSON line to the log file. Retrieval
loads all episodes (fine for a demo-scale log) and ranks them by cosine
similarity between the query goal's embedding and each stored goal's
embedding, which is (re)computed at query time via the supplied embed_fn.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from orchestrator.schemas import EpisodicRecord

DEFAULT_LOG_PATH = Path("data") / "episodic_log.jsonl"


class EpisodicMemory:
    """JSONL-backed episodic memory log with similarity-based retrieval."""

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        log_path: Optional[Path] = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._log_path = log_path or DEFAULT_LOG_PATH
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_episode(self, record: EpisodicRecord) -> None:
        if not record.timestamp:
            record.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump()) + "\n")

    def all_episodes(self) -> list[EpisodicRecord]:
        if not self._log_path.exists():
            return []
        episodes: list[EpisodicRecord] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(EpisodicRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return episodes

    def similar_episodes(self, goal: str, top_k: int = 3) -> list[EpisodicRecord]:
        """Return up to top_k past episodes with the most similar goal text."""
        episodes = self.all_episodes()
        if not episodes:
            return []
        query_vec = np.array(self._embed_fn(goal), dtype=float)
        scored = []
        for ep in episodes:
            ep_vec = np.array(self._embed_fn(ep.goal), dtype=float)
            score = _cosine_similarity(query_vec, ep_vec)
            scored.append((score, ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def clear(self) -> None:
        if self._log_path.exists():
            os.remove(self._log_path)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
