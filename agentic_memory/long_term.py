"""
Long-term memory: embedded facts/summaries persisted to a Qdrant vector store,
keyed (filtered) by session id.

Uses qdrant-client's local/in-memory mode (location=":memory:") by default so
the whole project runs without any external Qdrant server -- ideal for local
development and evaluation. To go to production, point QDRANT_URL at a real
Qdrant deployment (see .env.example) and this module will connect to it
instead, with no other code changes.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

COLLECTION_NAME = "long_term_memory"


@dataclass
class MemoryFact:
    session_id: str
    text: str
    score: float = 0.0
    created_at: float = 0.0


class LongTermMemory:
    """
    Stores short textual facts/summaries in Qdrant, embedded via the
    provided `embed_fn`, and retrieves the most semantically similar facts
    for a given session.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        embedding_dim: int = 64,
        url: Optional[str] = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._embedding_dim = embedding_dim

        qdrant_url = url if url is not None else os.getenv("QDRANT_URL", "").strip()
        if qdrant_url:
            # Production / real-server mode.
            self._client = QdrantClient(url=qdrant_url)
        else:
            # Default: local in-memory mode -- no external server required.
            self._client = QdrantClient(location=":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=self._embedding_dim, distance=qmodels.Distance.COSINE
                ),
            )

    def add_fact(self, session_id: str, text: str) -> str:
        """Embed and store a fact/summary for a session. Returns the point id."""
        point_id = str(uuid.uuid4())
        vector = self._embed_fn(text)
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "text": text,
                        "created_at": time.time(),
                    },
                )
            ],
        )
        return point_id

    def search(self, session_id: str, query: str, top_k: int = 3) -> list[MemoryFact]:
        """Return the top_k facts for `session_id` most similar to `query`."""
        vector = self._embed_fn(query)
        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))]
            ),
            limit=top_k,
        ).points
        return [
            MemoryFact(
                session_id=session_id,
                text=point.payload.get("text", "") if point.payload else "",
                score=point.score,
                created_at=(point.payload or {}).get("created_at", 0.0),
            )
            for point in results
        ]

    def clear_session(self, session_id: str) -> None:
        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="session_id", match=qmodels.MatchValue(value=session_id)
                        )
                    ]
                )
            ),
        )
