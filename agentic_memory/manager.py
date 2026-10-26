"""
MemoryManager: single entry point wiring together short-term, long-term
(Qdrant-backed), episodic memory, and the semantic cache.

Also hosts `cached_llm_call`, the LLM-call wrapper that all orchestrator
nodes should use for text completions so that repeated/similar prompts skip
the underlying LLM call via the semantic cache.
"""

from __future__ import annotations

import os
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from agentic_memory.episodic import EpisodicMemory
from agentic_memory.long_term import LongTermMemory
from agentic_memory.semantic_cache import SemanticCache
from agentic_memory.short_term import ShortTermMemory
from orchestrator.idempotency import IdempotencyStore
from orchestrator.llm_client import LLMClient, get_llm_client

T = TypeVar("T", bound=BaseModel)


class MemoryManager:
    """Facade over all memory components, constructed once per process/app."""

    def __init__(self, llm_client: Optional[LLMClient] = None, short_term_window: int = 12) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.short_term = ShortTermMemory(max_turns=short_term_window)
        self.long_term = LongTermMemory(embed_fn=self.llm_client.embed)
        self.episodic = EpisodicMemory(embed_fn=self.llm_client.embed)
        self.semantic_cache = SemanticCache(embed_fn=self.llm_client.embed)
        # Completed worker attempts by idempotency key (in-memory unless IDEMPOTENCY_DB is set).
        self.idempotency = IdempotencyStore(os.getenv("IDEMPOTENCY_DB") or None)

        # Simple counters used by the eval harness for efficiency scoring.
        self.llm_call_count = 0
        self.cache_hit_count = 0
        # Token usage across LLM calls (cache hits cost nothing). Budgets read these.
        self.input_tokens = 0
        self.output_tokens = 0
        self.tokens_estimated = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def reset_counters(self) -> None:
        self.llm_call_count = 0
        self.cache_hit_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tokens_estimated = False

    def _record_usage(self) -> None:
        self.llm_call_count += 1
        usage = self.llm_client.last_usage
        if usage is None:
            return
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.tokens_estimated = self.tokens_estimated or usage.estimated

    def cached_invoke_text(self, system: str, prompt: str) -> str:
        """Text completion, short-circuited by the semantic cache when possible."""
        cache_key = f"{system}\n---\n{prompt}"
        cached = self.semantic_cache.get(cache_key)
        if cached is not None:
            self.cache_hit_count += 1
            return cached
        response = self.llm_client.invoke_text(system, prompt)
        self._record_usage()
        self.semantic_cache.put(cache_key, response)
        return response

    def cached_invoke_structured(self, system: str, prompt: str, schema: Type[T]) -> T:
        """
        Structured completion. Structured calls are NOT served from the
        semantic cache (each caller needs a fresh validated object bound to
        a specific schema type), but they still count toward llm_call_count
        for efficiency scoring, and go through the same client.
        """
        result = self.llm_client.invoke_structured(system, prompt, schema)
        self._record_usage()
        return result
