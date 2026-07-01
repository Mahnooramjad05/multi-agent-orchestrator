"""The semantic cache must actually short-circuit a repeated/similar call."""

from __future__ import annotations

from agentic_memory.semantic_cache import SemanticCache
from orchestrator.llm_client import deterministic_embedding


def _embed(text: str) -> list[float]:
    return deterministic_embedding(text, dim=32)


def test_cache_miss_then_hit_on_identical_prompt():
    cache = SemanticCache(embed_fn=_embed, similarity_threshold=0.97)

    assert cache.get("What is the capital of France?") is None
    cache.put("What is the capital of France?", "Paris is the capital of France.")

    cached = cache.get("What is the capital of France?")
    assert cached == "Paris is the capital of France."
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_does_not_match_unrelated_prompt():
    cache = SemanticCache(embed_fn=_embed, similarity_threshold=0.97)
    cache.put("What is the capital of France?", "Paris.")

    result = cache.get("Explain quantum entanglement in physics.")
    assert result is None


def test_memory_manager_cached_invoke_text_short_circuits_llm_call(memory_manager):
    first = memory_manager.cached_invoke_text("system prompt", "Summarize the internet's history.")
    calls_after_first = memory_manager.llm_call_count
    assert calls_after_first == 1

    second = memory_manager.cached_invoke_text("system prompt", "Summarize the internet's history.")
    calls_after_second = memory_manager.llm_call_count

    # No new LLM call should have been made for the repeated prompt.
    assert calls_after_second == calls_after_first
    assert second == first
    assert memory_manager.cache_hit_count == 1


def test_cache_clear_resets_state():
    cache = SemanticCache(embed_fn=_embed)
    cache.put("hello", "world")
    cache.clear()
    assert cache.stats()["size"] == 0
    assert cache.get("hello") is None
