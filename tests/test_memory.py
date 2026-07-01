"""Unit tests for the individual agentic memory components."""

from __future__ import annotations

from pathlib import Path

from agentic_memory.episodic import EpisodicMemory
from agentic_memory.short_term import ShortTermMemory
from orchestrator.llm_client import deterministic_embedding
from orchestrator.schemas import EpisodicRecord


def _embed(text: str) -> list[float]:
    return deterministic_embedding(text, dim=32)


def test_short_term_memory_respects_max_window():
    stm = ShortTermMemory(max_turns=3)
    for i in range(5):
        stm.add_turn("s1", "user", f"message {i}")
    turns = stm.get_turns("s1")
    assert len(turns) == 3
    assert turns[0].content == "message 2"
    assert turns[-1].content == "message 4"


def test_short_term_memory_is_scoped_per_session():
    stm = ShortTermMemory(max_turns=5)
    stm.add_turn("s1", "user", "hello from s1")
    stm.add_turn("s2", "user", "hello from s2")
    assert len(stm.get_turns("s1")) == 1
    assert len(stm.get_turns("s2")) == 1
    assert stm.get_turns("s1")[0].content != stm.get_turns("s2")[0].content


def test_episodic_memory_logs_and_retrieves(tmp_path: Path):
    log_path = tmp_path / "episodes.jsonl"
    episodic = EpisodicMemory(embed_fn=_embed, log_path=log_path)

    episodic.log_episode(
        EpisodicRecord(goal="Research coffee production", plan=["research"], outcome="Brazil leads.", score=1.0)
    )
    episodic.log_episode(
        EpisodicRecord(goal="Calculate 2 + 2", plan=["calculator"], outcome="4", score=1.0)
    )

    all_eps = episodic.all_episodes()
    assert len(all_eps) == 2

    similar = episodic.similar_episodes("Which country grows the most coffee?", top_k=1)
    assert len(similar) == 1
    assert "coffee" in similar[0].goal.lower()


def test_long_term_memory_add_and_search(memory_manager):
    memory_manager.long_term.add_fact("s1", "Acme Robotics is headquartered in Austin, Texas.")
    memory_manager.long_term.add_fact("s1", "The coffee market is dominated by Brazil.")

    results = memory_manager.long_term.search("s1", "Where is Acme Robotics located?", top_k=1)
    assert len(results) == 1
    assert "Austin" in results[0].text


def test_long_term_memory_is_scoped_per_session(memory_manager):
    memory_manager.long_term.add_fact("session-x", "Fact only visible in session-x.")
    results = memory_manager.long_term.search("session-y", "Fact only visible in session-x.", top_k=5)
    assert results == []
