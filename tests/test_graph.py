"""End-to-end orchestrator graph tests using the mock LLM (no network calls)."""

from __future__ import annotations

from agentic_memory.manager import MemoryManager
from orchestrator.graph import run_goal


def test_run_goal_produces_final_answer_and_trace(memory_manager: MemoryManager):
    final_state = run_goal("What is the capital of France?", memory_manager, session_id="test-session")

    assert final_state["final_answer"]
    assert len(final_state["executions"]) >= 1
    assert final_state["replans"] >= 0
    assert any(event["event"] == "decompose" for event in final_state["trace"])
    assert any(event["event"] == "synthesize" for event in final_state["trace"])


def test_run_goal_populates_short_term_memory(memory_manager: MemoryManager):
    run_goal("Tell me about solar panels.", memory_manager, session_id="session-a")
    turns = memory_manager.short_term.get_turns("session-a")
    assert len(turns) == 2  # one user turn, one assistant turn
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"


def test_run_goal_writes_long_term_fact(memory_manager: MemoryManager):
    run_goal("Summarize machine learning basics.", memory_manager, session_id="session-b")
    facts = memory_manager.long_term.search("session-b", "machine learning", top_k=3)
    assert len(facts) >= 1
