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


def test_writer_sees_results_from_the_same_pass(monkeypatch, memory_manager: MemoryManager):
    """Regression test: the writer used to get the context built before the pass started,
    so on the first pass it never saw the research result and always scored 0.6."""
    from orchestrator import workers
    from orchestrator.schemas import WorkerResult

    seen_contexts: list[str] = []

    def research(subtask, memory):
        return WorkerResult(output="Brazil is the largest coffee producer.", confidence=0.9)

    def writer(subtask, memory, context=""):
        seen_contexts.append(context)
        return workers.writer_worker(subtask, memory, context)

    monkeypatch.setitem(workers.WORKER_REGISTRY, "research", research)
    monkeypatch.setitem(workers.WORKER_REGISTRY, "writer", writer)

    state = run_goal("Find facts about coffee.", memory_manager, log_episode=False)

    assert len(seen_contexts) == 1
    assert "Brazil is the largest coffee producer." in seen_contexts[0]
    writer_attempt = next(a for a in state["attempts"] if a.worker == "writer")
    assert writer_attempt.raw_confidence == 0.9  # 0.6 means it had no upstream context
