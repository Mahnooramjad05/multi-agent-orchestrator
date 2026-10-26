"""Token and wall-clock budgets (mock LLM with fixed token usage, no network calls)."""

from __future__ import annotations

import pytest

from agentic_memory.manager import MemoryManager
from orchestrator import workers
from orchestrator.budget import Budget, BudgetTracker
from orchestrator.graph import run_goal
from orchestrator.llm_client import MockLLMClient, Usage
from orchestrator.schemas import WorkerResult


class FixedUsageClient(MockLLMClient):
    """Mock client where every call costs exactly `per_call` tokens (input + output)."""

    def __init__(self, per_call: int = 100) -> None:
        self.per_call = per_call

    def invoke_text(self, system, prompt):
        text = super().invoke_text(system, prompt)
        self.last_usage = Usage(input_tokens=self.per_call // 2, output_tokens=self.per_call // 2)
        return text

    def invoke_structured(self, system, prompt, schema):
        result = super().invoke_structured(system, prompt, schema)
        self.last_usage = Usage(input_tokens=self.per_call // 2, output_tokens=self.per_call // 2)
        return result


def _always_low(monkeypatch):
    """Research worker that calls the LLM (costing tokens) but is never confident."""
    def worker(subtask, memory, context=""):
        memory.cached_invoke_text("sys", f"research {subtask.description} {subtask.id}")
        return WorkerResult(output="unsure", confidence=0.1)
    monkeypatch.setitem(workers.WORKER_REGISTRY, "research", worker)


def test_budget_validation_and_env(monkeypatch):
    with pytest.raises(ValueError):
        Budget(max_tokens=0)
    with pytest.raises(ValueError):
        Budget(synthesis_reserve=1.0)
    monkeypatch.setenv("BUDGET_MAX_TOKENS", "5000")
    monkeypatch.setenv("BUDGET_MAX_SECONDS", "")
    budget = Budget.from_env()
    assert budget.max_tokens == 5000 and budget.max_seconds is None


def test_tracker_reserve_math():
    memory = MemoryManager(llm_client=FixedUsageClient(100))
    tracker = BudgetTracker(Budget(max_tokens=1000, synthesis_reserve=0.2), memory)
    memory.input_tokens += 700
    assert tracker.exhausted() is None              # 700 < 800 work limit
    assert tracker.can_afford(100, 0) is None
    assert tracker.can_afford(101, 0) == "token_budget"
    memory.output_tokens += 100
    assert tracker.exhausted() == "token_budget"     # 800 >= 800


def test_token_counts_are_recorded():
    memory = MemoryManager(llm_client=FixedUsageClient(100))
    state = run_goal("Tell me about solar panels.", memory, log_episode=False)
    assert memory.total_tokens == memory.llm_call_count * 100
    assert state["usage"]["input_tokens"] + state["usage"]["output_tokens"] == memory.total_tokens
    assert sum(a.input_tokens + a.output_tokens for a in state["attempts"]) > 0


def test_mock_client_estimates_tokens(memory_manager: MemoryManager):
    run_goal("Tell me about solar panels.", memory_manager, log_episode=False)
    assert memory_manager.total_tokens > 0
    assert memory_manager.tokens_estimated is True


def test_token_budget_stops_replanning_but_still_synthesizes(monkeypatch):
    _always_low(monkeypatch)
    memory = MemoryManager(llm_client=FixedUsageClient(100))
    # decompose (100) + research (100) uses 200 of a 340-token work limit (400 minus 15% reserve);
    # a replan is projected at ~200 more, which doesn't fit.
    state = run_goal("Find facts about coffee.", memory, budget=Budget(max_tokens=400), log_episode=False)

    assert state["stop_reason"] == "token_budget"
    assert state["replans"] == 0
    assert state["final_answer"]  # the synthesis reserve guarantees an answer
    assert any(e["event"] == "stop" and e["reason"] == "token_budget" for e in state["trace"])


def test_unlimited_budget_runs_to_max_replans(monkeypatch):
    _always_low(monkeypatch)
    memory = MemoryManager(llm_client=FixedUsageClient(100))
    state = run_goal("Find facts about coffee.", memory, budget=Budget(), log_episode=False)
    assert state["stop_reason"] == "max_replans"


def test_exhausted_budget_skips_remaining_attempts(monkeypatch):
    memory = MemoryManager(llm_client=FixedUsageClient(100))
    # The decomposition alone (100 tokens) uses up the 85-token work limit.
    state = run_goal("Find facts about coffee.", memory, budget=Budget(max_tokens=100), log_episode=False)
    assert state["stop_reason"] == "token_budget"
    assert {a.status for a in state["attempts"]} == {"skipped_budget"}
    assert state["final_answer"]


def test_time_budget(monkeypatch):
    _always_low(monkeypatch)
    memory = MemoryManager(llm_client=FixedUsageClient(10))
    clock = iter(range(0, 10_000, 30))  # every monotonic() call advances 30 seconds
    monkeypatch.setattr("orchestrator.budget._clock", lambda: float(next(clock)))
    state = run_goal("Find facts about coffee.", memory, budget=Budget(max_seconds=60), log_episode=False)
    assert state["stop_reason"] == "time_budget"
    assert state["final_answer"]
