"""Per-attempt logging and idempotency keys (mock LLM, no network calls)."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_memory.episodic import EpisodicMemory
from agentic_memory.manager import MemoryManager
from orchestrator import workers
from orchestrator.graph import MAX_REPLANS, run_goal
from orchestrator.idempotency import IdempotencyStore, make_idempotency_key
from orchestrator.llm_client import deterministic_embedding
from orchestrator.schemas import WorkerResult


def _counting_worker(confidence: float, calls: list):
    def worker(subtask, memory, context=""):
        calls.append(subtask.description)
        return WorkerResult(output=f"answer for {subtask.description}", confidence=confidence)
    return worker


def test_every_retry_is_a_separate_attempt(monkeypatch, memory_manager: MemoryManager):
    calls: list = []
    monkeypatch.setitem(workers.WORKER_REGISTRY, "research", _counting_worker(0.1, calls))

    state = run_goal("Find facts about coffee.", memory_manager, log_episode=False)

    research = [a for a in state["attempts"] if a.subtask_id == "task-1"]
    assert [a.attempt for a in research] == list(range(1, MAX_REPLANS + 2))
    assert all(a.status == "low_confidence" for a in research)
    assert len({a.idempotency_key for a in research}) == len(research)
    assert all(a.run_id == state["run_id"] for a in state["attempts"])
    assert state["stop_reason"] == "max_replans"
    assert len(calls) == MAX_REPLANS + 1


def test_confident_run_stops_without_replanning(memory_manager: MemoryManager):
    state = run_goal("Tell me about solar panels.", memory_manager, log_episode=False)
    assert state["stop_reason"] == "all_confident"
    assert state["replans"] == 0
    assert all(a.attempt == 1 for a in state["attempts"])


def test_resumed_run_reuses_completed_attempts(monkeypatch, memory_manager: MemoryManager):
    calls: list = []
    monkeypatch.setitem(workers.WORKER_REGISTRY, "research", _counting_worker(0.9, calls))

    first = run_goal("Find facts about coffee.", memory_manager, run_id="run-123", log_episode=False)
    second = run_goal("Find facts about coffee.", memory_manager, run_id="run-123", log_episode=False)

    assert len(calls) == 1  # the research worker did not run again
    assert [a.status for a in second["attempts"]] == ["reused"] * len(second["attempts"])
    assert [a.idempotency_key for a in first["attempts"]] == [a.idempotency_key for a in second["attempts"]]


def test_crashing_worker_is_recorded_and_replanned(monkeypatch, memory_manager: MemoryManager):
    def boom(subtask, memory):
        raise RuntimeError("tool unavailable")

    monkeypatch.setitem(workers.WORKER_REGISTRY, "research", boom)
    state = run_goal("Find facts about coffee.", memory_manager, log_episode=False)

    errors = [a for a in state["attempts"] if a.status == "error"]
    assert errors and "tool unavailable" in errors[0].output
    assert errors[0].raw_confidence == 0.0
    assert state["replans"] == MAX_REPLANS
    assert state["final_answer"]


def test_run_goal_logs_episode_with_attempts(tmp_path: Path, memory_manager: MemoryManager):
    memory_manager.episodic._log_path = tmp_path / "log.jsonl"
    state = run_goal("Calculate 12 * 7 + 5", memory_manager)

    [episode] = memory_manager.episodic.all_episodes()
    assert episode.run_id == state["run_id"]
    assert episode.stop_reason == "all_confident"
    assert len(episode.attempts) == len(state["attempts"]) >= 2
    assert episode.score is None  # unscored outside the eval harness


def test_old_episodic_lines_still_load(tmp_path: Path):
    log = tmp_path / "old.jsonl"
    log.write_text(json.dumps({
        "goal": "g", "plan": ["p"], "outcome": "o", "score": 1.0,
        "llm_calls": 3, "worker_invocations": 2, "replans": 0, "timestamp": "2026-01-01T00:00:00Z",
    }) + "\n", encoding="utf-8")
    episodic = EpisodicMemory(embed_fn=lambda t: deterministic_embedding(t, 16), log_path=log)
    [episode] = episodic.all_episodes()
    assert episode.score == 1.0 and episode.attempts == [] and episode.run_id == ""


def test_idempotency_key_is_stable_and_normalized():
    a = make_idempotency_key("r", "task-1", 1, "research", "Find   Coffee facts")
    b = make_idempotency_key("r", "task-1", 1, "research", "find coffee facts")
    c = make_idempotency_key("r", "task-1", 2, "research", "find coffee facts")
    assert a == b and a != c and len(a) == 24


def test_sqlite_store_survives_restart(tmp_path: Path):
    path = tmp_path / "idem.sqlite"
    store = IdempotencyStore(path)
    store.put("k1", WorkerResult(output="done", confidence=0.8))
    store.close()

    reopened = IdempotencyStore(path)
    assert reopened.get("k1").output == "done"
    assert reopened.get("missing") is None
    reopened.close()


def test_every_worker_declares_side_effects():
    assert set(workers.WORKER_SIDE_EFFECTS) == set(workers.WORKER_REGISTRY)
