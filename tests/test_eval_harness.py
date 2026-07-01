"""The eval harness should run benchmark goals through the graph and produce scores."""

from __future__ import annotations

from pathlib import Path

from agentic_memory.manager import MemoryManager
from eval.harness import run_eval, score_completion, score_efficiency


def test_score_completion_finds_exact_substring_facts(memory_manager: MemoryManager):
    answer = "Acme Robotics is headquartered in Austin and makes the AutoPick arm."
    score, found, missed = score_completion(
        answer, ["Austin", "AutoPick"], memory_manager.llm_client.embed
    )
    assert score == 1.0
    assert set(found) == {"Austin", "AutoPick"}
    assert missed == []


def test_score_completion_partial_when_fact_missing(memory_manager: MemoryManager):
    answer = "Acme Robotics is headquartered in Austin."
    score, found, missed = score_completion(
        answer, ["Austin", "SomethingNotPresentAtAll"], memory_manager.llm_client.embed
    )
    assert 0.0 < score < 1.0
    assert "Austin" in found


def test_score_efficiency_decreases_with_more_operations():
    cheap = score_efficiency(llm_calls=2, worker_invocations=2, replans=0)
    expensive = score_efficiency(llm_calls=10, worker_invocations=10, replans=3)
    assert cheap > expensive
    assert 0.0 <= expensive <= 1.0
    assert 0.0 <= cheap <= 1.0


def test_run_eval_produces_a_report_with_scores(tmp_path: Path):
    # Use a small subset benchmark file for a fast, deterministic test run.
    subset_path = tmp_path / "goals.yaml"
    subset_path.write_text(
        "- goal: \"What is the AutoPick arm and how many items per hour can it sort?\"\n"
        "  expected_facts:\n"
        "    - \"AutoPick\"\n"
        "- goal: \"Calculate 3 + 4\"\n"
        "  expected_facts:\n"
        "    - \"7\"\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "episodic_log.jsonl"

    memory = MemoryManager()
    memory.episodic._log_path = log_path  # redirect episodic log to a temp file

    report = run_eval(benchmarks_path=subset_path, memory=memory, log_episodes=True)

    assert len(report.results) == 2
    for result in report.results:
        assert 0.0 <= result.completion_score <= 1.0
        assert 0.0 <= result.efficiency_score <= 1.0
        assert result.llm_calls >= 1

    assert 0.0 <= report.average_completion_score <= 1.0
    assert 0.0 <= report.average_efficiency_score <= 1.0

    # Episodic log should now contain one entry per goal.
    episodes = memory.episodic.all_episodes()
    assert len(episodes) == 2
