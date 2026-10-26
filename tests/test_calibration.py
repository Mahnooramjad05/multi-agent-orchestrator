"""Confidence calibration: binning, smoothing, metrics, persistence and graph wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_memory.manager import MemoryManager
from eval.calibrate import fit_from_benchmark, label_attempts
from orchestrator import workers
from orchestrator.calibration import (
    Calibrator,
    Sample,
    brier_score,
    expected_calibration_error,
    load_calibrator,
)
from orchestrator.graph import run_goal
from orchestrator.schemas import AttemptRecord, WorkerResult


def _samples(worker: str, confidence: float, successes: int, failures: int) -> list[Sample]:
    return ([Sample(worker, confidence, True)] * successes
            + [Sample(worker, confidence, False)] * failures)


def test_calibrated_value_is_smoothed_success_rate():
    # 8 attempts at raw 0.85, only 2 succeeded: the worker is overconfident.
    cal = Calibrator(num_bins=5, prior_strength=2.0).fit(_samples("research", 0.85, 2, 6))
    # (successes + k * raw) / (n + k) = (2 + 2 * 0.85) / (8 + 2) = 0.37
    assert cal.calibrate("research", 0.85) == pytest.approx(0.37)
    # Same bin, different raw score: the prior term moves with it.
    assert cal.calibrate("research", 0.95) == pytest.approx((2 + 2 * 0.95) / 10)


def test_unseen_bins_and_workers_fall_back_to_raw():
    cal = Calibrator(num_bins=5).fit(_samples("research", 0.85, 1, 1))
    assert cal.calibrate("research", 0.1) == 0.1     # empty bin
    assert cal.calibrate("calculator", 0.97) == 0.97  # unseen worker


def test_confidence_of_one_lands_in_top_bin():
    cal = Calibrator(num_bins=5).fit(_samples("calculator", 1.0, 3, 0))
    assert cal.bins["calculator"][4].n == 3


def test_ece_and_brier_hand_computed():
    pairs = [(0.9, True), (0.9, False), (0.1, False), (0.1, False)]
    # bin 0.0-0.2: conf 0.1, accuracy 0.0 -> gap 0.1; bin 0.8-1.0: conf 0.9, accuracy 0.5 -> gap 0.4
    assert expected_calibration_error(pairs, num_bins=5) == pytest.approx(0.5 * 0.1 + 0.5 * 0.4)
    assert brier_score(pairs) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    assert expected_calibration_error([]) == 0.0


def test_save_and_load_roundtrip(tmp_path: Path, monkeypatch):
    cal = Calibrator(num_bins=4).fit(_samples("writer", 0.6, 1, 3))
    path = cal.save(tmp_path / "cal.json")
    loaded = load_calibrator(path)
    assert loaded.calibrate("writer", 0.6) == pytest.approx(cal.calibrate("writer", 0.6))

    monkeypatch.setenv("CALIBRATION_PATH", str(tmp_path / "missing.json"))
    assert load_calibrator() is None


def _attempt(worker: str, output: str, status: str = "ok", confidence: float = 0.8) -> AttemptRecord:
    return AttemptRecord(run_id="r", subtask_id="task-1", attempt=1, idempotency_key="k",
                         worker=worker, description="d", status=status,
                         raw_confidence=confidence, output=output)


def test_label_attempts():
    attempts = [
        _attempt("research", "Brazil is the largest producer."),
        _attempt("research", "No relevant documents."),
        _attempt("calculator", "12 * 7 + 5 = 89"),
        _attempt("writer", "anything"),
        _attempt("research", "Brazil", status="reused"),
    ]
    samples = label_attempts(attempts, ["Brazil", "89"], completion_score=0.4)
    assert [(s.worker, s.success) for s in samples] == [
        ("research", True), ("research", False), ("calculator", True), ("writer", False)]


def test_graph_applies_calibrated_confidence(monkeypatch, memory_manager: MemoryManager):
    """A worker that reports 0.9 but historically fails should trigger a replan."""
    monkeypatch.setitem(workers.WORKER_REGISTRY, "research",
                        lambda subtask, memory: WorkerResult(output="x", confidence=0.9))
    overconfident = Calibrator().fit(_samples("research", 0.9, 0, 20))

    state = run_goal("Find facts about coffee.", memory_manager,
                     calibrator=overconfident, log_episode=False)
    first = state["attempts"][0]
    assert first.raw_confidence == 0.9
    assert first.calibrated_confidence < 0.5
    assert first.status == "low_confidence"
    assert state["replans"] > 0

    uncalibrated = run_goal("Find facts about coffee.", memory_manager,
                            calibrator=None, log_episode=False)
    assert uncalibrated["replans"] == 0
    assert uncalibrated["attempts"][0].calibrated_confidence is None


def test_fit_from_small_benchmark(tmp_path: Path):
    goals = tmp_path / "goals.yaml"
    goals.write_text(
        '- goal: "Which country produces the most coffee in the world?"\n'
        '  expected_facts: ["Brazil"]\n'
        '- goal: "Who painted the Mona Lisa?"\n'
        '  expected_facts: ["Leonardo da Vinci"]\n'
        '- goal: "Calculate 3 + 4"\n'
        '  expected_facts: ["7"]\n',
        encoding="utf-8",
    )
    calibrator, samples = fit_from_benchmark(goals, num_bins=5)
    assert {s.worker for s in samples} >= {"research", "calculator", "writer"}
    assert any(not s.success for s in samples)  # the Mona Lisa answer isn't in the docs
    assert set(calibrator.bins) == {s.worker for s in samples}
