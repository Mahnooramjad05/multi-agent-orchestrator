"""
Fit worker confidence calibration from eval-harness outcomes.

Runs the benchmark goals with calibration switched off (raw worker scores),
labels every worker attempt as a success or failure, fits a per-worker
Calibrator, and writes it to data/calibration.json, where run_goal() picks
it up automatically.

Labels are a proxy, since the benchmark has expected facts per goal, not per
subtask (see eval.harness.label_attempts):
  - research / calculator attempts succeed if their output contains at least
    one of the goal's expected facts (case-insensitive containment);
  - writer attempts succeed if the goal's final completion score is >= 0.5.

ECE and Brier score are reported before and after calibration. The "after"
numbers are in-sample (fitted and measured on the same goals), so treat
them as an upper bound on the improvement; grow the benchmark for a fairer
estimate.

Usage:
    python -m eval.calibrate [--out data/calibration.json] [--bins 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from agentic_memory.manager import MemoryManager
from eval.harness import EvalReport, label_attempts, run_eval
from orchestrator.calibration import (
    DEFAULT_CALIBRATION_PATH,
    Calibrator,
    Sample,
    brier_score,
    expected_calibration_error,
)

__all__ = ["fit_from_benchmark", "label_attempts", "samples_from_report"]


def samples_from_report(report: EvalReport) -> list[Sample]:
    samples = []
    for result in report.results:
        samples += label_attempts(result.attempts, result.expected_facts, result.completion_score)
    return samples


def fit_from_benchmark(
    benchmarks_path: Optional[Path] = None, num_bins: int = 5, memory: Optional[MemoryManager] = None
) -> tuple[Calibrator, list[Sample]]:
    report = run_eval(
        benchmarks_path=benchmarks_path,
        memory=memory or MemoryManager(),
        log_episodes=False,
        calibrator=None,  # fit on raw scores
    )
    samples = samples_from_report(report)
    return Calibrator(num_bins=num_bins).fit(samples), samples


def print_calibration_report(calibrator: Calibrator, samples: list[Sample]) -> None:
    print("=" * 72)
    print("CONFIDENCE CALIBRATION")
    print("=" * 72)
    for worker, worker_bins in sorted(calibrator.bins.items()):
        print(f"\nWorker: {worker}")
        print(f"  {'bin':<12}{'n':>4}{'avg raw':>10}{'success':>10}{'calibrated':>12}")
        for b in worker_bins:
            if not b.n:
                continue
            rate = b.successes / b.n
            calibrated = calibrator.calibrate(worker, b.mean_confidence)
            print(f"  {b.lower:.1f}-{b.upper:.1f}{'':<5}{b.n:>4}{b.mean_confidence:>10.2f}"
                  f"{rate:>10.2f}{calibrated:>12.2f}")

    raw_pairs = [(s.confidence, s.success) for s in samples]
    cal_pairs = [(calibrator.calibrate(s.worker, s.confidence), s.success) for s in samples]
    print("\n" + "-" * 72)
    print(f"Samples: {len(samples)}  (successes: {sum(s.success for s in samples)})")
    print(f"ECE   raw: {expected_calibration_error(raw_pairs, calibrator.num_bins):.3f}   "
          f"calibrated (in-sample): {expected_calibration_error(cal_pairs, calibrator.num_bins):.3f}")
    print(f"Brier raw: {brier_score(raw_pairs):.3f}   "
          f"calibrated (in-sample): {brier_score(cal_pairs):.3f}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit worker confidence calibration.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--benchmarks", type=Path, default=None)
    args = parser.parse_args()

    calibrator, samples = fit_from_benchmark(args.benchmarks, args.bins)
    print_calibration_report(calibrator, samples)
    path = calibrator.save(args.out)
    print(f"\nCalibration written to {path}")


if __name__ == "__main__":
    main()
