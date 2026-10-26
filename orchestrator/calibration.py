"""
Confidence calibration for worker scores.

Each worker reports a confidence score from a fixed heuristic (for example,
the research worker scores higher when it retrieves more documents). A score
of 0.85 is only useful if subtasks scored 0.85 actually succeed about 85% of
the time. This module measures that and corrects for it.

Method: per-worker reliability bins. Raw scores are grouped into equal-width
bins; each bin's calibrated value is the observed success rate of attempts
in that bin, shrunk toward the raw score with a Beta prior so bins with only
a few samples don't swing to 0 or 1:

    calibrated = (successes + prior_strength * raw) / (n + prior_strength)

Bins (or workers) with no data return the raw score unchanged.

The fitted calibrator is saved as JSON (data/calibration.json by default)
and loaded automatically by run_goal() when the file exists. Fit it with:

    python -m eval.calibrate
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_CALIBRATION_PATH = Path("data") / "calibration.json"


@dataclass
class Sample:
    worker: str
    confidence: float
    success: bool


@dataclass
class BinStats:
    lower: float
    upper: float
    n: int = 0
    successes: int = 0
    mean_confidence: float = 0.0


@dataclass
class Calibrator:
    num_bins: int = 5
    prior_strength: float = 2.0
    bins: dict[str, list[BinStats]] = field(default_factory=dict)

    def _bin_index(self, confidence: float) -> int:
        return min(int(confidence * self.num_bins), self.num_bins - 1)

    def fit(self, samples: Iterable[Sample]) -> "Calibrator":
        width = 1.0 / self.num_bins
        sums: dict[str, list[float]] = {}
        self.bins = {}
        for s in samples:
            if s.worker not in self.bins:
                self.bins[s.worker] = [
                    BinStats(lower=round(i * width, 6), upper=round((i + 1) * width, 6))
                    for i in range(self.num_bins)
                ]
                sums[s.worker] = [0.0] * self.num_bins
            idx = self._bin_index(s.confidence)
            stats = self.bins[s.worker][idx]
            stats.n += 1
            stats.successes += int(s.success)
            sums[s.worker][idx] += s.confidence
        for worker, worker_bins in self.bins.items():
            for idx, stats in enumerate(worker_bins):
                if stats.n:
                    stats.mean_confidence = sums[worker][idx] / stats.n
        return self

    def calibrate(self, worker: str, confidence: float) -> float:
        worker_bins = self.bins.get(worker)
        if not worker_bins:
            return confidence
        stats = worker_bins[self._bin_index(confidence)]
        if stats.n == 0:
            return confidence
        k = self.prior_strength
        return (stats.successes + k * confidence) / (stats.n + k)

    # --- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "num_bins": self.num_bins,
            "prior_strength": self.prior_strength,
            "bins": {
                worker: [vars(b) for b in worker_bins] for worker, worker_bins in self.bins.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        return cls(
            num_bins=int(data["num_bins"]),
            prior_strength=float(data["prior_strength"]),
            bins={
                worker: [BinStats(**b) for b in worker_bins]
                for worker, worker_bins in data.get("bins", {}).items()
            },
        )

    def save(self, path: Optional[str | Path] = None) -> Path:
        path = Path(path or DEFAULT_CALIBRATION_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_calibrator(path: Optional[str | Path] = None) -> Optional[Calibrator]:
    """Load the calibrator from CALIBRATION_PATH or data/calibration.json; None if absent."""
    path = Path(path or os.getenv("CALIBRATION_PATH") or DEFAULT_CALIBRATION_PATH)
    if not path.exists():
        return None
    return Calibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --- calibration quality metrics ----------------------------------------------


def expected_calibration_error(pairs: list[tuple[float, bool]], num_bins: int = 5) -> float:
    """Weighted mean gap between average confidence and success rate across bins."""
    if not pairs:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for confidence, success in pairs:
        bins[min(int(confidence * num_bins), num_bins - 1)].append((confidence, success))
    total = len(pairs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(s for _, s in bucket) / len(bucket)
        ece += len(bucket) / total * abs(avg_conf - accuracy)
    return ece


def brier_score(pairs: list[tuple[float, bool]]) -> float:
    """Mean squared error between confidence and outcome (lower is better)."""
    if not pairs:
        return 0.0
    return sum((c - float(s)) ** 2 for c, s in pairs) / len(pairs)
