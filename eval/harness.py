"""
Agentic-eval harness.

Runs a set of benchmark goals (benchmarks/goals.yaml) through the
orchestrator graph and scores each run on:

  - task completion : were the expected key facts present in the final
                       answer? Checked via simple case-insensitive
                       containment first, then (for facts not found
                       verbatim) via embedding cosine similarity against
                       the final answer text -- NOT exact string matching.
  - efficiency       : number of LLM calls, worker invocations, and
                        replans used to reach the answer. Fewer is better;
                        reported as raw counts plus a normalized efficiency
                        score.

Produces a summary report: printed to stdout and returned/dumped as JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from agentic_memory.manager import MemoryManager
from orchestrator.graph import episode_from_state, run_goal
from orchestrator.schemas import AttemptRecord

DEFAULT_BENCHMARKS_PATH = Path("benchmarks") / "goals.yaml"
SIMILARITY_MATCH_THRESHOLD = 0.65


@dataclass
class GoalResult:
    goal: str
    final_answer: str
    expected_facts: list[str]
    facts_found: list[str] = field(default_factory=list)
    facts_missed: list[str] = field(default_factory=list)
    completion_score: float = 0.0
    llm_calls: int = 0
    cache_hits: int = 0
    worker_invocations: int = 0
    replans: int = 0
    efficiency_score: float = 0.0
    duration_seconds: float = 0.0
    run_id: str = ""
    stop_reason: str = ""
    attempts: list[AttemptRecord] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[GoalResult]
    average_completion_score: float
    average_efficiency_score: float
    total_llm_calls: int
    total_cache_hits: int
    total_replans: int

    def to_dict(self) -> dict:
        return {
            "results": [
                {
                    "goal": r.goal,
                    "final_answer": r.final_answer,
                    "expected_facts": r.expected_facts,
                    "facts_found": r.facts_found,
                    "facts_missed": r.facts_missed,
                    "completion_score": r.completion_score,
                    "llm_calls": r.llm_calls,
                    "cache_hits": r.cache_hits,
                    "worker_invocations": r.worker_invocations,
                    "replans": r.replans,
                    "efficiency_score": r.efficiency_score,
                    "duration_seconds": round(r.duration_seconds, 4),
                    "run_id": r.run_id,
                    "stop_reason": r.stop_reason,
                    "attempts": [a.model_dump() for a in r.attempts],
                }
                for r in self.results
            ],
            "summary": {
                "average_completion_score": round(self.average_completion_score, 4),
                "average_efficiency_score": round(self.average_efficiency_score, 4),
                "total_llm_calls": self.total_llm_calls,
                "total_cache_hits": self.total_cache_hits,
                "total_replans": self.total_replans,
                "num_goals": len(self.results),
            },
        }


def load_benchmark_goals(path: Optional[Path] = None) -> list[dict]:
    path = path or DEFAULT_BENCHMARKS_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or []


def _fact_present(fact: str, answer: str, embed_fn) -> bool:
    if fact.lower() in answer.lower():
        return True
    # Fall back to embedding similarity rather than requiring an exact match.
    fact_vec = np.array(embed_fn(fact), dtype=float)
    answer_vec = np.array(embed_fn(answer), dtype=float)
    denom = np.linalg.norm(fact_vec) * np.linalg.norm(answer_vec)
    if denom == 0:
        return False
    similarity = float(np.dot(fact_vec, answer_vec) / denom)
    return similarity >= SIMILARITY_MATCH_THRESHOLD


def score_completion(final_answer: str, expected_facts: list[str], embed_fn) -> tuple[float, list[str], list[str]]:
    if not expected_facts:
        return 1.0, [], []
    found, missed = [], []
    for fact in expected_facts:
        if _fact_present(fact, final_answer, embed_fn):
            found.append(fact)
        else:
            missed.append(fact)
    score = len(found) / len(expected_facts)
    return score, found, missed


def score_efficiency(llm_calls: int, worker_invocations: int, replans: int) -> float:
    """
    Normalize raw counts into a [0, 1] efficiency score where fewer calls is
    better. Uses a simple decay so that very economical runs (few calls, no
    replans) score close to 1.0 and expensive/looping runs score closer to 0.
    """
    total_ops = llm_calls + worker_invocations + 2 * replans
    if total_ops <= 0:
        return 1.0
    # Decay constant chosen so ~10 total ops -> ~0.5 score.
    return float(np.exp(-total_ops / 10.0))


def run_eval(
    benchmarks_path: Optional[Path] = None,
    memory: Optional[MemoryManager] = None,
    log_episodes: bool = True,
) -> EvalReport:
    """Run every benchmark goal through the orchestrator graph and score it."""
    memory = memory or MemoryManager()
    goals = load_benchmark_goals(benchmarks_path)

    results: list[GoalResult] = []
    for entry in goals:
        goal_text = entry["goal"]
        expected_facts = entry.get("expected_facts", [])

        memory.reset_counters()
        start = time.perf_counter()
        final_state = run_goal(
            goal_text, memory, session_id=f"eval-{abs(hash(goal_text))}", log_episode=False
        )
        duration = time.perf_counter() - start

        final_answer = final_state["final_answer"]
        worker_invocations = sum(1 for a in final_state["attempts"] if a.status != "reused")
        replans = final_state["replans"]

        completion_score, found, missed = score_completion(
            final_answer, expected_facts, memory.llm_client.embed
        )
        efficiency_score = score_efficiency(
            memory.llm_call_count, worker_invocations, replans
        )

        result = GoalResult(
            goal=goal_text,
            final_answer=final_answer,
            expected_facts=expected_facts,
            facts_found=found,
            facts_missed=missed,
            completion_score=completion_score,
            llm_calls=memory.llm_call_count,
            cache_hits=memory.cache_hit_count,
            worker_invocations=worker_invocations,
            replans=replans,
            efficiency_score=efficiency_score,
            duration_seconds=duration,
            run_id=final_state["run_id"],
            stop_reason=final_state["stop_reason"],
            attempts=list(final_state["attempts"]),
        )
        results.append(result)

        if log_episodes:
            memory.episodic.log_episode(episode_from_state(final_state, memory, score=completion_score))

    avg_completion = sum(r.completion_score for r in results) / len(results) if results else 0.0
    avg_efficiency = sum(r.efficiency_score for r in results) / len(results) if results else 0.0

    return EvalReport(
        results=results,
        average_completion_score=avg_completion,
        average_efficiency_score=avg_efficiency,
        total_llm_calls=sum(r.llm_calls for r in results),
        total_cache_hits=sum(r.cache_hits for r in results),
        total_replans=sum(r.replans for r in results),
    )


def print_report(report: EvalReport) -> None:
    print("=" * 72)
    print("AGENTIC EVAL REPORT")
    print("=" * 72)
    for r in report.results:
        status = "PASS" if r.completion_score >= 0.99 else f"PARTIAL ({r.completion_score:.2f})"
        print(f"\nGoal: {r.goal}")
        print(f"  Completion: {status}  | Efficiency: {r.efficiency_score:.3f}")
        print(f"  LLM calls: {r.llm_calls}  Cache hits: {r.cache_hits}  "
              f"Workers: {r.worker_invocations}  Replans: {r.replans}  "
              f"Time: {r.duration_seconds:.3f}s")
        if r.facts_missed:
            print(f"  Missed facts: {r.facts_missed}")
    print("\n" + "-" * 72)
    print(f"Average completion score: {report.average_completion_score:.4f}")
    print(f"Average efficiency score: {report.average_efficiency_score:.4f}")
    print(f"Total LLM calls: {report.total_llm_calls}  Total cache hits: {report.total_cache_hits}  "
          f"Total replans: {report.total_replans}")
    print("=" * 72)


def main() -> None:
    report = run_eval()
    print_report(report)
    output_path = Path("eval_report.json")
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nJSON report written to {output_path}")


if __name__ == "__main__":
    main()
