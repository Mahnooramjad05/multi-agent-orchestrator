"""Pydantic models shared by the orchestrator graph, workers, and memory."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

WorkerName = Literal["research", "calculator", "writer"]


class SubTask(BaseModel):
    """A single unit of work the orchestrator delegates to a worker."""

    id: str = Field(description="Unique identifier for this subtask, e.g. 'task-1'.")
    description: str = Field(description="Natural-language description of the subtask.")
    worker: WorkerName = Field(description="Which specialized worker should handle this subtask.")


class Plan(BaseModel):
    """Structured decomposition of a goal into subtasks."""

    goal: str = Field(description="The original high-level goal.")
    subtasks: list[SubTask] = Field(description="Ordered list of subtasks to accomplish the goal.")
    reasoning: str = Field(default="", description="Brief rationale for the decomposition.")


class WorkerResult(BaseModel):
    """Structured output produced by any worker after handling a subtask."""

    output: str = Field(description="The worker's answer/result for the subtask.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence in [0, 1].")
    notes: str = Field(default="", description="Optional notes, caveats, or explanation.")


AttemptStatus = Literal["ok", "low_confidence", "reused", "error", "skipped_budget"]


class SubTaskExecution(BaseModel):
    """A subtask paired with its execution result, for tracing."""

    subtask: SubTask
    result: Optional[WorkerResult] = None
    replanned: bool = False
    attempt: int = 1
    calibrated_confidence: Optional[float] = Field(
        default=None,
        description="Confidence after calibration; None when no calibration is loaded.",
    )

    @property
    def effective_confidence(self) -> Optional[float]:
        """The confidence the replanning threshold is applied to."""
        if self.result is None:
            return None
        if self.calibrated_confidence is not None:
            return self.calibrated_confidence
        return self.result.confidence


class AttemptRecord(BaseModel):
    """One worker attempt at one subtask slot. A replanned slot produces several."""

    run_id: str
    subtask_id: str = Field(description="Slot id; stays the same across replans.")
    attempt: int
    idempotency_key: str
    worker: WorkerName
    description: str
    status: AttemptStatus
    raw_confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    output: str = Field(default="", description="Worker output, truncated for the log.")
    notes: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    started_at: str = ""


class EpisodicRecord(BaseModel):
    """A single row of the episodic memory log (one full orchestrator run).

    Fields added after v1 all have defaults, so older log lines still load.
    """

    goal: str
    plan: list[str] = Field(description="Subtask descriptions in the final plan.")
    outcome: str = Field(description="Final synthesized answer.")
    score: Optional[float] = Field(
        default=None,
        description="Quality score in [0, 1] from the eval harness; None for unscored runs.",
    )
    llm_calls: int = 0
    worker_invocations: int = 0
    replans: int = 0
    timestamp: str = ""
    run_id: str = ""
    attempts: list[AttemptRecord] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    duration_seconds: float = 0.0
    stop_reason: str = ""
