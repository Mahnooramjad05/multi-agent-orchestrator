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


class SubTaskExecution(BaseModel):
    """A subtask paired with its execution result, for tracing."""

    subtask: SubTask
    result: Optional[WorkerResult] = None
    replanned: bool = False
    attempt: int = 1


class EpisodicRecord(BaseModel):
    """A single row of the episodic memory log (one full orchestrator run)."""

    goal: str
    plan: list[str] = Field(description="Subtask descriptions in the final plan.")
    outcome: str = Field(description="Final synthesized answer.")
    score: float = Field(description="Simple heuristic quality score in [0, 1].")
    llm_calls: int = 0
    worker_invocations: int = 0
    replans: int = 0
    timestamp: str = ""
