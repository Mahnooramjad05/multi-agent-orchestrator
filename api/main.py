"""
FastAPI application exposing the multi-agent orchestrator over HTTP.

Endpoints
---------
GET  /health   -> basic liveness/readiness check, reports which LLM backend is active.
POST /run      -> accepts a goal string, runs it through the orchestrator
                  graph, and returns the final synthesized answer plus a
                  trace of subtasks/workers/replans.

Run locally with:
    uvicorn api.main:app --reload

The underlying MemoryManager (and therefore LLM client, semantic cache, and
Qdrant in-memory vector store) is created once at startup and reused across
requests.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agentic_memory.manager import MemoryManager
from orchestrator.budget import Budget
from orchestrator.graph import run_goal
from orchestrator.schemas import AttemptRecord

app = FastAPI(
    title="Multi-Agent Orchestration System",
    description="Orchestrator-workers multi-agent system with agentic memory, "
    "semantic caching, dynamic replanning, and an eval harness.",
    version="1.0.0",
)

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


class RunRequest(BaseModel):
    goal: str = Field(..., description="The high-level goal to accomplish.")
    session_id: str = Field(default="default", description="Session id for memory scoping.")
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="Token budget for this run (default: BUDGET_MAX_TOKENS or unlimited)."
    )
    max_seconds: Optional[float] = Field(
        default=None, gt=0, description="Wall-clock budget in seconds (default: BUDGET_MAX_SECONDS or unlimited)."
    )


class SubtaskTrace(BaseModel):
    id: str
    description: str
    worker: str
    confidence: Optional[float] = None
    replanned: bool = False


class RunResponse(BaseModel):
    goal: str
    run_id: str
    stop_reason: str
    final_answer: str
    subtasks: list[SubtaskTrace]
    replans: int
    llm_calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    tokens_estimated: bool
    budget: dict
    attempts: list[AttemptRecord]
    trace: list[dict]


@app.get("/health")
def health() -> dict:
    memory = get_memory_manager()
    return {
        "status": "ok",
        "llm_provider": memory.llm_client.name,
        "mock_mode": memory.llm_client.name == "mock",
    }


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    memory = get_memory_manager()
    memory.reset_counters()

    budget = Budget.from_env()
    if request.max_tokens is not None:
        budget.max_tokens = request.max_tokens
    if request.max_seconds is not None:
        budget.max_seconds = request.max_seconds
    final_state = run_goal(request.goal, memory, session_id=request.session_id, budget=budget)

    subtasks = [
        SubtaskTrace(
            id=ex.subtask.id,
            description=ex.subtask.description,
            worker=ex.subtask.worker,
            confidence=ex.result.confidence if ex.result else None,
            replanned=ex.replanned,
        )
        for ex in final_state["executions"]
    ]

    return RunResponse(
        goal=request.goal,
        run_id=final_state["run_id"],
        stop_reason=final_state["stop_reason"],
        final_answer=final_state["final_answer"],
        subtasks=subtasks,
        replans=final_state["replans"],
        llm_calls=memory.llm_call_count,
        cache_hits=memory.cache_hit_count,
        input_tokens=final_state["usage"]["input_tokens"],
        output_tokens=final_state["usage"]["output_tokens"],
        tokens_estimated=final_state["usage"]["estimated"],
        budget=final_state["budget"],
        attempts=final_state["attempts"],
        trace=final_state["trace"],
    )
