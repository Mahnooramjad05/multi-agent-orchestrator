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
from orchestrator.graph import run_goal

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


class SubtaskTrace(BaseModel):
    id: str
    description: str
    worker: str
    confidence: Optional[float] = None
    replanned: bool = False


class RunResponse(BaseModel):
    goal: str
    final_answer: str
    subtasks: list[SubtaskTrace]
    replans: int
    llm_calls: int
    cache_hits: int
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

    final_state = run_goal(request.goal, memory, session_id=request.session_id)

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
        final_answer=final_state["final_answer"],
        subtasks=subtasks,
        replans=final_state["replans"],
        llm_calls=memory.llm_call_count,
        cache_hits=memory.cache_hit_count,
        trace=final_state["trace"],
    )
