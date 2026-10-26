"""
LangGraph graph definition for the orchestrator-workers system.

Graph structure
----------------

    START -> decompose -> dispatch_workers -> assess -> synthesize -> END
                                 ^               |
                                 |               | (loop, capped)
                                 +--- replan <---+

- decompose        : orchestrator node. Decomposes the incoming goal into a
                      Plan (structured, via a Pydantic schema + LLM call).
- dispatch_workers  : runs every pending subtask through its assigned worker
                      node (research / calculator / writer), collecting a
                      WorkerResult (with self-reported confidence) for each.
                      Every attempt is recorded as an AttemptRecord and keyed
                      by an idempotency key, so a completed attempt is never
                      executed twice.
- assess            : decides whether to stop, and records why in
                      `stop_reason`. The conditional edge after it routes to
                      `replan` when any subtask's confidence is below
                      CONFIDENCE_THRESHOLD and we have not exceeded
                      MAX_REPLANS; otherwise to `synthesize`.
- replan            : re-decomposes only the failed subtask(s) into fresh
                      subtasks (dynamic replanning / error recovery), then
                      loops back to dispatch_workers. Bounded by MAX_REPLANS
                      to avoid infinite loops.
- synthesize        : merges all worker outputs into a final answer.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agentic_memory.manager import MemoryManager
from orchestrator.idempotency import make_idempotency_key
from orchestrator.schemas import (
    AttemptRecord,
    EpisodicRecord,
    Plan,
    SubTask,
    SubTaskExecution,
    WorkerResult,
)
from orchestrator.workers import WORKER_REGISTRY

CONFIDENCE_THRESHOLD = 0.5
MAX_REPLANS = 2
ATTEMPT_OUTPUT_LIMIT = 2000  # characters of worker output kept per attempt in the log


class OrchestratorState(TypedDict):
    goal: str
    session_id: str
    run_id: str
    plan_reasoning: str
    executions: list[SubTaskExecution]
    attempts: list[AttemptRecord]
    replans: int
    stop_reason: str
    final_answer: str
    trace: list[dict]
    duration_seconds: float


def _log(state: OrchestratorState, event: str, **details) -> None:
    state["trace"].append({"event": event, **details})


def _low_confidence(state: OrchestratorState) -> list[SubTaskExecution]:
    return [
        ex
        for ex in state["executions"]
        if ex.effective_confidence is not None and ex.effective_confidence < CONFIDENCE_THRESHOLD
    ]


def build_graph(memory: MemoryManager):
    """Construct and compile the LangGraph StateGraph for one orchestrator run."""

    def decompose_node(state: OrchestratorState) -> OrchestratorState:
        system = (
            "You are an orchestrator agent. Decompose the user's goal into a small "
            "number of concrete subtasks. Each subtask must be assigned to exactly "
            "one worker: 'research' (retrieval/fact-finding), 'calculator' "
            "(arithmetic), or 'writer' (drafting/synthesis prose). Prefer 2-4 subtasks."
        )
        prompt = f"Goal: {state['goal']}"
        plan: Plan = memory.cached_invoke_structured(system, prompt, Plan)

        executions = [SubTaskExecution(subtask=st, attempt=1) for st in plan.subtasks]
        state["executions"] = executions
        state["plan_reasoning"] = plan.reasoning
        _log(state, "decompose", subtasks=[st.model_dump() for st in plan.subtasks])
        return state

    def run_attempt(state: OrchestratorState, execution: SubTaskExecution, context: str) -> None:
        """Run (or reuse) one worker attempt and record it."""
        subtask = execution.subtask
        key = make_idempotency_key(
            state["run_id"], subtask.id, execution.attempt, subtask.worker, subtask.description
        )
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start = time.perf_counter()

        reused = memory.idempotency.get(key)
        if reused is not None:
            result, status = reused, "reused"
        else:
            worker_fn = WORKER_REGISTRY[subtask.worker]
            try:
                if subtask.worker == "writer":
                    result = worker_fn(subtask, memory, context)
                else:
                    result = worker_fn(subtask, memory)
                status = "ok"
            except Exception as exc:  # a crashing worker becomes a low-confidence result, not a failed run
                result = WorkerResult(
                    output=f"Worker error: {type(exc).__name__}: {exc}",
                    confidence=0.0,
                    notes="Worker raised an exception; eligible for replanning.",
                )
                status = "error"
            else:
                memory.idempotency.put(key, result)

        execution.result = result
        effective = execution.effective_confidence
        if status == "ok" and effective is not None and effective < CONFIDENCE_THRESHOLD:
            status = "low_confidence"

        state["attempts"].append(
            AttemptRecord(
                run_id=state["run_id"],
                subtask_id=subtask.id,
                attempt=execution.attempt,
                idempotency_key=key,
                worker=subtask.worker,
                description=subtask.description,
                status=status,
                raw_confidence=result.confidence,
                calibrated_confidence=execution.calibrated_confidence,
                output=result.output[:ATTEMPT_OUTPUT_LIMIT],
                notes=result.notes,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                started_at=started_at,
            )
        )
        _log(
            state,
            "worker_run",
            subtask_id=subtask.id,
            worker=subtask.worker,
            confidence=result.confidence,
            attempt=execution.attempt,
            idempotency_key=key,
            status=status,
        )

    def dispatch_workers_node(state: OrchestratorState) -> OrchestratorState:
        # Build a running context string from already-completed subtasks so
        # later workers (especially the writer) can build on earlier results.
        context_parts = [
            f"- {ex.subtask.description}: {ex.result.output}"
            for ex in state["executions"]
            if ex.result is not None
        ]
        context = "\n".join(context_parts)

        for execution in state["executions"]:
            confidence = execution.effective_confidence
            if confidence is not None and confidence >= CONFIDENCE_THRESHOLD:
                continue  # already succeeded in a previous pass
            run_attempt(state, execution, context)
        return state

    def assess_node(state: OrchestratorState) -> OrchestratorState:
        """Decide whether the run stops here; `stop_reason` stays empty to keep going."""
        if not _low_confidence(state):
            state["stop_reason"] = "all_confident"
        elif state["replans"] >= MAX_REPLANS:
            state["stop_reason"] = "max_replans"
        else:
            state["stop_reason"] = ""
        if state["stop_reason"]:
            _log(state, "stop", reason=state["stop_reason"])
        return state

    def route_after_assess(state: OrchestratorState) -> str:
        return "synthesize" if state["stop_reason"] else "replan"

    def replan_node(state: OrchestratorState) -> OrchestratorState:
        state["replans"] += 1
        system = (
            "You are an orchestrator agent recovering from a low-confidence worker "
            "result. Re-decompose the failed subtask into ONE improved, more specific "
            "subtask, assigned to the worker best suited to succeed this time "
            "('research', 'calculator', or 'writer')."
        )
        for execution in _low_confidence(state):
            prompt = (
                f"Original goal: {state['goal']}\n"
                f"Failed subtask: {execution.subtask.description}\n"
                f"Worker used: {execution.subtask.worker}\n"
                f"Reported confidence: {execution.result.confidence if execution.result else 'n/a'}\n"
                f"Worker notes: {execution.result.notes if execution.result else ''}\n"
                "Produce a replacement subtask."
            )
            new_subtask: SubTask = memory.cached_invoke_structured(system, prompt, SubTask)
            new_subtask.id = execution.subtask.id  # keep the same slot/id
            execution.subtask = new_subtask
            execution.result = None
            execution.calibrated_confidence = None
            execution.replanned = True
            execution.attempt += 1
            _log(
                state,
                "replan",
                subtask_id=new_subtask.id,
                new_description=new_subtask.description,
                new_worker=new_subtask.worker,
                replan_count=state["replans"],
            )
        return state

    def synthesize_node(state: OrchestratorState) -> OrchestratorState:
        bullet_points = "\n".join(
            f"- ({ex.subtask.worker}) {ex.subtask.description} -> "
            f"{ex.result.output if ex.result else '[no result]'}"
            for ex in state["executions"]
        )
        system = (
            "You are a synthesis agent. Combine the worker outputs below into one "
            "clear, coherent final answer to the original goal. Do not repeat the "
            "raw bullet list verbatim -- write connected prose."
        )
        prompt = f"Goal: {state['goal']}\n\nWorker outputs:\n{bullet_points}\n\nFinal answer:"
        final_answer = memory.cached_invoke_text(system, prompt)
        state["final_answer"] = final_answer
        _log(state, "synthesize", final_answer_preview=final_answer[:200])
        return state

    graph = StateGraph(OrchestratorState)
    graph.add_node("decompose", decompose_node)
    graph.add_node("dispatch_workers", dispatch_workers_node)
    graph.add_node("assess", assess_node)
    graph.add_node("replan", replan_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "dispatch_workers")
    graph.add_edge("dispatch_workers", "assess")
    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {"replan": "replan", "synthesize": "synthesize"},
    )
    graph.add_edge("replan", "dispatch_workers")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_goal(
    goal: str,
    memory: MemoryManager,
    session_id: str = "default",
    run_id: Optional[str] = None,
    log_episode: bool = True,
) -> OrchestratorState:
    """Convenience entry point: run one goal through the compiled graph end to end.

    Pass an existing `run_id` to resume a run: attempts that already completed
    under that id are reused via their idempotency keys instead of re-executed.
    Set `log_episode=False` when the caller logs the episode itself (the eval
    harness does, so it can attach a completion score).
    """
    compiled = build_graph(memory)
    initial_state: OrchestratorState = {
        "goal": goal,
        "session_id": session_id,
        "run_id": run_id or uuid.uuid4().hex,
        "plan_reasoning": "",
        "executions": [],
        "attempts": [],
        "replans": 0,
        "stop_reason": "",
        "final_answer": "",
        "trace": [],
        "duration_seconds": 0.0,
    }
    start = time.perf_counter()
    memory.short_term.add_turn(session_id, "user", goal)
    final_state: OrchestratorState = compiled.invoke(initial_state)
    memory.short_term.add_turn(session_id, "assistant", final_state["final_answer"])
    memory.long_term.add_fact(session_id, f"Goal: {goal} -> {final_state['final_answer'][:300]}")
    final_state["duration_seconds"] = round(time.perf_counter() - start, 4)

    if log_episode:
        memory.episodic.log_episode(episode_from_state(final_state, memory))
    return final_state


def episode_from_state(
    state: OrchestratorState, memory: MemoryManager, score: Optional[float] = None
) -> EpisodicRecord:
    """Build the episodic log entry for a finished run."""
    return EpisodicRecord(
        goal=state["goal"],
        plan=[ex.subtask.description for ex in state["executions"]],
        outcome=state["final_answer"],
        score=score,
        llm_calls=memory.llm_call_count,
        worker_invocations=sum(1 for a in state["attempts"] if a.status != "reused"),
        replans=state["replans"],
        run_id=state["run_id"],
        attempts=state["attempts"],
        duration_seconds=state["duration_seconds"],
        stop_reason=state["stop_reason"],
    )
