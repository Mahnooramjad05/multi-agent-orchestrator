"""
LangGraph graph definition for the orchestrator-workers system.

Graph structure
----------------

    START -> decompose -> dispatch_workers -> check_confidence -> synthesize -> END
                                                     |  ^
                                                     |  | (loop, capped)
                                                     v  |
                                                  replan

- decompose        : orchestrator node. Decomposes the incoming goal into a
                      Plan (structured, via a Pydantic schema + LLM call).
- dispatch_workers  : runs every pending subtask through its assigned worker
                      node (research / calculator / writer), collecting a
                      WorkerResult (with self-reported confidence) for each.
- check_confidence  : conditional edge. If any subtask's latest result has
                      confidence below CONFIDENCE_THRESHOLD *and* we have not
                      exceeded MAX_REPLANS, route to `replan`; otherwise route
                      to `synthesize`.
- replan            : re-decomposes only the failed subtask(s) into fresh
                      subtasks (dynamic replanning / error recovery), then
                      loops back to dispatch_workers. Bounded by MAX_REPLANS
                      to avoid infinite loops.
- synthesize        : merges all worker outputs into a final answer.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agentic_memory.manager import MemoryManager
from orchestrator.schemas import Plan, SubTask, SubTaskExecution, WorkerResult
from orchestrator.workers import WORKER_REGISTRY

CONFIDENCE_THRESHOLD = 0.5
MAX_REPLANS = 2


class OrchestratorState(TypedDict):
    goal: str
    session_id: str
    plan_reasoning: str
    executions: list[SubTaskExecution]
    replans: int
    final_answer: str
    trace: list[dict]


def _log(state: OrchestratorState, event: str, **details) -> None:
    state["trace"].append({"event": event, **details})


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
            if execution.result is not None and execution.result.confidence >= CONFIDENCE_THRESHOLD:
                continue  # already succeeded in a previous pass

            worker_fn = WORKER_REGISTRY[execution.subtask.worker]
            if execution.subtask.worker == "writer":
                result: WorkerResult = worker_fn(execution.subtask, memory, context)
            else:
                result = worker_fn(execution.subtask, memory)

            execution.result = result
            _log(
                state,
                "worker_run",
                subtask_id=execution.subtask.id,
                worker=execution.subtask.worker,
                confidence=result.confidence,
                attempt=execution.attempt,
            )
        return state

    def check_confidence(state: OrchestratorState) -> str:
        low_confidence = [
            ex
            for ex in state["executions"]
            if ex.result is not None and ex.result.confidence < CONFIDENCE_THRESHOLD
        ]
        if low_confidence and state["replans"] < MAX_REPLANS:
            return "replan"
        return "synthesize"

    def replan_node(state: OrchestratorState) -> OrchestratorState:
        state["replans"] += 1
        low_confidence = [
            ex
            for ex in state["executions"]
            if ex.result is not None and ex.result.confidence < CONFIDENCE_THRESHOLD
        ]
        system = (
            "You are an orchestrator agent recovering from a low-confidence worker "
            "result. Re-decompose the failed subtask into ONE improved, more specific "
            "subtask, assigned to the worker best suited to succeed this time "
            "('research', 'calculator', or 'writer')."
        )
        for execution in low_confidence:
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
    graph.add_node("replan", replan_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "dispatch_workers")
    graph.add_conditional_edges(
        "dispatch_workers",
        check_confidence,
        {"replan": "replan", "synthesize": "synthesize"},
    )
    graph.add_edge("replan", "dispatch_workers")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_goal(goal: str, memory: MemoryManager, session_id: str = "default") -> OrchestratorState:
    """Convenience entry point: run one goal through the compiled graph end to end."""
    compiled = build_graph(memory)
    initial_state: OrchestratorState = {
        "goal": goal,
        "session_id": session_id,
        "plan_reasoning": "",
        "executions": [],
        "replans": 0,
        "final_answer": "",
        "trace": [],
    }
    memory.short_term.add_turn(session_id, "user", goal)
    final_state: OrchestratorState = compiled.invoke(initial_state)
    memory.short_term.add_turn(session_id, "assistant", final_state["final_answer"])
    memory.long_term.add_fact(session_id, f"Goal: {goal} -> {final_state['final_answer'][:300]}")
    return final_state
