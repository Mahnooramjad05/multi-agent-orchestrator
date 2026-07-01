"""Task decomposition should produce valid, structured output (a Plan of SubTasks)."""

from __future__ import annotations

from orchestrator.llm_client import MockLLMClient
from orchestrator.schemas import Plan, SubTask


def test_mock_decomposition_returns_valid_plan(mock_client: MockLLMClient):
    system = "Decompose the goal."
    prompt = "Goal: Research the history of the internet and write a summary."

    plan = mock_client.invoke_structured(system, prompt, Plan)

    assert isinstance(plan, Plan)
    assert plan.goal
    assert len(plan.subtasks) >= 1
    for subtask in plan.subtasks:
        assert isinstance(subtask, SubTask)
        assert subtask.id
        assert subtask.description
        assert subtask.worker in ("research", "calculator", "writer")


def test_decomposition_routes_calculation_goals_to_calculator(mock_client: MockLLMClient):
    plan = mock_client.invoke_structured(
        "system", "Goal: Calculate 4 + 5 and tell me the result.", Plan
    )
    workers = {st.worker for st in plan.subtasks}
    assert "calculator" in workers


def test_decomposition_routes_informational_goals_to_research(mock_client: MockLLMClient):
    plan = mock_client.invoke_structured(
        "system", "Goal: What is the capital of France?", Plan
    )
    workers = {st.worker for st in plan.subtasks}
    assert "research" in workers


def test_decomposition_always_includes_a_writer_step(mock_client: MockLLMClient):
    plan = mock_client.invoke_structured("system", "Goal: Anything at all.", Plan)
    workers = [st.worker for st in plan.subtasks]
    assert workers[-1] == "writer"
