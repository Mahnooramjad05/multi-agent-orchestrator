"""FastAPI endpoint tests using TestClient (in-process, no real server needed)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_reports_mock_mode():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "mock"
    assert body["mock_mode"] is True


def test_run_endpoint_returns_final_answer_and_trace():
    response = client.post("/run", json={"goal": "What is the capital of France?", "session_id": "api-test"})
    assert response.status_code == 200
    body = response.json()
    assert body["goal"] == "What is the capital of France?"
    assert body["final_answer"]
    assert isinstance(body["subtasks"], list)
    assert len(body["subtasks"]) >= 1
    assert "replans" in body
    assert "llm_calls" in body
    assert "trace" in body
    assert body["run_id"]
    assert body["stop_reason"] in {"all_confident", "max_replans"}
    assert body["attempts"] and body["attempts"][0]["idempotency_key"]


def test_run_endpoint_applies_budget():
    response = client.post("/run", json={"goal": "Find facts about coffee.", "max_tokens": 50})
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "token_budget"
    assert body["budget"]["max_tokens"] == 50
    assert body["final_answer"]
    assert body["input_tokens"] > 0 and body["tokens_estimated"] is True


def test_run_endpoint_rejects_invalid_budget():
    response = client.post("/run", json={"goal": "x", "max_tokens": 0})
    assert response.status_code == 422
