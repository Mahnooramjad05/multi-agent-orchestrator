"""
Pluggable LLM client layer.

Selects a concrete LLM backend based on the LLM_PROVIDER environment variable:
    LLM_PROVIDER=openai     -> uses OPENAI_API_KEY via langchain-openai
    LLM_PROVIDER=anthropic  -> uses ANTHROPIC_API_KEY via langchain-anthropic
    (anything else / unset) -> falls back to a deterministic MOCK client

The mock client is clearly labeled and exists so the whole project (including
pytest and a basic smoke run) works with zero API keys and zero network
calls. Swap in a real provider by setting LLM_PROVIDER and the matching API
key environment variable -- no code changes required.

All LLM calls in this project should go through `get_llm_client()` /
`LLMClient.invoke_json()` / `LLMClient.invoke_text()` so that the semantic
cache (see agentic_memory.semantic_cache) can wrap every call transparently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Common interface every provider (real or mock) must implement."""

    #: Human-readable name of the backend, used in traces/logs.
    name: str = "base"

    @abstractmethod
    def invoke_text(self, system: str, prompt: str) -> str:
        """Return a plain-text completion for the given system+user prompt."""
        raise NotImplementedError

    @abstractmethod
    def invoke_structured(self, system: str, prompt: str, schema: Type[T]) -> T:
        """Return a completion parsed and validated into `schema`."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for `text`."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock backend -- deterministic, offline, no network calls.
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """
    Deterministic mock LLM client.

    MOCK MODE: this backend does not call any real API. It produces
    deterministic, rule-based responses so that the orchestrator graph,
    memory system, semantic cache, and eval harness can all be exercised in
    tests and local smoke runs without an API key. It is intentionally
    simple and is NOT meant to produce high-quality answers -- swap in a
    real provider (LLM_PROVIDER=openai or LLM_PROVIDER=anthropic) for
    production-quality output.
    """

    name = "mock"

    _EMBED_DIM = 64

    def invoke_text(self, system: str, prompt: str) -> str:
        # Simple deterministic "summary" style response so downstream
        # synthesis / writing workers have something coherent to work with.
        snippet = prompt.strip().replace("\n", " ")[:280]
        return (
            f"[mock-llm response] Based on the input, here is a deterministic "
            f"summary: {snippet}"
        )

    def invoke_structured(self, system: str, prompt: str, schema: Type[T]) -> T:
        """
        Deterministic structured output.

        Since the mock has no real language understanding, it inspects the
        Pydantic schema's fields and produces plausible, deterministic
        placeholder content per field type. Task-decomposition and worker
        schemas are special-cased below so the orchestrator graph produces
        sensible subtasks/results even in mock mode.
        """
        from orchestrator.schemas import (
            Plan,
            SubTask,
            WorkerResult,
        )

        if schema is Plan:
            return _mock_plan(prompt)  # type: ignore[return-value]
        if schema is WorkerResult:
            return _mock_worker_result(prompt)  # type: ignore[return-value]
        if schema is SubTask:
            return _mock_subtask(prompt)  # type: ignore[return-value]

        # Generic fallback: fill required fields with deterministic stubs.
        return _mock_generic(schema, prompt)

    def embed(self, text: str) -> list[float]:
        return deterministic_embedding(text, dim=self._EMBED_DIM)


def deterministic_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Produce a deterministic pseudo-embedding for `text`.

    Not a real semantic embedding -- it is a hash-based feature vector that
    is nonetheless stable (same text -> same vector) and has enough locality
    (similar text -> loosely similar vector via shared token hashing) to
    exercise cosine-similarity code paths (semantic cache, long-term memory
    retrieval) deterministically and offline.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        tokens = [text.lower()]
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _mock_plan(prompt: str):
    from orchestrator.schemas import Plan, SubTask

    goal = _extract_goal(prompt)
    lowered = goal.lower()
    subtasks = []

    if any(k in lowered for k in ("calculate", "compute", "math", "+", "-", "*", "/", "sum")):
        subtasks.append(
            SubTask(
                id="task-1",
                description=f"Perform any required calculation for: {goal}",
                worker="calculator",
            )
        )
    if any(k in lowered for k in ("research", "find", "look up", "what is", "who is", "information", "facts")):
        subtasks.append(
            SubTask(
                id=f"task-{len(subtasks) + 1}",
                description=f"Research relevant facts for: {goal}",
                worker="research",
            )
        )
    if not subtasks:
        subtasks.append(
            SubTask(
                id="task-1",
                description=f"Research relevant background for: {goal}",
                worker="research",
            )
        )
    subtasks.append(
        SubTask(
            id=f"task-{len(subtasks) + 1}",
            description=f"Write a clear final answer for: {goal}",
            worker="writer",
        )
    )
    return Plan(goal=goal, subtasks=subtasks, reasoning="Deterministic mock decomposition.")


def _mock_subtask(prompt: str):
    from orchestrator.schemas import SubTask

    goal = _extract_goal(prompt)
    return SubTask(id="task-1", description=f"Re-decomposed subtask for: {goal}", worker="research")


def _mock_worker_result(prompt: str):
    from orchestrator.schemas import WorkerResult

    snippet = prompt.strip().replace("\n", " ")[:200]
    return WorkerResult(
        output=f"[mock-worker output] {snippet}",
        confidence=0.85,
        notes="Deterministic mock worker result.",
    )


def _mock_generic(schema: Type[T], prompt: str) -> T:
    field_values: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        annotation = field.annotation
        if annotation is str:
            field_values[field_name] = f"mock-{field_name}"
        elif annotation is float:
            field_values[field_name] = 0.75
        elif annotation is int:
            field_values[field_name] = 1
        elif annotation is bool:
            field_values[field_name] = True
        elif annotation is list or getattr(annotation, "__origin__", None) is list:
            field_values[field_name] = []
        elif field.default is not None:
            field_values[field_name] = field.default
    return schema(**field_values)  # type: ignore[arg-type]


def _extract_goal(prompt: str) -> str:
    """Best-effort extraction of the user's goal text out of a prompt string."""
    marker = "Goal:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip().splitlines()[0].strip()
    return prompt.strip().splitlines()[0].strip()[:200] if prompt.strip() else "unknown goal"


# ---------------------------------------------------------------------------
# Real backends -- OpenAI / Anthropic, via LangChain chat model wrappers.
# ---------------------------------------------------------------------------


class OpenAILLMClient(LLMClient):
    """Real OpenAI-backed client using langchain-openai. Requires OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        self._chat = ChatOpenAI(model=model, temperature=0)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def invoke_text(self, system: str, prompt: str) -> str:
        messages = [("system", system), ("human", prompt)]
        result = self._chat.invoke(messages)
        return str(result.content)

    def invoke_structured(self, system: str, prompt: str, schema: Type[T]) -> T:
        structured_llm = self._chat.with_structured_output(schema)
        messages = [("system", system), ("human", prompt)]
        result = structured_llm.invoke(messages)
        if isinstance(result, dict):
            return schema(**result)
        return result  # type: ignore[return-value]

    def embed(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


class AnthropicLLMClient(LLMClient):
    """Real Anthropic-backed client using langchain-anthropic. Requires ANTHROPIC_API_KEY.

    Structured output is obtained via tool-calling JSON mode
    (`with_structured_output`, which LangChain implements for Anthropic
    models using forced tool use), not via a hand-rolled prompt.

    Anthropic does not offer a first-party embeddings endpoint, so this
    backend falls back to the same deterministic hash-based embedding used
    by the mock client. This keeps the vector-store code path provider
    agnostic; swap in a dedicated embeddings provider for production if
    real semantic embeddings are required.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        from langchain_anthropic import ChatAnthropic

        self._chat = ChatAnthropic(model_name=model, temperature=0, max_tokens_to_sample=2048)

    def invoke_text(self, system: str, prompt: str) -> str:
        messages = [("system", system), ("human", prompt)]
        result = self._chat.invoke(messages)
        return str(result.content)

    def invoke_structured(self, system: str, prompt: str, schema: Type[T]) -> T:
        structured_llm = self._chat.with_structured_output(schema)
        messages = [("system", system), ("human", prompt)]
        result = structured_llm.invoke(messages)
        if isinstance(result, dict):
            return schema(**result)
        return result  # type: ignore[return-value]

    def embed(self, text: str) -> list[float]:
        # No native Anthropic embeddings API -- deterministic fallback.
        return deterministic_embedding(text, dim=64)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_client_singleton: Optional[LLMClient] = None


def get_llm_client(force_provider: Optional[str] = None) -> LLMClient:
    """
    Build (and cache) the LLM client selected by the LLM_PROVIDER env var.

    Resolution order:
      1. `force_provider` argument, if given.
      2. LLM_PROVIDER env var ("openai" | "anthropic").
      3. If the relevant API key is missing, or provider is unset/unknown,
         fall back to MockLLMClient.
    """
    global _client_singleton
    if force_provider is None and _client_singleton is not None:
        return _client_singleton

    provider = (force_provider or os.getenv("LLM_PROVIDER", "")).strip().lower()

    client: LLMClient
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAILLMClient(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        except Exception:
            client = MockLLMClient()
    elif provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            client = AnthropicLLMClient(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))
        except Exception:
            client = MockLLMClient()
    else:
        client = MockLLMClient()

    if force_provider is None:
        _client_singleton = client
    return client


def reset_llm_client_cache() -> None:
    """Test helper: clear the memoized client singleton."""
    global _client_singleton
    _client_singleton = None
