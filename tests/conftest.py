"""Shared pytest fixtures. Forces mock LLM mode so tests never hit the network."""

from __future__ import annotations

import os

import pytest

# Ensure no real provider is ever selected during the test suite, regardless
# of what's in the developer's shell environment or a stray .env file.
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["LLM_PROVIDER"] = "mock"

from agentic_memory.manager import MemoryManager  # noqa: E402
from orchestrator.llm_client import MockLLMClient, reset_llm_client_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_llm_client_cache():
    reset_llm_client_cache()
    yield
    reset_llm_client_cache()


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def memory_manager(mock_client: MockLLMClient) -> MemoryManager:
    return MemoryManager(llm_client=mock_client)
