"""
Token and wall-clock budgets for one orchestrator run.

The retry cap (MAX_REPLANS) bounds how many times the graph loops; a budget
also bounds what a run may cost and how long it may take. A fraction of each
budget is held back as a synthesis reserve, so a run that hits its budget
still ends with a synthesized answer from the results it has, rather than an
error.

Enforcement happens between steps: before each worker attempt and before a
replan (using the average cost of attempts so far as the projection). A call
that is already in flight cannot be interrupted; for real providers, set
LLM_REQUEST_TIMEOUT so a single slow call can't overshoot the time budget
by much.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agentic_memory.manager import MemoryManager


# Indirection so tests can drive the clock without patching time.monotonic globally.
_clock = time.monotonic


def _env_number(name: str, cast):
    value = os.getenv(name, "").strip()
    return cast(value) if value else None


@dataclass
class Budget:
    """Limits for one run. None means unlimited."""

    max_tokens: Optional[int] = None
    max_seconds: Optional[float] = None
    synthesis_reserve: float = 0.15

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if not 0.0 <= self.synthesis_reserve < 1.0:
            raise ValueError("synthesis_reserve must be in [0, 1)")

    @classmethod
    def from_env(cls) -> "Budget":
        """Read BUDGET_MAX_TOKENS / BUDGET_MAX_SECONDS (unset means unlimited)."""
        return cls(
            max_tokens=_env_number("BUDGET_MAX_TOKENS", int),
            max_seconds=_env_number("BUDGET_MAX_SECONDS", float),
        )

    @property
    def unlimited(self) -> bool:
        return self.max_tokens is None and self.max_seconds is None


@dataclass
class BudgetTracker:
    """Measures one run's spend against its Budget."""

    budget: Budget
    memory: "MemoryManager"
    _start_tokens: int = field(init=False)
    _start_time: float = field(init=False)

    def __post_init__(self) -> None:
        self._start_tokens = self.memory.total_tokens
        self._start_time = _clock()

    @property
    def tokens_used(self) -> int:
        return self.memory.total_tokens - self._start_tokens

    @property
    def seconds_used(self) -> float:
        return _clock() - self._start_time

    def _work_limit(self, limit: Optional[float]) -> Optional[float]:
        """The part of a limit available before the synthesis reserve."""
        return None if limit is None else limit * (1.0 - self.budget.synthesis_reserve)

    def exhausted(self) -> Optional[str]:
        """'token_budget' or 'time_budget' once work has used up its share, else None."""
        token_limit = self._work_limit(self.budget.max_tokens)
        if token_limit is not None and self.tokens_used >= token_limit:
            return "token_budget"
        time_limit = self._work_limit(self.budget.max_seconds)
        if time_limit is not None and self.seconds_used >= time_limit:
            return "time_budget"
        return None

    def can_afford(self, projected_tokens: float, projected_seconds: float) -> Optional[str]:
        """None if the projected extra work fits the remaining budget, else the limiting reason."""
        token_limit = self._work_limit(self.budget.max_tokens)
        if token_limit is not None and self.tokens_used + projected_tokens > token_limit:
            return "token_budget"
        time_limit = self._work_limit(self.budget.max_seconds)
        if time_limit is not None and self.seconds_used + projected_seconds > time_limit:
            return "time_budget"
        return None

    def snapshot(self) -> dict:
        return {
            "tokens_used": self.tokens_used,
            "seconds_used": round(self.seconds_used, 3),
            "max_tokens": self.budget.max_tokens,
            "max_seconds": self.budget.max_seconds,
        }
