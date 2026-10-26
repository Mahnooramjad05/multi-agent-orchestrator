"""
Worker implementations.

Three distinct specialties, each returning a structured WorkerResult with a
self-reported confidence score in [0, 1]:

  - research worker   : keyword-overlap retrieval over a local JSON doc
                         store (data/docs.json), then LLM-synthesized answer
                         grounded in the retrieved passages.
  - calculator worker : extracts an arithmetic expression from the subtask
                         description and evaluates it via orchestrator.safe_eval
                         (AST-whitelisted -- never raw eval()/exec() on LLM
                         or user-controlled text).
  - writer worker      : LLM-based synthesis/writing over the accumulated
                         context (used both as a subtask worker and, from
                         the graph, as the final synthesis step).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.safe_eval import UnsafeExpressionError, safe_eval
from orchestrator.schemas import SubTask, WorkerResult

if TYPE_CHECKING:
    from agentic_memory.manager import MemoryManager

DOCS_PATH = Path("data") / "docs.json"

_ARITHMETIC_PATTERN = re.compile(r"[-+*/(). \d]{3,}")


def _load_docs() -> list[dict]:
    if not DOCS_PATH.exists():
        return []
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_DOCS_CACHE: list[dict] | None = None


def _docs() -> list[dict]:
    global _DOCS_CACHE
    if _DOCS_CACHE is None:
        _DOCS_CACHE = _load_docs()
    return _DOCS_CACHE


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_documents(query: str, top_k: int = 2) -> list[dict]:
    """Simple keyword-overlap retrieval over the local document store."""
    query_tokens = _tokenize(query)
    scored = []
    for doc in _docs():
        doc_tokens = _tokenize(doc["title"] + " " + doc["text"])
        overlap = len(query_tokens & doc_tokens)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def research_worker(subtask: SubTask, memory: "MemoryManager") -> WorkerResult:
    """Retrieval-augmented research worker over a local document store."""
    retrieved = retrieve_documents(subtask.description)

    if not retrieved:
        return WorkerResult(
            output=f"No relevant local documents found for: {subtask.description}",
            confidence=0.2,
            notes="Empty retrieval set; low confidence by design.",
        )

    context = "\n\n".join(f"[{doc['title']}]\n{doc['text']}" for doc in retrieved)
    system = (
        "You are a research assistant. Answer the user's request using ONLY the "
        "provided context passages. Be concise and factual."
    )
    prompt = f"Context:\n{context}\n\nRequest: {subtask.description}\n\nAnswer:"
    answer = memory.cached_invoke_text(system, prompt)

    # Confidence heuristic: scales with how many docs were retrieved and how
    # much term overlap they had with the query (proxy for grounding quality).
    confidence = min(0.95, 0.55 + 0.15 * len(retrieved))
    return WorkerResult(
        output=answer,
        confidence=confidence,
        notes=f"Retrieved {len(retrieved)} document(s): "
        + ", ".join(doc["title"] for doc in retrieved),
    )


def calculator_worker(subtask: SubTask, memory: "MemoryManager") -> WorkerResult:
    """
    Safe calculation worker.

    Extracts a candidate arithmetic expression from the subtask description
    and evaluates it via the AST-whitelisted safe_eval -- never via raw
    eval()/exec() on LLM or user-controlled text. If no safe expression can
    be extracted, reports low confidence rather than guessing.
    """
    candidates = _ARITHMETIC_PATTERN.findall(subtask.description)
    candidates = [c.strip() for c in candidates if any(ch.isdigit() for ch in c)]
    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        try:
            result = safe_eval(candidate)
            return WorkerResult(
                output=f"{candidate.strip()} = {result}",
                confidence=0.97,
                notes="Evaluated via AST-whitelisted safe_eval (no eval()/exec()).",
            )
        except UnsafeExpressionError:
            continue

    return WorkerResult(
        output=f"Could not safely extract a valid arithmetic expression from: "
        f"{subtask.description!r}",
        confidence=0.15,
        notes="No candidate substring parsed as a safe arithmetic expression.",
    )


def writer_worker(subtask: SubTask, memory: "MemoryManager", context: str = "") -> WorkerResult:
    """Writing/synthesis worker: drafts prose for a subtask given optional context."""
    system = (
        "You are a clear, concise technical writer. Produce a well-organized "
        "final answer for the user's request, incorporating any provided context."
    )
    prompt = f"Context so far:\n{context}\n\nWriting task: {subtask.description}\n\nDraft:"
    answer = memory.cached_invoke_text(system, prompt)
    confidence = 0.9 if context.strip() else 0.6
    return WorkerResult(
        output=answer,
        confidence=confidence,
        notes="Writer worker used accumulated context." if context.strip() else "Writer worker had no upstream context.",
    )


WORKER_REGISTRY = {
    "research": research_worker,
    "calculator": calculator_worker,
    "writer": writer_worker,
}

# Whether a worker changes anything outside the run (emails, DB writes, paid
# API calls). None of the current workers do, but the orchestrator checks
# idempotency keys for every worker so adding one with side effects is safe.
WORKER_SIDE_EFFECTS = {
    "research": False,
    "calculator": False,
    "writer": False,
}
