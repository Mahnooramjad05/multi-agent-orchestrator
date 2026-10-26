Multi-Agent Orchestration System (Orchestrator-Workers) with agentic memory, semantic caching, and an agentic-eval harness.

## Overview

This project is a runnable implementation of an orchestrator-workers multi-agent architecture. An orchestrator agent decomposes an incoming goal into a structured set of subtasks and delegates each one to a specialized worker agent (research, calculation, or writing/synthesis). Worker outputs carry a self-reported confidence score; if a worker's confidence falls below a threshold, the orchestrator dynamically replans that subtask (bounded by a retry cap, to avoid infinite loops) rather than failing the whole run. The system is backed by three distinct forms of agentic memory (short-term, long-term, and episodic) plus an in-process semantic cache that skips repeated LLM calls, and it ships with an agentic-eval harness that scores runs on task completion and efficiency rather than lexical overlap. The whole system is exposed through a small FastAPI service and can run in a container via the included Dockerfile.

## What's New in 1.1

Three improvements prompted by community feedback on the first release (full list in `CHANGELOG.md`):

- **Calibrated confidence.** Worker confidence scores are heuristics. `python -m eval.calibrate` measures how often attempts at each score actually succeed on the benchmark, fits a per-worker calibration, and the graph applies the replanning threshold to the calibrated value. On the 30-goal benchmark in mock mode, expected calibration error drops from 0.341 to 0.023 (in-sample). Most of that gap comes from the writer, which reports 0.9 whenever it has upstream context but only produces a complete answer about a quarter of the time.
- **Traceable, idempotent retries.** Every worker attempt is logged with its own idempotency key, status, confidence, tokens and latency. Resuming a run with the same `run_id` reuses completed attempts instead of re-running them.
- **Token and wall-clock budgets.** Runs can be capped by tokens and seconds, not just retries. Replans only happen when their projected cost fits, and a reserve guarantees the run still returns an answer. Every run reports a `stop_reason`.

## Key Features

- LangGraph-based orchestrator graph with an orchestrator node, three distinct worker nodes, a synthesis node, and a conditional replanning edge.
- Structured task decomposition via a Pydantic schema and an LLM call (JSON mode / structured output), producing a typed `Plan` of `SubTask` objects.
- Three specialized worker types:
  - Research worker: keyword-overlap retrieval over a local JSON document store, followed by an LLM-grounded answer.
  - Calculator/code worker: extracts and evaluates arithmetic expressions using an AST-whitelisted safe evaluator (`orchestrator/safe_eval.py`) -- it never calls `eval()`/`exec()` on LLM or user-controlled text.
  - Writer worker: LLM-based synthesis/drafting over accumulated context, also used as the final synthesis step.
- Worker confidence scores (0-1), optionally calibrated against eval outcomes, and a conditional LangGraph edge that triggers replanning when confidence is below a threshold. Replanning is capped by a maximum number of replans and by optional token and wall-clock budgets.
- Per-attempt logging with idempotency keys: every retry is a separate record in the episodic log, and completed attempts are never executed twice for the same run.
- Agentic memory with three real, distinct components:
  - Short-term: an in-process conversation buffer with a max-turn window, scoped per session.
  - Long-term: embedded facts/summaries persisted to a Qdrant vector store, keyed by session id.
  - Episodic: a structured JSON Lines log of full past runs (goal, plan, outcome, score), retrievable by similarity over the goal embedding.
- An in-process semantic cache (numpy cosine similarity over cached prompt-embedding -> response pairs) wired into the LLM-call wrapper so repeated or paraphrased prompts skip the LLM call.
- An agentic-eval harness that runs a small benchmark set of goals through the graph and scores each run on task completion (are the expected key facts present, checked via containment and, as a fallback, embedding similarity -- not exact string match) and efficiency (LLM calls, worker invocations, replans), producing both a printed report and a JSON report.
- A pluggable LLM client supporting OpenAI, Anthropic, and a deterministic mock backend, selected purely via environment variables.
- A FastAPI HTTP layer (`POST /run`, `GET /health`) and a Dockerfile for containerized deployment.
- A pytest suite covering task decomposition, the safe-expression evaluator, the semantic cache, the eval harness, the memory components, the orchestrator graph, and the API -- all running against the mock LLM client with no network calls.

## Tech Stack

- Python 3.11+
- LangGraph (the orchestrator-workers graph and conditional replanning edge)
- LangChain (`langchain-openai`, `langchain-anthropic` chat model wrappers)
- OpenAI and Anthropic LLM clients, pluggable via an `LLM_PROVIDER` environment variable
- Qdrant (`qdrant-client`), used in local/in-memory mode by default
- NumPy (cosine similarity for the semantic cache and embedding-based retrieval)
- Pydantic (structured LLM output schemas)
- FastAPI + Uvicorn (HTTP API)
- pytest + httpx (tests, including FastAPI's `TestClient`)
- Docker (containerized deployment)

## Architecture

**Orchestrator graph (LangGraph).** The graph is defined in `orchestrator/graph.py` with the following structure:

```
START -> decompose -> dispatch_workers -> assess --(replan)--> replan -> dispatch_workers (loop)
                                             |
                                             +--(synthesize)--> synthesize -> END
```

- `decompose`: the orchestrator node. Sends the goal to the LLM with a Pydantic `Plan` schema (via `MemoryManager.cached_invoke_structured`), producing an ordered list of `SubTask` objects, each assigned to exactly one worker specialty.
- `dispatch_workers`: runs every pending subtask through its assigned worker function (`research_worker`, `calculator_worker`, or `writer_worker` in `orchestrator/workers.py`), collecting a structured `WorkerResult` (output, confidence, notes) for each. Already-successful subtasks are skipped on subsequent passes.
  Each attempt gets an idempotency key (a hash of run id, slot id, attempt number, worker and description) and is recorded as an `AttemptRecord`. If the key already completed, the stored result is reused. A worker that raises an exception becomes a zero-confidence result that can be replanned. If a calibration file is present, each raw confidence is mapped to a calibrated one before the threshold is applied.
- `assess` (followed by a conditional edge): records why the run stops. It routes to `replan` only if some subtask's confidence is below the threshold (default 0.5), fewer than the maximum replans (default 2) have happened, and the projected cost of replanning fits the remaining token and time budget. Otherwise it routes to `synthesize` with `stop_reason` set to `all_confident`, `max_replans`, `token_budget` or `time_budget`.
- `replan`: re-decomposes only the failed subtask(s) into a fresh, more specific `SubTask` (via another structured LLM call), keeping the same slot id, then loops back to `dispatch_workers`. The replan counter is capped so a persistently low-confidence subtask cannot loop forever.
- `synthesize`: merges all worker outputs into one coherent final answer via an LLM call, and is also the last node before `END`.

**Memory.** `agentic_memory/manager.py` defines `MemoryManager`, the single facade the graph and workers use for all memory and LLM access:

- `ShortTermMemory` (`agentic_memory/short_term.py`): a per-session, fixed-size deque of recent conversation turns.
- `LongTermMemory` (`agentic_memory/long_term.py`): wraps a Qdrant collection. Facts are embedded via the active LLM client's `embed()` method and upserted with a `session_id` payload field; retrieval is a filtered vector search scoped to that session.
- `EpisodicMemory` (`agentic_memory/episodic.py`): appends one JSON line per completed run (goal, plan, outcome, score, efficiency counters, token usage, stop reason and the full list of worker attempts) to `data/episodic_log.jsonl`, and retrieves similar past episodes by embedding the query goal and comparing cosine similarity against each logged goal's embedding. Older log lines without the newer fields still load.
- `SemanticCache` (`agentic_memory/semantic_cache.py`): an in-process, numpy-backed cache of `(prompt embedding, response)` pairs. `MemoryManager.cached_invoke_text()` is the wrapper every text-generating call in the graph goes through: it embeds the prompt, looks for a cached entry above a cosine-similarity threshold, and only calls the underlying LLM client on a miss.

**Semantic caching** is therefore not a separate bolt-on -- it sits directly in the call path of every worker and the synthesis/decomposition steps that use `cached_invoke_text`, so a repeated or paraphrased prompt anywhere in a run (or across an eval batch) skips the LLM call entirely.

**Eval harness.** `eval/harness.py` loads benchmark goals from `benchmarks/goals.yaml` (each with a goal string and a list of expected key facts), runs each goal through the full orchestrator graph, and scores it on two axes: task completion (fact containment, falling back to embedding cosine similarity rather than requiring an exact string match) and efficiency (a decay function over LLM calls, worker invocations, and replans). It also logs each run into episodic memory and prints a human-readable report in addition to writing a JSON report.

**Budgets.** `orchestrator/budget.py` defines `Budget(max_tokens, max_seconds)` and a `BudgetTracker` that measures a run's token use (from provider usage metadata, or estimated at ~4 characters per token in mock mode) and elapsed time. 15% of each budget is held back as a synthesis reserve. Budgets are checked between steps; a call already in flight can't be interrupted, so real clients also use a per-request timeout (`LLM_REQUEST_TIMEOUT`).

**Confidence calibration.** `orchestrator/calibration.py` fits per-worker reliability bins: raw scores are grouped into equal-width bins, and each bin's calibrated value is its observed success rate, shrunk toward the raw score with a Beta prior so sparse bins stay stable. `python -m eval.calibrate` produces the labels from benchmark outcomes (an attempt succeeds if its output contains one of the goal's expected facts; writer attempts use the goal's completion score), prints a reliability table with ECE and Brier score, and writes `data/calibration.json`. These labels are a proxy and the reported improvement is in-sample, so refit with a real provider and a larger benchmark before relying on it.

**LLM client layer.** `orchestrator/llm_client.py` defines an abstract `LLMClient` interface with three concrete implementations: `OpenAILLMClient`, `AnthropicLLMClient`, and `MockLLMClient`. `get_llm_client()` reads `LLM_PROVIDER` and the relevant API key from the environment and returns the matching client, falling back to the mock client whenever the provider is unset/unrecognized or its API key is missing.

## Setup / Installation

```bash
# from the project root
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# copy the example env file and fill in real values only if you want a real provider
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

By default (no `.env` changes needed), the project runs entirely offline against the deterministic mock LLM client and an in-memory Qdrant instance -- no API keys and no external services are required to install, test, or smoke-run it.

## Usage

**Run the API locally:**

```bash
uvicorn api.main:app --reload
```

**Check health / active provider:**

```bash
curl http://127.0.0.1:8000/health
```

**Run a goal through the orchestrator:**

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d "{\"goal\": \"Tell me about Acme Robotics and calculate 12 * 7\", \"session_id\": \"demo\"}"
```

The response includes the final synthesized answer plus a trace of every subtask, which worker handled it, its confidence score, whether it was replanned, and the raw event trace (decompose / worker_run / replan / synthesize events).

**Run a goal with a budget:**

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d "{\"goal\": \"Tell me about Acme Robotics\", \"max_tokens\": 4000, \"max_seconds\": 30}"
```

The response adds `run_id`, `stop_reason`, token counts, budget usage and the full list of worker `attempts`.

**Fit confidence calibration:**

```bash
python -m eval.calibrate
```

**Run the agentic-eval harness:**

```bash
python -m eval.harness
```

This prints a per-goal and summary report to the console and writes `eval_report.json` in the project root.

**Run the tests:**

```bash
pytest -v
```

All tests run against the mock LLM client (forced in `tests/conftest.py`), so the full suite requires no API keys and makes no network calls.

**Build and run with Docker:**

```bash
docker build -t multi-agent-orchestrator .
docker run -p 8000:8000 --env-file .env multi-agent-orchestrator
```

## Environment Variables

All variables are documented in `.env.example` (with empty placeholder values -- never commit real secrets):

- `LLM_PROVIDER`: selects the LLM backend. Set to `openai` or `anthropic` to use a real provider; leave blank/unset to use the deterministic mock client.
- `OPENAI_API_KEY`: your OpenAI API key. Only required when `LLM_PROVIDER=openai`.
- `OPENAI_MODEL`: OpenAI chat model id to use (defaults to `gpt-4o-mini` if unset).
- `ANTHROPIC_API_KEY`: your Anthropic API key. Only required when `LLM_PROVIDER=anthropic`.
- `ANTHROPIC_MODEL`: Anthropic chat model id to use (defaults to `claude-sonnet-5` if unset).
- `BUDGET_MAX_TOKENS` / `BUDGET_MAX_SECONDS`: default per-run budgets. Leave blank for unlimited.
- `LLM_REQUEST_TIMEOUT`: per-request timeout in seconds for real providers (default 60).
- `CALIBRATION_PATH`: fitted calibration file (default `data/calibration.json`; used automatically when present).
- `IDEMPOTENCY_DB`: optional SQLite path so completed attempts survive restarts (in-memory by default).
- `EPISODIC_LOG_PATH`: where episodic entries are appended (default `data/episodic_log.jsonl`).
- `QDRANT_URL`: URL of a real Qdrant server (e.g. `http://localhost:6333`). Leave blank to use `qdrant-client`'s local in-memory mode.

## Notes on Local/Mock Defaults and Moving to Production

The vector database runs in local/in-memory mode by default (`qdrant_client.QdrantClient(location=":memory:")`) so the entire project -- installation, tests, the eval harness, and a local API smoke run -- works without standing up any external Qdrant server. To use a real Qdrant deployment, set `QDRANT_URL` in `.env`; no application code changes are needed.

Similarly, the LLM layer defaults to a clearly labeled mock backend (`MockLLMClient`, in `orchestrator/llm_client.py`) whenever `LLM_PROVIDER` is unset or the corresponding API key is missing. This keeps the project fully runnable and testable with zero paid API usage. To go to production with a real model, set `LLM_PROVIDER=openai` (with `OPENAI_API_KEY`) or `LLM_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY`) in `.env` -- the orchestrator graph, workers, memory system, and API require no code changes to switch providers.
