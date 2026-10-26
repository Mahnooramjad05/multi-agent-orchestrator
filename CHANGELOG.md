# Changelog

## 1.1.0

Three improvements prompted by community feedback on the original release:
the stop rule was only as good as the confidence scores feeding it, retries
weren't traceable, and the retry cap bounded loops but not cost or latency.

### Added

- **Per-attempt logging.** Every worker attempt is recorded as an
  `AttemptRecord` (run id, slot id, attempt number, status, raw and calibrated
  confidence, output, tokens, latency). Episodic log entries now carry the full
  attempt history, a `run_id` and a `stop_reason`, and `run_goal()` logs API
  runs too, not just eval runs.
- **Idempotency keys.** Each attempt is keyed by run, slot, attempt number,
  worker and description. Completed attempts are stored (in memory, or in
  SQLite via `IDEMPOTENCY_DB`) and reused when a run is resumed with the same
  `run_id`, so a worker never executes the same attempt twice. Workers declare
  a `side_effects` flag.
- **Token and wall-clock budgets.** `Budget(max_tokens=..., max_seconds=...)`
  via `run_goal()`, the `/run` request body, or `BUDGET_MAX_TOKENS` /
  `BUDGET_MAX_SECONDS`. Replans only happen if their projected cost fits; a
  15% synthesis reserve guarantees the run still returns an answer. Token
  usage is recorded per call (estimated in mock mode).
- **Confidence calibration.** `python -m eval.calibrate` labels each attempt
  from benchmark outcomes, fits per-worker reliability bins with Beta-prior
  smoothing, reports ECE and Brier score, and writes `data/calibration.json`,
  which the graph then uses for the replanning threshold.
- **Bigger benchmark.** 30 goals (from 9) and 16 documents (from 8),
  including goals whose answers are not in the document store.
- `stop_reason` on every run: `all_confident`, `max_replans`,
  `token_budget` or `time_budget`.

### Fixed

- The writer now sees results produced earlier in the same pass. Previously
  `dispatch_workers` built the shared context before any worker ran, so on
  the first pass the writer always received an empty context and reported
  0.6 confidence.

### Changed

- New `assess` node between `dispatch_workers` and the routing edge; it
  records why the run stopped.
- A worker that raises an exception now produces a zero-confidence result
  (status `error`) that is eligible for replanning, instead of failing the run.
- `EpisodicRecord.score` is optional (unscored runs outside the harness).
  Older log lines still load.
- The `/run` response adds `run_id`, `stop_reason`, `attempts`, token counts
  and budget usage.
