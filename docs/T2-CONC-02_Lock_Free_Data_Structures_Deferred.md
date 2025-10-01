# T2-CONC-02: Lock-Free Data Structures — Deferred

Date: 2025-10-01
Status: Deferred (no contention hotspots identified)
Priority: P2

## Research Summary
- Mutex usage is primarily in the metrics subsystem (src/metrics/mod.rs) to guard a global MetricsCollector and counters.
- Matching/data paths use Rayon (fork-join), and we recently implemented parallel blocking via per-thread HashMaps + reduce (lock-free in the hot path).
- No evidence of contended Mutex in candidate generation, scoring, or classification. Aggregation is sequential or per-worker local.

## Decision
- Defer code changes. The expected benefit (10–20%) applies to highly contended shared queues or maps; our hot paths avoid such contention by design.

## Guidance
- If future profiling shows contention, consider:
  - Adopt crossbeam::SegQueue or lock-free MPSC for producer/consumer pipelines.
  - Use sharded structures (DashMap) cautiously; ensure determinism where needed.
  - Keep per-worker local buffers + batched merges (retain current pattern where possible).

## Validation
- No code changes. All tests continue to pass (32/32). Release build succeeds.

