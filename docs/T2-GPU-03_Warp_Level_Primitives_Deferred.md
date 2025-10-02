# T2-GPU-03: Warp-Level Primitives — Deferred

Date: 2025-10-01
Status: Deferred (no safe, measurable subset)
Priority: P2

## Rationale
The fused fuzzy kernel uses a one-thread-per-pair design where Levenshtein, Jaro, and Jaro-Winkler are computed within a single thread. There are no inter-thread reductions or scans; thus warp-collectives (e.g., `__shfl_down_sync`) provide no natural benefit without a fundamental redesign (one-warp-per-pair cooperative DP).

- Redesign Risk: High (synchronization, shared memory layout changes, occupancy trade-offs)
- Scope: Exceeds Phase 2 "medium effort" and jeopardizes stability
- Benefit Uncertain: Given 64-char cap and per-thread DP, synchronization overhead likely offsets gains

## Decision
- Defer kernel redesign. Keep current shared-memory per-thread DP which is already optimized and validated.
- Revisit in a dedicated GPU-kernel iteration with targeted benchmarks and feature gating if needed.

## Validation
No code changes. Existing GPU tests and all 32 unit tests remain passing.

