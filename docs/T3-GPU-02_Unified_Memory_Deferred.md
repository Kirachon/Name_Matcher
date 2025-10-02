# T3-GPU-02: Unified Memory for Simplified Transfers — Deferred

Date: 2025-10-01
Status: Deferred (high risk, unclear benefit for our access pattern)
Risk: High
Priority: P3 (Optional)

## Problem Statement
Replace explicit host↔device copies and manual memory management with CUDA Unified Memory to simplify code and potentially improve overlap.

## Design Summary (Would-Be)
- Replace device buffers with unified allocations; rely on on-demand paging between host and device.

## Correctness and Performance Considerations
- Semantics unchanged but performance is workload-dependent; random access can thrash pages and reduce performance.
- Our tiled, explicit-copy design is predictable and already efficient post Phase 1/2.

## Risks
- Requires Pascal+ GPUs and driver support; may degrade performance by up to 10% on some patterns.
- Complex to validate across divergent GPUs/environments.

## Decision
- Defer. Maintain current explicit-copy design with stream overlap and shared-memory optimizations.

## Validation
- No code changes; 32/32 tests pass and release build OK.

## Revisit When
- Nsight profiling shows significant memcpy wait or poor overlap despite pinned memory.
- Strong need to simplify memory management overrides predictable performance.

