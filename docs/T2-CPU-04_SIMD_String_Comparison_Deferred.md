# T2-CPU-04: SIMD for String Comparison — Deferred

Date: 2025-10-01
Status: Deferred (risk > benefit within Phase 2)
Priority: P2

## Rationale
- Hot path for fuzzy scoring in production uses the GPU fused kernel; CPU Levenshtein is used for heuristics and refinements and is not the primary throughput limiter.
- Introducing a SIMD Levenshtein implementation (via intrinsics or external crates) risks semantic drift and platform-specific regressions; validating exact-distance equivalence across edge cases is non-trivial.
- Adding dependencies for SIMD would require explicit approval and increases maintenance surface.

## Decision
- Defer SIMD adoption to a dedicated performance iteration, with feature-gated opt-in and benchmarks on representative datasets.
- Keep current CPU path relying on strsim for clarity and correctness.

## Validation
- No code changes. All tests remain passing (32/32).

