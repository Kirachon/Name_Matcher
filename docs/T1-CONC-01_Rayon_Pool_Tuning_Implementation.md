# T1-CONC-01: Rayon Thread Pool Tuning — Implementation Report

Date: 2025-09-30
Status: ✅ Complete
Priority: P0

## Summary
Adjusted dedicated Rayon pool reservation policy to maximize parallelism while retaining a small headroom on high-core machines.

## Changes
- File: src/matching/rayon_pool.rs
- Logic changed:
  - Before: reserve 1–2 cores (`cores > 8 ? 2 : 1`)
  - After: reserve at most 1 core on >16-core machines; otherwise reserve 0
    - `let reserved = if cores > 16 { 1 } else { 0 };`

## Validation
- cargo test: 32/32 passed
- cargo build --release: success

## Expected Impact
- 5–10% higher CPU utilization on many-core systems
- Slight throughput improvement (5–8%) in CPU-heavy stages (normalization, CPU post-filter)

## Notes
- Environment override `NAME_MATCHER_RAYON_THREADS` remains supported and takes precedence

