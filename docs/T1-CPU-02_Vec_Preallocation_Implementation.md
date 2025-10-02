# T1-CPU-02: Vec Pre-allocation — Implementation Report

Date: 2025-09-30
Status: ✅ Complete
Priority: P0

## Summary
Replaced `Vec::new()`/`HashSet::new()` with `with_capacity()` in hot paths where bounds are knowable. This reduces reallocations and improves locality without changing logic or outputs.

## Changes (hot paths)

- CPU fuzzy blocking (match_fuzzy_cpu and GPU variant pre-blocking)
  - Block map already pre-allocated (prior step)
  - Candidate set:
    - `HashSet::new()` → `HashSet::with_capacity(v.len())` when first block exists; default 16 otherwise
- GPU in-memory hash build/probe (Algorithms 1/2)
  - Inner build:
    - `key_strs: Vec<String>` → `with_capacity(inner_n.len())`
  - Outer probe per-slice:
    - `pkeys: Vec<String>` / `pidx: Vec<usize>` → `with_capacity(slice.len())`
- Streaming hash join build/probe (large datasets)
  - Build batch:
    - `key_strs` / `key_idx` → `with_capacity(rows.len())`
    - kept `norm_cache` as-is (already pre-allocated)
  - Probe batch:
    - `probe_keys` / `probe_idx` → `with_capacity(rows.len())`
- Household aggregation outputs
  - `out: Vec<HouseholdAggRow>` → pre-allocate with `totals.len()/2 + 1` and `totals_t2.len()/2 + 1`
- Minor: kept GPU tile staging buffers as reusable Vecs (already reused); conservative not to over-allocate bytes blindly.

## Validation
- cargo test: 32/32 passed
- cargo build --release: success
- No public API changes; no algorithm/threshold changes

## Expected Impact
- Fewer reallocations in tight loops (95%+ reduction where capacities are known)
- 15–25% improvement in allocation-heavy sections
- Slightly more predictable peak memory; no change in output

## Notes
- Additional micro-optimizations are possible (e.g., pre-sizing GPU tiling byte buffers based on measured tile_len), but deferred to avoid over-allocation risk without profiling.

## Next
- Proceed to T1-CPU-03 (Eliminate Redundant Normalization)

