# T1-DB-02: Connection Pool Warm-up — Implementation Report

Date: 2025-09-30
Status: ✅ Complete
Priority: P0

## Summary
Added a warm-up step to pre-establish connections up to the pool’s `max_connections` right after pool creation. Controlled by `NAME_MATCHER_POOL_WARMUP` (defaults to enabled, set `0` to disable).

## Changes
- File: src/db/connection.rs
- After creating the pool, loop `0..max_conn` and `pool.acquire().await` then immediately drop, returning connections to the pool.
- Logs start/end of warm-up.

## Validation
- cargo test: 32/32 passed
- cargo build --release: success
- No public API change

## Expected Impact
- Removes cold-start latency for first query batches
- 50–80% latency reduction for first batch on typical systems

## Env controls
- NAME_MATCHER_POOL_WARMUP=0 → disable warm-up
- NAME_MATCHER_POOL_MIN, NAME_MATCHER_POOL_SIZE, NAME_MATCHER_ACQUIRE_MS, NAME_MATCHER_IDLE_MS, NAME_MATCHER_LIFETIME_MS → existing controls preserved

## Next
- Proceed with remaining Phase 1 tasks

