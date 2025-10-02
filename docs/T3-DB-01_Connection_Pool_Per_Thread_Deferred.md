# T3-DB-01: Connection Pool Per Thread — Deferred

Date: 2025-10-01
Status: Deferred (complexity vs. benefit; DB limits)
Risk: High
Priority: P3 (Optional)

## Problem Statement
Provide a thread-local MySQL pool to eliminate shared-pool lock contention during high concurrency streaming.

## Current State
- Single global `MySqlPool` with warmup and larger size (up to 64 conns).
- Streaming is mostly sequential I/O bound per worker; limited evidence of pool contention.

## Risks
- Multiplying pool size by number of threads can exceed DB connection limits.
- Complex lifecycle management; risk of leaking or exhausting connections.

## Decision
- Defer. Maintain single shared pool with tuning; monitor pool metrics.

## Revisit When
- Metrics show frequent pool exhaustion/acquire timeouts even after tuning.
- DB has sufficient max connections to support per-thread pools safely.

