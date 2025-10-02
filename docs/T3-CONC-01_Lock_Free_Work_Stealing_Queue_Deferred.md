# T3-CONC-01: Lock-Free Work-Stealing Queue — Deferred

Date: 2025-10-01
Status: Deferred (Rayon already provides work-stealing; no hotspot identified)
Risk: High
Priority: P3 (Optional)

## Problem Statement
Introduce custom lock-free work-stealing queues to further reduce contention and improve scalability for CPU pipelines.

## Current State
- We use Rayon which implements efficient work-stealing.
- Our hot paths avoid shared mutable state via per-thread buffers + reduce (T2-CPU-05).

## Risks
- Replacing Rayon’s internals adds complexity and potential for subtle concurrency bugs.
- Determinism/reproducibility could regress.

## Decision
- Defer. Keep Rayon’s proven scheduler; investigate only if profiling shows scheduler overhead dominating.

## Revisit When
- Profiling indicates >10% time lost to scheduling/steal contention in production workloads.

