# T3-GPU-01: Persistent Kernel — Deferred

Date: 2025-10-01
Status: Deferred (architectural change; unclear benefit with current fusion)
Risk: High
Priority: P3 (Optional)

## Problem Statement
Run a persistent GPU kernel that manages its own work queue to amortize launch overhead and improve scheduling.

## Current State
- We already use a fused kernel for fuzzy metrics (T1-GPU-01). Launch overhead is low relative to compute after fusion.

## Risks
- Complex device-side queueing, synchronization, and host–device coordination.
- Debuggability and deadlock risk increase significantly.

## Decision
- Defer. Revisit only if profiling shows >20% time in kernel launch/dispatch despite fusion and overlap.

