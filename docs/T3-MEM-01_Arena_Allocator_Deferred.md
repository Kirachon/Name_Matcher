# T3-MEM-01: Arena Allocator (System-wide) — Deferred

Date: 2025-10-01
Status: Deferred (public API and lifetime complexity)
Risk: High
Priority: P3 (Optional)

## Problem Statement
Adopt an arena/bump allocator for match artifacts and frequently-created temporary objects to reduce allocator overhead and fragmentation.

## Current State
- Public `MatchPair` exposes owned `Vec<String>` (API commitment).
- Streaming callbacks and serialization require owned data with `'static` lifetimes today.

## Risks
- Would require pervasive lifetime plumbing or API changes to hold arena-backed references.
- Potential use-after-free if arenas outlive referenced data incorrectly.

## Decision
- Defer. Maintain owned allocations in public structs; use targeted pooling where safe.

## Revisit When
- A future major version allows API changes, or profiling shows allocator hot spots dominating runtime.

