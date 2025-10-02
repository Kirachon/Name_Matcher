# T2-MEM-02: Arena Allocator for MatchPair — Deferred (Partial)

Date: 2025-10-01
Status: Deferred (no API-safe gain); Partial micro-optimizations retained
Priority: P2

## Rationale
- The public struct `MatchPair` exposes `matched_fields: Vec<String>` (src/matching/mod.rs:547). Changing it to a different collection (e.g., `SmallVec`) would be a breaking public API change.
- Streaming paths pass owned `MatchPair` values to callbacks. Allocating them from an arena (e.g., `bumpalo`) would not remove the owned allocations we must return; without changing field types (to references/lifetimes), an arena yields negligible benefit.
- Therefore, the API-compatible subset (e.g., pre-sizing the Vec) offers only minor wins and is already covered by existing pre-allocation practices.

## Decision
- Defer introducing `smallvec` and `bumpalo` to avoid public API changes and added complexity with minimal measurable benefit.
- Retain safe micro-optimizations (pre-sizing vectors where applicable), already present from Phase 1 (T1-CPU-02).

## Validation
- No code changes required for this item. Full test suite already passes (32/32).

