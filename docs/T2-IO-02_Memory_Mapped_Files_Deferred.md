# T2-IO-02: Memory-Mapped Files (CSV) — Deferred

Date: 2025-10-01
Status: Deferred (not a current hot path)
Priority: P2

## Research Summary
- The application’s hot path is MySQL streaming (see src/matching/mod.rs stream_*, src/engine/db_pipeline.rs).
- CSV usage is export-oriented (src/export/csv_export.rs). There is no CSV input reader in the codebase.
- GUI supports loading and saving configuration from CSV (a tiny, line-by-line setting import), not bulk data ingest.

## Decision
- No code changes. Memory-mapped CSV reading would not impact the current bottlenecks.
- If an offline CSV ingest mode is introduced later, revisit with a prototype behind a feature flag (memmap2).

## Prototype Plan (future, if needed)
- Add feature flag `csv_mmap_reader` (default off).
- Implement a zero-copy line scanner over memmap2 with SIMD newline search (e.g., bytecount).
- Integrate with an incremental CSV parser to feed Person rows into the existing matching engine.

## Validation
- No code changes. Existing tests remain passing (32/32). Release build succeeds.

