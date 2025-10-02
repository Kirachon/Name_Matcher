# T2-DB-03: Prepared Statement Caching — Deferred

Date: 2025-10-01
Status: Deferred (documentation-only)
Priority: P2

## Research Summary
- Streaming queries are implemented in src/db/schema.rs:
  - get_person_count, fetch_person_rows_chunk, get_person_count_where, get_person_rows_where, fetch_person_rows_chunk_where
- Table names are dynamic (interpolated with backticks), while WHERE, LIMIT, and OFFSET use bound parameters.
- With dynamic table identifiers, server-side prepared statement reuse has limited benefit because each table name change produces a distinct SQL statement signature.
- sqlx builds and executes textual queries; parameters are bound for values but identifiers cannot be parameterized.

## Current Patterns (good)
- Identifiers validated via validate_ident(table) to avoid injection.
- Binds used for WHERE values and for LIMIT/OFFSET, enabling statement parameterization for values.

## Opportunity & Constraints
- Server-side prepared-statement caches (where available) are keyed by the full SQL string; dynamic table names prevent reuse across tables.
- Client-side statement caching in sqlx for MySQL is not generally configurable for dynamic identifiers; the benefit would be marginal due to changing SQL strings.

## Decision
- No code changes at this time. Maintain current safe patterns (identifier validation + value binds).
- Document best practices for DB usage.

## Best Practices (Recommended)
- Avoid unnecessary switching between many table names within one run (reduces distinct SQL texts).
- Keep SELECT column lists stable to maximize cache hits where possible.
- Ensure proper indexes exist for partition predicates (see T2-DB-04 Index Optimization Guide).
- Keep pool size reasonably high (we already default to up to 64 conns) and warm-up the pool (T1-DB-02).

## Validation
- No code changes. Existing tests remain passing (32/32). Release build succeeds.

