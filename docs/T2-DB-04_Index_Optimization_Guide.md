# T2-DB-04: Index Optimization Guide (MySQL)

Date: 2025-10-01
Status: Documentation (actionable DDL)
Priority: P2

## Goal
Reduce I/O and CPU during streaming and partitioned scans by aligning indexes with our WHERE clauses and access patterns.

## Workload Recap
- Streaming fetches by ORDER BY id LIMIT ? OFFSET ? (full table scan ordered by PK)
- Partitioned streaming uses two predicate families (src/util/partition.rs):
  1) Last initial: UPPER(LEFT(last_name,1)) = ?
  2) Birth-year ranges: YEAR(birthdate) BETWEEN ? AND ?

## Recommendation: Generated Columns + Indexes
MySQL doesn't index function expressions directly; use generated columns and index them.

1) Last initial
- Add persisted generated column last_initial and an index

```sql
ALTER TABLE `people`
  ADD COLUMN last_initial CHAR(1) GENERATED ALWAYS AS (UPPER(LEFT(`last_name`, 1))) STORED,
  ADD INDEX idx_last_initial (last_initial);
```

2) Birth year
- Add persisted generated column birth_year and an index

```sql
ALTER TABLE `people`
  ADD COLUMN birth_year INT GENERATED ALWAYS AS (YEAR(`birthdate`)) STORED,
  ADD INDEX idx_birth_year (birth_year);
```

3) Optional composite indexes
- For tighter ranges and secondary orderings, consider composite indexes that match typical filters and sort:

```sql
-- For queries that filter by birth_year and then scan by id
CREATE INDEX idx_birth_year_id ON `people` (birth_year, id);

-- For queries that filter by last_initial and then scan by id
CREATE INDEX idx_last_initial_id ON `people` (last_initial, id);
```

## Deterministic (A1/A2) Streaming Considerations
Our matching is done in application memory; DB isnt joining on name/date. However, keeping the access path ordered by id benefits LIMIT/OFFSET paging. Ensure `id` is PRIMARY KEY (clustered) for best locality.

## Multi-table Environments
Repeat the generated-column/index additions for each participating table (e.g., table1, table2). Use consistent names to simplify tooling.

## Validation Checklist
- EXPLAIN SELECT ... WHERE last_initial = 'A' ORDER BY id LIMIT 50000 OFFSET 0;
- EXPLAIN SELECT ... WHERE birth_year BETWEEN 1980 AND 1984 ORDER BY id LIMIT 50000 OFFSET 0;
- Confirm `Using index condition` and predictable low rows examined.

## Rollback
- If needed: DROP INDEX idx_last_initial, idx_last_initial_id, idx_birth_year, idx_birth_year_id; ALTER TABLE ... DROP COLUMN last_initial; DROP COLUMN birth_year;

## Notes
- Keep statistics updated (ANALYZE TABLE). 
- If tables are massive and updates frequent, evaluate partial or partitioned indexes by year.

