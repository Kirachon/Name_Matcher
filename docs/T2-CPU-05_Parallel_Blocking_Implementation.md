# T2-CPU-05: Parallel Blocking — Implementation Report

Date: 2025-10-01
Status: Complete
Priority: P1

## Summary
Converted the construction of the blocking index (HashMap<BKey, Vec<usize>>) to a parallel fold/reduce using Rayon in both CPU and GPU fuzzy paths. This preserves correctness and public APIs while reducing wall-clock time for block building on multi-core CPUs.

## Design
- Data: n2 (normalized persons for table2)
- Key: BKey(year, first_initial, last_initial, soundex4(last_name))
- Map: per-thread local HashMap via `.fold(|| HashMap::with_capacity(256), |local, (j,p)| { ... })`
- Reduce: merge local maps with `.reduce(...)`, appending vectors per key

## Code excerpts
- CPU fuzzy path:
````rust
// Build blocks for table2 (parallelized with Rayon fold/reduce)
let block: HashMap<BKey, Vec<usize>> = n2
    .par_iter()
    .enumerate()
    .fold(
        || HashMap::<BKey, Vec<usize>>::with_capacity(256),
        |mut local, (j, p)| {
            let (Some(d), Some(fn_str), Some(ln_str)) = (p.birthdate.as_ref(), p.first_name.as_deref(), p.last_name.as_deref()) else { return local; };
            let year = d.year() as u16;
            let fi = fn_str.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let li = ln_str.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let sx = soundex4_ascii(ln_str);
            local.entry(BKey(year, fi, li, sx)).or_default().push(j);
            local
        },
    )
    .reduce(
        || HashMap::with_capacity(n2.len().saturating_mul(2)),
        |mut a, mut b| { for (k, mut v) in b.drain() { a.entry(k).or_default().append(&mut v); } a },
    );
````
- GPU fuzzy path: identical structure using `super::soundex4_ascii`.

## Correctness
- Same keys and candidate indices as sequential version.
- Deterministic semantics: insertion order within vectors is not relied upon by any logic.

## Performance
- Expected: ~cores× speedup (clamped by memory bandwidth and hashing cost) for the block construction phase.
- Overhead: minimal; merges happen once per worker in reduce stage.

## Validation
- Tests: All 32/32 tests pass (`cargo test --lib --features gpu,new_cli,new_engine -- --test-threads=1`).
- Build: Release build succeeds with GPU/new_cli/new_engine features.

## Rollback
- Change is localized to two loops; revert by restoring the prior sequential for-loop.

