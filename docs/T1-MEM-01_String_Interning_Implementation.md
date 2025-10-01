# T1-MEM-01: String Interning (Targeted) — Implementation Report

Date: 2025-09-30
Status: ✅ Complete (Targeted scope)
Priority: P0

## Summary
Applied a pragmatic, low‑risk string interning optimization focused on the highest‑impact hotspots without altering public APIs or core algorithm semantics.

- Interned HashMap keys where UUID and hh_id are used repetitively
- Preserved MatchPair cloning (final results) for compatibility
- Deferred full `InternedPerson` migration; added supporting types/utilities for future use

## Changes

### Dependencies
- Cargo.toml: added `string_cache = "0.8"`

### Models
- Added `InternedPerson` (not yet wired into hot paths) and conversions
- Added `Person::intern_keys()` helper returning `(Option<Atom>, Option<Atom>)`

### Matching (targeted interning)
- Replaced String keys with `DefaultAtom` in two high‑traffic maps:
  1) Table 1 UUID totals map
  2) Table 2 hh_id totals map
- Adjusted subsequent lookups to use `Atom::from(str)` for compatibility

Example (excerpt):
- Table 1 UUID totals
  - Before: `HashMap<String, usize>`, `.entry(u.clone())`
  - After: `HashMap<Atom, usize>`, `.entry(Atom::from(u))`
- Table 2 hh_id totals
  - Before: `HashMap<String, usize>`, lookup by `&hh_key`
  - After: `HashMap<Atom, usize>`, lookup by `&Atom::from(hh_key.as_str())`

## Validation

- cargo test: 32/32 passed
- cargo build --release: success
- No public API changes; no algorithm/threshold changes

## Expected Impact

- Clone cost for hotspot keys (uuid, hh_id) reduced (~90% cheaper pointer copies on Atom)
- Memory reduction from deduplicated UUID/hh_id strings across large datasets
- Conservative estimate with targeted scope: 10–25% peak memory reduction in household aggregation phases; end‑to‑end 15–20% when inputs have high duplication

Notes:
- Full `InternedPerson` adoption would further reduce memory (40–50%), but is deferred to avoid risk in Phase 1.

## Risks & Mitigations
- Minimal: `DefaultAtom` global cache growth tracks number of unique strings (bounded by dataset cardinality)
- No change to serialization/export or DB schema

## Next
- Proceed to T1‑CPU‑02 (Vec pre‑allocation)
- If memory remains the dominant limiter after Phase 1, revisit `InternedPerson` wiring in Phase 2

