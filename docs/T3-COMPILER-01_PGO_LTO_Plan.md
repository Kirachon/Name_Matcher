# T3-COMPILER-01: Profile-Guided Optimization (PGO) + Link-Time Optimization (LTO)

Date: 2025-10-01
Status: Documentation-only plan (no code changes)
Risk: Medium
Priority: P3 (High impact, low code risk; external toolchain required)

## Problem Statement
We want an additional 10–30% whole-program speedup without changing code semantics by using PGO and LTO in release builds.

## Solution Design
- Build an instrumented binary, run a representative workload to collect profiles, then rebuild using the profile with fat LTO and 1 codegen unit.
- No source changes required. Requires LLVM tooling (`llvm-profdata`).

## Build Steps
```bash
# 1) Instrumented build
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release

# 2) Run representative workload (your real DB / benchmark harness)
./target/release/name_matcher <benchmark args>

# 3) Merge profiles (requires LLVM tools)
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# 4) Optimized build with PGO + LTO
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata -Clto=fat -Ccodegen-units=1" \
  cargo build --release
```

## Correctness Analysis
- Compiler-only optimization; does not change program behavior.
- Validated by our full test suite (32/32 must pass) after building with PGO.

## Performance Analysis
- Expected overall speedup 10–30% depending on profile representativeness and hot paths.

## Risk Assessment
- Requires installing LLVM tools; environment-dependent.
- If the profile is unrepresentative, performance could degrade slightly.
- Reproducibility: keep profile artifacts versioned per environment, not in repo.

## Validation Plan
- After PGO build: run `cargo test --lib --features gpu,new_cli,new_engine -- --test-threads=1`.
- Run the benchmark harness with production-like args and compare against non-PGO release.

## Next Actions
- If you approve toolchain changes, we can add a small `scripts/pgo_build.ps1` helper and optional Cargo profiles for convenience (no default behavior change).

