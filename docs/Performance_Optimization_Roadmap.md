# Performance Optimization Implementation Roadmap

**Project**: Name_Matcher Performance Enhancement Initiative  
**Date**: 2025-09-30  
**Goal**: Achieve 3.5-5.5x speedup for million-record workloads  
**Timeline**: 7-11 weeks (phased implementation)

---

## Executive Summary

This roadmap provides a **week-by-week implementation plan** for the 23 high-priority optimizations identified in the comprehensive performance audit. The plan is organized into 4 phases with clear milestones, success criteria, and rollback strategies.

### Target Metrics

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| **Runtime** (1M records) | 45-60s | 12-18s | **3.5-4.5x faster** |
| **Memory** (peak) | 2-3 GB | 1-1.5 GB | **50% reduction** |
| **GPU Utilization** | 40-50% | 75-85% | **40-60% improvement** |
| **Throughput** | 16-22K rec/s | 55-85K rec/s | **3.5-4.5x improvement** |

---

## Phase 1: Quick Wins (Week 1-2)

**Goal**: Achieve **2-2.5x speedup** with low-risk, high-impact optimizations  
**Effort**: 1-2 weeks  
**Risk**: Low

### Week 1: GPU and Memory Optimizations

#### Day 1-2: T1-GPU-01 - Kernel Fusion

**Tasks**:
1. ✅ Implement fused CUDA kernel (Lev + Jaro + JW + Max3)
2. ✅ Update Rust integration code
3. ✅ Run unit tests (all 32 tests must pass)
4. ✅ Benchmark GPU performance (expect 2-3x speedup)

**Success Criteria**:
- Fused kernel produces identical results to separate kernels
- GPU kernel launch count reduced by 75% (4 → 1)
- GPU execution time reduced by 50-60%

**Rollback Strategy**:
```rust
#[cfg(feature = "fused_kernel")]
use fused_fuzzy_kernel;
#[cfg(not(feature = "fused_kernel"))]
use separate_kernels;
```

**Validation**:
```bash
cargo test --features gpu,new_engine -- --test-threads=1
cargo run --release --bin benchmark -- <args> --output week1_day2.json
python scripts/compare_benchmarks.py baseline.json week1_day2.json
```

---

#### Day 3: T1-MEM-01 - String Interning

**Tasks**:
1. ✅ Add `string_cache = "0.8"` to Cargo.toml
2. ✅ Create `InternedPerson` struct
3. ✅ Update normalization and matching code
4. ✅ Measure memory reduction

**Success Criteria**:
- Memory usage reduced by 40-50% for 1M records
- Clone operations 90% faster
- All tests pass

**Validation**:
```bash
cargo test --lib
# Memory test
cargo run --release --bin benchmark -- <args> --memory-profile week1_day3_mem.json
```

---

#### Day 4: T1-CPU-02 - Vec Pre-allocation

**Tasks**:
1. ✅ Find all `Vec::new()` in hot paths (grep search)
2. ✅ Replace with `Vec::with_capacity()`
3. ✅ Benchmark allocation performance

**Success Criteria**:
- Allocation count reduced by 95%
- Overall speedup of 15-25%

**Code Changes**:
```rust
// Before
let mut a_offsets: Vec<i32> = Vec::new();

// After
let mut a_offsets: Vec<i32> = Vec::with_capacity(tile_max);
```

---

#### Day 5: T1-CPU-03 - Eliminate Redundant Normalization

**Tasks**:
1. ✅ Add `phonetic_full` and `dmeta_code` to `FuzzyCache`
2. ✅ Pre-compute during cache building
3. ✅ Update `classify_pair_cached()` to use cached values

**Success Criteria**:
- Normalization calls reduced by 99%
- CPU time reduced by 10-15%

---

### Week 2: Database and I/O Optimizations

#### Day 1: T1-DB-01 - Batch INSERT

**Tasks**:
1. ✅ Implement batch INSERT helper function
2. ✅ Update seeding code to use batches of 1000
3. ✅ Benchmark insert performance

**Success Criteria**:
- Insert time reduced by 50-100x
- 1M records inserted in <1 minute

---

#### Day 2: T1-DB-02 - Connection Pool Warm-up

**Tasks**:
1. ✅ Implement pool warm-up logic
2. ✅ Test with various pool sizes
3. ✅ Measure first-query latency

**Success Criteria**:
- First batch latency reduced by 50-80%
- No cold start spikes

---

#### Day 3: T1-IO-01 - Buffered CSV Writing

**Tasks**:
1. ✅ Add `BufWriter` to CSV export
2. ✅ Benchmark export performance

**Success Criteria**:
- CSV export 30-50% faster
- Syscalls reduced by 99%

---

#### Day 4: T1-CONC-01 - Rayon Thread Pool Tuning

**Tasks**:
1. ✅ Adjust thread reservation logic
2. ✅ Benchmark CPU utilization

**Success Criteria**:
- CPU utilization improved by 5-10%
- Throughput improved by 5-8%

---

#### Day 5: Phase 1 Integration and Testing

**Tasks**:
1. ✅ Run full test suite (all 32 tests)
2. ✅ Run end-to-end benchmark (1M records)
3. ✅ Compare against baseline
4. ✅ Document results

**Success Criteria**:
- **All tests pass**
- **2-2.5x speedup achieved**
- **No regressions**

**Validation**:
```bash
cargo test --all-features -- --test-threads=1
cargo run --release --bin benchmark -- <full args> --output phase1_final.json
python scripts/compare_benchmarks.py baseline.json phase1_final.json
```

**Expected Results**:
```json
{
  "speedup": "2.3x",
  "runtime_baseline": "52.4s",
  "runtime_optimized": "22.8s",
  "memory_reduction": "45%",
  "gpu_utilization": "62%"
}
```

---

## Phase 2: Medium Effort Optimizations (Week 3-4)

**Goal**: Achieve **additional 1.5-2x speedup** (cumulative 3.5-4.5x)  
**Effort**: 2-3 weeks  
**Risk**: Medium

### Week 3: GPU and Memory Optimizations

#### Day 1-2: T2-GPU-02 - Pinned Memory

**Tasks**:
1. ✅ Implement pinned memory pool
2. ✅ Update memcpy calls to use pinned memory
3. ✅ Add fallback to pageable memory
4. ✅ Benchmark transfer speed

**Success Criteria**:
- Transfer speed improved by 30-40%
- Overall GPU speedup of 10-15%

**Risk Mitigation**:
- Limit pinned memory to 25% of system RAM
- Graceful fallback if allocation fails

---

#### Day 3: T2-MEM-02 - Buffer Pooling

**Tasks**:
1. ✅ Implement `TileBufferPool` struct
2. ✅ Update tile processing to reuse buffers
3. ✅ Benchmark allocation performance

**Success Criteria**:
- Allocation count reduced by 99%
- Overall speedup of 5-10%

---

#### Day 4-5: T2-CPU-01 - SIMD String Comparison

**Tasks**:
1. ✅ Implement AVX2 string comparison
2. ✅ Add runtime CPU feature detection
3. ✅ Maintain scalar fallback
4. ✅ Benchmark string comparison speed

**Success Criteria**:
- String comparison 3-5x faster (AVX2)
- Overall CPU speedup of 15-25%

**Risk Mitigation**:
- Extensive testing on various CPUs
- Fallback to scalar implementation

---

### Week 4: Database and Concurrency Optimizations

#### Day 1-2: T2-DB-01 - Prepared Statement Caching

**Tasks**:
1. ✅ Implement LRU statement cache
2. ✅ Update query functions to use cache
3. ✅ Benchmark query performance

**Success Criteria**:
- Query latency reduced by 20-30%
- Throughput improved by 25-35%

---

#### Day 3: T2-CPU-02 - FxHashMap

**Tasks**:
1. ✅ Replace `HashMap` with `FxHashMap` in hot paths
2. ✅ Benchmark hash performance

**Success Criteria**:
- HashMap operations 20-30% faster
- Overall speedup of 5-10%

---

#### Day 4: T2-GPU-03 - Asynchronous Kernel Execution

**Tasks**:
1. ✅ Implement stream pipelining
2. ✅ Overlap CPU and GPU work
3. ✅ Benchmark GPU utilization

**Success Criteria**:
- GPU utilization improved by 20-30%
- Overall speedup of 15-25%

---

#### Day 5: Phase 2 Integration and Testing

**Tasks**:
1. ✅ Run full test suite
2. ✅ Run end-to-end benchmark
3. ✅ Compare against Phase 1 results
4. ✅ Document results

**Success Criteria**:
- **All tests pass**
- **Cumulative 3.5-4.5x speedup achieved**
- **No regressions**

**Expected Results**:
```json
{
  "speedup_vs_baseline": "4.1x",
  "speedup_vs_phase1": "1.8x",
  "runtime_baseline": "52.4s",
  "runtime_optimized": "12.8s",
  "memory_reduction": "48%",
  "gpu_utilization": "78%"
}
```

---

## Phase 3: Major Refactoring (Week 5-7)

**Goal**: Achieve **additional 1.2-1.5x speedup** (cumulative 4.5-5.5x)  
**Effort**: 3-4 weeks  
**Risk**: High

### Week 5: Compiler Optimizations

#### Day 1-3: T3-COMPILER-01 - PGO + LTO

**Tasks**:
1. ✅ Set up PGO build pipeline
2. ✅ Create representative benchmark workload
3. ✅ Build instrumented binary
4. ✅ Run workload to collect profile data
5. ✅ Build optimized binary with PGO + LTO
6. ✅ Benchmark performance

**Success Criteria**:
- Overall speedup of 15-30%
- No code changes required

**Build Process**:
```bash
# Step 1: Instrumented build
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release

# Step 2: Run workload
./target/release/name_matcher <benchmark args>

# Step 3: Merge profile data
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Step 4: Optimized build
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata -Clto=fat -Ccodegen-units=1" \
  cargo build --release
```

---

### Week 6-7: Advanced GPU and Memory Optimizations

#### T3-GPU-01 - Persistent Kernel (Optional)

**Status**: **Deferred to future sprint**  
**Reason**: High complexity, uncertain benefit  
**Conditions to revisit**: GPU profiling shows >20% kernel launch overhead

---

#### T3-MEM-01 - Arena Allocator (Optional)

**Status**: **Deferred to future sprint**  
**Reason**: Complex lifetime management  
**Conditions to revisit**: Profiling shows >15% time in allocator

---

## Phase 4: Validation and Production Deployment (Week 8+)

### Week 8: Comprehensive Testing

**Tasks**:
1. ✅ Run full regression test suite (all 32 tests)
2. ✅ Run stress tests (10M records)
3. ✅ Run memory leak tests (Valgrind)
4. ✅ Run GPU profiling (Nsight Compute)
5. ✅ Run CPU profiling (perf + flamegraph)
6. ✅ Document all performance characteristics

**Success Criteria**:
- All tests pass
- No memory leaks
- No performance regressions
- GPU utilization >75%
- Memory usage <1.5 GB for 1M records

---

### Week 9: Production Deployment

**Tasks**:
1. ✅ Create release branch
2. ✅ Update documentation
3. ✅ Create deployment guide
4. ✅ Train users on new features
5. ✅ Monitor production performance

**Rollback Plan**:
- Keep baseline binary available
- Feature flags for all optimizations
- Gradual rollout (10% → 50% → 100%)

---

## Risk Management

### High-Risk Optimizations

| Optimization | Risk | Mitigation |
|--------------|------|------------|
| T2-GPU-02 (Pinned Memory) | OOM on low-RAM systems | Limit to 25% RAM, fallback to pageable |
| T2-CPU-01 (SIMD) | CPU compatibility | Runtime feature detection, scalar fallback |
| T3-GPU-01 (Persistent Kernel) | Deadlocks, race conditions | Extensive testing, defer to future sprint |
| T3-MEM-01 (Arena Allocator) | Lifetime bugs | Defer to future sprint |

### Rollback Strategy

**Feature Flags**:
```toml
[features]
default = ["gpu", "new_engine"]
fused_kernel = []
pinned_memory = []
simd_strings = []
pgo_optimized = []
```

**Runtime Configuration**:
```bash
# Disable specific optimizations
NAME_MATCHER_DISABLE_FUSED_KERNEL=1
NAME_MATCHER_DISABLE_PINNED_MEMORY=1
NAME_MATCHER_DISABLE_SIMD=1
```

---

## Success Metrics

### Performance Targets

| Phase | Cumulative Speedup | Runtime (1M) | Memory | GPU Util |
|-------|-------------------|--------------|--------|----------|
| Baseline | 1.0x | 52.4s | 2.5 GB | 45% |
| Phase 1 | 2.3x | 22.8s | 1.4 GB | 62% |
| Phase 2 | 4.1x | 12.8s | 1.3 GB | 78% |
| Phase 3 | 5.2x | 10.1s | 1.2 GB | 82% |

### Quality Targets

- ✅ All 32 unit tests pass
- ✅ No memory leaks (Valgrind clean)
- ✅ No performance regressions
- ✅ Code coverage >80%
- ✅ Documentation updated

---

## Monitoring and Observability

### Metrics to Track

**Runtime Metrics**:
- Total execution time
- Time per algorithm (1-6)
- GPU kernel time
- Database query time
- Normalization time

**Resource Metrics**:
- Peak memory usage
- GPU memory usage
- CPU utilization
- GPU utilization
- Disk I/O

**Quality Metrics**:
- Match accuracy (precision/recall)
- False positive rate
- False negative rate

### Dashboards

**Grafana Dashboard**:
```json
{
  "panels": [
    {"title": "Execution Time", "metric": "runtime_seconds"},
    {"title": "Memory Usage", "metric": "memory_mb"},
    {"title": "GPU Utilization", "metric": "gpu_util_pct"},
    {"title": "Throughput", "metric": "records_per_sec"}
  ]
}
```

---

## Conclusion

This roadmap provides a **structured, phased approach** to implementing 23 performance optimizations over 8-9 weeks. The plan prioritizes:

1. **Quick wins first** (Phase 1): Low risk, high impact
2. **Medium effort next** (Phase 2): Moderate risk, high impact
3. **Major refactoring last** (Phase 3): High risk, moderate impact
4. **Comprehensive validation** (Phase 4): Ensure production readiness

**Expected Outcome**: **3.5-5.5x speedup** with **50% memory reduction** and **40-60% GPU utilization improvement**.

---

**Document Version**: 1.0  
**Last Updated**: 2025-09-30  
**Status**: Ready for Implementation
