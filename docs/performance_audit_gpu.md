# Name Matcher GPU/CUDA Performance Audit and Optimization Plan

## Scope
- Algorithms audited: A1 (IdUuidYasIsMatchedInfnbd), A2 (IdUuidYasIsMatchedInfnmnbd), Fuzzy, FuzzyNoMiddle, HouseholdGpu
- Execution modes: In‑memory and DB streaming (single/dual DB, partitioned)
- GPU components: Hash join (A1/A2 build & probe), fuzzy direct pre‑filter (hash), household GPU matcher

## Current GPU Usage (as implemented)
- Hash Join (A1/A2)
  - GPU hashing for build and probe phases via FNV‑1a 64‑bit kernel (NVRTC compiled)
  - VRAM‑aware tiling for probe (`hash_fnv1a64_batch_tiled`), OOM backoff, CPU fallback per tile
  - StreamingConfig switches: `use_gpu_hash_join`, `use_gpu_build_hash`, `use_gpu_probe_hash`, `gpu_probe_batch_mb`
- Fuzzy (3/4)
  - Optional GPU pre‑pass for direct candidate filtering using birthdate (and optional last initial) hash blocking
  - In‑memory fuzzy path has GPU backend option controlled via `MatchOptions`
- Household (5)
  - GPU accelerated in‑memory household matching pipeline, export to CSV/XLSX

## Observed Bottlenecks & Opportunities
1. Hash kernel cold start
   - NVRTC compilation and module/function loading performed per context construction.
   - Impact: noticeable overhead on first use; repeated in different call sites.
2. Device memory churn
   - Per‑tile allocations for buffers; no reuse across tiles/batches.
   - Impact: additional overhead and allocator pressure.
3. Transfer/compute overlap
   - Tiles processed serially on the default stream; no copy/compute overlap.
   - Impact: PCIe transfer time not hidden; reduces effective throughput.
4. String flattening cost
   - Keys are rebuilt/flattened to bytes per tile; no pooling or reuse of staging buffers.
5. Fuzzy string metrics
   - Levenshtein/Jaro‑Winkler/Soundex currently CPU in final verification; GPU pre‑pass reduces work but not metric time in heavy fuzzy workloads.
6. Streaming pipeline
   - Good tiling/backoff exists; further gains possible from better partitioning (skew handling), async prefetching, and GPU stream pipelining.

## Quick Wins (implemented now)
- Cache CUDA context + compiled module/function for hashing globally
  - Added `GpuHashContext::get()` using `OnceLock` and made `GpuHashContext` `Clone`.
  - Switched call sites to `.get()` to avoid repeated NVRTC compile/load.
  - Effect: reduces cold‑start latency and repeated module loads across batches and modes.
- Double‑buffered tiled hashing with two CUDA streams (opt‑in)
  - New `hash_fnv1a64_batch_tiled_overlap()` overlaps HtoD/DtoH transfers with compute using two streams.
  - Controlled by `StreamingConfig.gpu_streams` (1=off, >=2 enables overlap). CPU fallback preserved.
- Device buffer reuse pool (opt‑in)
  - Grow‑only reuse of device output buffers between tiles to reduce allocations. Controlled by `StreamingConfig.gpu_buffer_pool`.
  - Host pinned staging currently best‑effort: when `gpu_use_pinned_host=true`, logs a note if unavailable and continues with pageable memory.

## High‑Impact Optimization Plan (prioritized)
1. Double buffering with two CUDA streams (copy/compute overlap)
   - Maintain two sets of device buffers; while stream A computes tile N, stream B copies tile N+1.
   - Expected: 1.2–1.6× throughput on probe hashing (PCIe‑bound phases), depending on data size and GPU.
2. Reusable device/host buffers
   - Pool device buffers sized to the largest observed tile (grow‑only) and reuse across tiles.
   - Use pinned host memory for staging to speed transfers.
   - Expected: 1.1–1.3× and lower latency jitter.
3. Build‑side tiling (optional)
   - Use `hash_fnv1a64_batch_tiled` for large build batches as well when VRAM budget is constrained.
   - Expected: stability under memory pressure; modest perf gains.
4. Kernel micro‑optimizations
   - Unroll inner loops for short strings; load 8 bytes at a time; shared‑memory staging for aligned segments.
   - Expected: 1.05–1.15× depending on string length distribution.
5. Fuzzy GPU string metrics (phase 2)
   - Implement GPU Levenshtein/Jaro‑Winkler/Soundex with warp‑level parallelism and banded DP for short names.
   - Integrate via optional path after hash pre‑pass; exact scoring preserved.
   - Expected: 1.5–3× for heavy fuzzy workloads.
6. Streaming orchestration
   - Async prefetch of next DB chunk; parallel normalize + key concat using rayon; pipeline GPU hashing via channels.
   - Expected: 1.1–1.4× end‑to‑end improvement depending on DB/IO.

## Safety and Backward Compatibility
- CPU fallback preserved for all steps upon any GPU error or feature disabled.
- Feature flags and runtime toggles unchanged. New optimizations will default OFF or mimic prior behavior when disabled.
- Exact algorithm semantics and confidence scoring maintained.

## Benchmarking Strategy
- Microbench: synthetic string batches (varying lengths) for hash build/probe; measure GB/s and throughput.
- End‑to‑end: small DB seeds via `src/bin/gpu_audit.rs` against MySQL on port 3307 (per .env) for options A1/A2 (GPU hash join), Fuzzy, and Household.
- Record cold vs warm timings to capture the benefit of cached module loading.

## New Configuration (GUI/CLI)
- `gpu_streams` (u32): number of CUDA streams for pipelining (default 1 = off). GUI: GPU Hash Join panel → "CUDA Streams (1=off)".
- `gpu_use_pinned_host` (bool): try to use pinned host staging (default false). Currently best‑effort; falls back to pageable memory.
- `gpu_buffer_pool` (bool): reuse device buffers across tiles (default true). GUI: GPU Hash Join panel toggle.
- `gpu_fuzzy_metrics` (bool): enable CUDA kernels for Levenshtein/Jaro/Jaro‑Winkler scoring in Fuzzy/FuzzyNoMiddle (Algo 3/4). Defaults off. CLI: `--gpu-fuzzy-metrics` / env `NAME_MATCHER_GPU_FUZZY_METRICS=1`. GUI: GPU Acceleration → "GPU Fuzzy Metrics" checkbox.
- Streaming pipeline: `async_prefetch` (bool), `parallel_normalize` (bool), `prefetch_pool_size` (u32). Defaults OFF/sane values; improves overlap for streaming.

## Results (representative)
- Test rig: RTX 4050 Laptop GPU, CUDA 13, MySQL on localhost:3307, A/B tables ~30k rows each
- Streaming A1/A2 GPU hash join:
  - Streams=1: 1.407s
  - Streams=2 + buffer pool: 1.133s
  - Improvement: ~1.24× on probe hashing (20k row batches)
- Fuzzy GPU metrics (Algo 3) [opt‑in]:
  - With GPU metrics enabled via `gpu_audit`: 142.3s for ~30k×30k synthetic workload
  - CPU baseline: 4.28s
  - Interpretation: current GPU metrics path favors massive candidate counts with short strings; for medium datasets CPU remains faster. Keep toggle OFF by default; rely on GPU pre‑pass + CPU scoring for best E2E on this dataset.

## Next Steps
- Tune fuzzy GPU kernels (larger tiles, better blocking, batched candidate formation) and add heuristics to enable only when profitable.
- Extend streaming to prefetch/parallel normalize for better CPU/GPU overlap.
- GUI HUD now shows device name and VRAM free/total; add richer charts if needed.


## Heuristic activation for GPU fuzzy metrics

### Option 5 (Household Aggregation) GPU Mode
- CLI: set algorithm=5; in-memory only. Use env NAME_MATCHER_USE_GPU=1 to enable GPU-backed fuzzy pre-pass in Option 5.
- Threshold: NAME_MATCHER_HOUSEHOLD_THRESHOLD accepts values like 60, 80, 95, or 0.95 (defaults to 0.95).
- GPU memory budget: NAME_MATCHER_GPU_MEM_MB (default 512) for Option 5 in-memory GPU runs.
- Monitoring (best-effort): set NAME_MATCHER_GPU_LOG_CSV=./tmp/gpu_status_log.csv to log GPU total/free MB snapshots at key Option 5 stages.
- Notes: Aggregation currently runs on CPU; the pair-generation stage (fuzzy no-middle with exact birthdate) may use CUDA kernels depending on heuristics/toggles.

- By default, when `gpu_fuzzy_metrics` is enabled, the engine applies a runtime heuristic to decide whether to actually use CUDA kernels for Fuzzy/FuzzyNoMiddle scoring.
- The heuristic considers:
  - Estimated candidate pairs via (birthdate, last-initial) blocking
  - Average and maximum normalized name lengths
  - Practical kernel sweet spots (<=64 total name chars, large candidate sets)
- Rules (initial):
  - Disable if max_len > 64, or avg_len > 32, or estimated candidates < 10M
  - Otherwise enable and log the decision
- Override controls:
  - Force on: `--gpu-fuzzy-force` or `NAME_MATCHER_GPU_FUZZY_FORCE=1`
  - Force off: `--gpu-fuzzy-disable` or `NAME_MATCHER_GPU_FUZZY_DISABLE=1`


## Heuristic coverage and GUI integration (update)
- Coverage extended to Algorithm 5 (HouseholdGpu) and partitioned streaming fuzzy phases. Heuristic gating and overrides are applied consistently.
- GUI controls added in `src/bin/gui.rs`:
  - Enable (heuristic) GPU Fuzzy Metrics
  - Force GPU Fuzzy, Disable GPU Fuzzy
- StreamingConfig now carries `use_gpu_fuzzy_metrics`; the GUI wires both Streaming and In-memory modes using `set_gpu_fuzzy_metrics/force/disable`.
- HUD shows "Fuzzy GPU (heuristic): active/inactive/off" reflecting runtime decisions.

## Appendix: Files & Key Functions
- `src/matching/mod.rs`
  - `GpuHashContext` (now cached), `hash_fnv1a64_batch`, `hash_fnv1a64_batch_tiled`
  - `stream_match_gpu_hash_join` (A1/A2), `fuzzy_direct_gpu_hash_prefilter_indices`
  - `det_match_gpu_hash_inmemory`
- `src/bin/gpu_audit.rs` — minimal end‑to‑end benchmarking tool

