## Name Matcher Algorithms: Technical Guide and Decision Playbook

This document explains the three matching algorithms used by the Name Matcher application, their technical criteria, performance characteristics, and practical decision rules for selecting the right algorithm and execution mode (CPU vs GPU; streaming vs in-memory).

- Algorithms covered
  1) Algorithm 1 (first+last+birthdate) — Deterministic, exact on normalized names and birthdate
  2) Algorithm 2 (first+middle+last+birthdate) — Deterministic with middle name constraint
  3) Fuzzy Algorithm — Probabilistic using Levenshtein, Jaro‑Winkler, and phonetic (Double Metaphone)

- Modes and acceleration
  - In‑memory mode: loads both tables into RAM; parallelized via Rayon; optional GPU for hashing (A1/A2) and for fuzzy kernels
  - Streaming mode: processes in chunks; supports checkpoints and partitioning; optional GPU hash‑join acceleration for A1/A2

  - Optional GPU fuzzy direct pre-pass: hash-based candidate filtering before fuzzy scoring (no change to scoring/threshold logic)

References to code
- Enum and algorithm implementations: src/matching/mod.rs (MatchingAlgorithm, matches_algo1/2, fuzzy_compare_names_new)
- GPU hashing and kernels: src/matching/mod.rs (gpu::*, hash_fnv1a64_batch_tiled, jaro/jw/lev kernels)
- CLI behavior and mode selection: src/main.rs
- DB config to URL: src/config.rs; pooling: src/db/connection.rs

---

## Algorithm 1 — Deterministic (first + last + birthdate)

Technical criteria
- Normalize fields (trim, case/space simplification, diacritic handling)
- Required fields present: first_name, last_name, birthdate
- Match condition (exact after normalization):
  - first_name == first_name AND last_name == last_name AND birthdate == birthdate
- Output confidence: 1.0 (100%) for each accepted pair; matched_fields includes id/uuid/name components and birthdate

Performance characteristics
- Speed: Very fast on CPU; O(N×M) in pure nested comparison, but typically bounded by indexing or hash prefilter
- Memory: Low to moderate in streaming; higher in in‑memory (both tables resident)
- Scalability: Excellent with streaming + hash prefilter (see GPU hash join)

Accuracy trade‑offs
- False positives: Extremely low (requires exact equality on all three dimensions)
- False negatives: Possible when data is noisy (nicknames, typos, missing birthdate)

Best‑fit datasets
- Clean and standardized data (consistent casing, normalized names, accurate dates)
- When high precision is required and recall can be lower

GPU vs CPU
- GPU acceleration benefits via hash‑join prefilter (build/probe hashing) in streaming mode
- In‑memory mode can use GPU hashing for key computation; verification remains CPU exact comparisons
- Use GPU when batches are large and VRAM budget is sufficient; otherwise CPU is already strong

---

## Algorithm 2 — Deterministic (first + middle + last + birthdate)

Technical criteria
- Same normalization as Algorithm 1
- Required fields: first_name, last_name, birthdate; middle_name treated as required when present (None == None is allowed)
- Match condition (exact after normalization):
  - first_name == first_name AND last_name == last_name AND birthdate == birthdate AND middle_name equality
    - If both middle names are empty/None, the constraint is satisfied; else exact equality is required

Performance characteristics
- Speed: Slightly slower than A1 due to the extra middle name check; similar behavior under hash prefilter
- Memory: Same profile as A1
- Scalability: Excellent with streaming + GPU hash‑join

Accuracy trade‑offs
- False positives: Extremely low
- False negatives: Higher than A1 if middle names have inconsistent formats (initials vs full, missing vs present)

Best‑fit datasets
- High‑quality datasets where middle name is frequently populated and consistent
- Regulatory/record‑linkage tasks where exact identity confirmation is critical

GPU vs CPU
- Same guidance as A1; GPU hash‑join prefilter is effective on large batches

---

## Algorithm 3 — Fuzzy (probabilistic using Levenshtein, Jaro‑Winkler, phonetic)

Technical criteria (summarized from fuzzy_compare_names_new)
- Birthdate must match exactly before applying name similarity
- Names are normalized to simple ASCII/spacing for metric computation
- Metrics computed on full name string:
  - Levenshtein similarity (length‑normalized) → percentage
  - Jaro‑Winkler → percentage
  - Phonetic match via Double Metaphone (binary 100%/0% on code equality)
- Decision logic:
  - DIRECT MATCH: exact full-string equality → 100%
  - CASE 1: lev ≥ 85%, jw ≥ 85%, dmeta == 100% → use average
  - CASE 2: at least 2 out of 3 metrics pass thresholds → use average
    - CASE 3 refinement: if avg ≥ 88% and per‑component Levenshtein distances are ≤ 2 on first/last/middle, label as CASE 3
- Final confidence is average of passing metrics scaled to 0–1; export and downstream thresholds typically ≥ 0.95

Performance characteristics
- Speed: Slower than deterministic due to string distance kernels; optimized with GPU kernels for lev/jaro/jw (MAX_STR=64 cap per component)
- Memory: Moderate to high depending on batch size
- Scalability: Good with streaming partitions and birthdate blocking; computationally heavier than A1/A2

Accuracy trade‑offs
- False positives: Tunable via threshold; can increase if threshold is lowered
- False negatives: Decrease as threshold lowers; higher when names are very noisy or differently formatted

Best‑fit datasets
- Noisy or heterogeneous names (typos, nicknames, transpositions)
- Need for higher recall while maintaining reasonable precision; dates are reliable

GPU vs CPU
- GPU kernels accelerate lev/jaro/jw scoring; useful on medium/large batches
- Optional GPU fuzzy direct pre-pass can reduce candidate pairs via birthdate and last-initial blocking before scoring, with CPU verification unchanged.

- Streaming fuzzy (partitioned) is supported for CSV; XLSX currently focuses on deterministic algorithms for streaming output

---

## Hash Join Performance Analysis (A1 & A2 only)

When GPU Hash Join Helps
- Large outer probe batches where hashing thousands to millions of normalized keys per chunk amortizes kernel launches
- Use cases: streaming A1/A2 with single‑DB; cross‑DB currently falls back to CPU hashing for build side

Key components
- Build side: Hash index over smaller table (CPU hash by default; optional GPU hashing when enabled)
- Probe side: GPU FNV‑1a 64‑bit hashing with VRAM‑aware tiling honoring a user budget
- Verification: Always CPU exact checks (matches_algo1/matches_algo2) after candidate lookup to guarantee correctness

Memory and VRAM considerations
- GPU probe hashing uses tiled kernel dispatch: hash_fnv1a64_batch_tiled(strings, budget_mb, reserve_mb)
  - Respects current free VRAM and a configured budget; halves tile size on CUDA OOM and falls back to CPU per‑tile if needed
  - GUI exposes the probe budget; for CLI, GPU hash‑join is enabled by NAME_MATCHER_GPU_HASH_JOIN=1 and internal defaults apply
- Recommended reserve: keep at least ~64 MB free headroom on the device to accommodate buffers and driver overhead

Streaming vs In‑Memory selection
- CLI heuristic: use streaming if NAME_MATCHER_STREAMING=1 or total rows (table1+table2) > 200,000; fuzzy CSV is allowed; deterministic A1/A2 can be streamed with or without GPU
- In‑memory: faster on small to medium datasets that fit RAM comfortably; A1/A2 can use GPU hashing for keys, with CPU verification

Batch size optimization
- Start with 50k–100k for ≥16–32 GB RAM systems; adjust upward if stable and DB I/O is not the bottleneck
- Monitor:
  - Chunk latency: should be stable; sudden spikes can indicate VRAM/RAM pressure
  - GPU probe OOM logs: reduce batch or increase tiling via lower probe budget
  - Memory soft minimum: system backoff may halve batch size automatically if RAM is tight
- Minimum enforced is 10,000 per chunk in streaming paths

Performance expectations (relative)
- A1/A2 CPU (in‑memory): High throughput on small/medium data; near linear scaling with threads
- A1/A2 GPU hash‑join (streaming): Highest throughput on large data; big gains when probe batches are large and VRAM tiling keeps kernels saturated
- Fuzzy CPU/GPU: Moderate throughput; GPU helps, but string distance remains heavier than hash checks; use partitioning by birthdate to bound comparisons

---

## Decision Matrix (Algorithm and Mode Selection)

| Factor | Recommended Choice |
|---|---|
| Data is clean, standardized; dates are reliable | Algorithm 1 (A1); if middle name is consistently present and needed, use Algorithm 2 (A2) |
| Need exact identity confirmation (regulatory, dedupe) | A2 (or A1 if middle name quality is inconsistent) |
| Names are noisy; need higher recall with acceptable precision | Fuzzy (threshold ≥ 0.95 recommended) |
| Dataset size small (<10K) | In‑memory; CPU is sufficient (GPU optional) |
| Dataset size medium (10K–200K) | In‑memory for speed; consider GPU for Fuzzy or when CPU is saturated |
| Dataset size large (>200K) | Streaming; enable GPU hash‑join for A1/A2 |
| Tight time constraints on large data | Streaming + GPU hash‑join (A1/A2) with tuned batch size and probe VRAM budget |
| Limited RAM | Streaming with smaller batch (≥10K); prefer CPU or modest GPU usage |
| GPU VRAM available (≥4–8 GB) | Enable GPU hash‑join for A1/A2; enable GPU for fuzzy kernels on larger batches |

Quick decision tree
1) Is data clean and exact matches acceptable? → Use A1 (or A2 if middle name consistency is required)
   - Large data or time‑critical? → Streaming + GPU hash‑join
   - Small/medium data? → In‑memory (GPU optional)
2) Data is noisy (typos/nicknames), but dates are reliable → Fuzzy
   - CSV only; consider streaming partitioned for large data; enable GPU kernels if available

---

## Practical Recommendations

Configuration examples
- .env for CLI (single DB)
```dotenv
# DB connection (example)
DB_HOST=127.0.0.1
DB_PORT=3307
DB_USER=root
DB_PASSWORD=secret
DB_NAME=people

# Streaming and GPU
NAME_MATCHER_STREAMING=1
NAME_MATCHER_GPU_HASH_JOIN=1
NAME_MATCHER_GPU_FUZZY_DIRECT_HASH=1
NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION=1
RUST_LOG=info
```

- CLI command examples
```bash
# CSV output, A1, streaming + GPU hash-join
name_matcher 127.0.0.1 3307 root secret people t1 t2 1 D:/out/matches.csv --gpu-hash-join

# In-memory, A2, CSV
name_matcher 127.0.0.1 3307 root secret people t1 t2 2 D:/out/matches.csv

# Fuzzy (CSV), with GPU fuzzy direct pre-pass and permissive normalization for A1/A2
name_matcher 127.0.0.1 3307 root secret people t1 t2 3 D:/out/matches.csv --gpu-fuzzy-direct-hash --direct-fuzzy-normalization


# Fuzzy (CSV only), in-memory by default unless streaming is forced
name_matcher 127.0.0.1 3307 root secret people t1 t2 3 D:/out/matches.csv
```

- GUI toggles
  - “GPU Hash Join (A1/A2)”, “GPU for Build Hash”, “GPU for Probe Hash”
  - “GPU Fuzzy Direct Pre‑Pass” (candidate filtering before fuzzy scoring)
  - “Apply Fuzzy‑style normalization to Algorithms 1 & 2” (makes A1/A2 normalization match Fuzzy’s permissive rules)
  - “Probe GPU Mem (MB)” controls VRAM tiling budget for probe hashing
  - Mode selection: In‑memory vs Streaming (Auto/Estimate can help)

Common pitfalls and troubleshooting
- Connection issues: verify DB_HOST/DB_PORT and firewall; use “Test Connection” in GUI
- Empty middle names in A2: None/None is OK; inconsistent presence across tables will cause mismatches → prefer A1
- Fuzzy in streaming CSV: supported via partitioned path; XLSX streaming focuses on deterministic algorithms
- GPU OOM during probe hashing: lower batch size or reduce “Probe GPU Mem (MB)”; logs show OOM backoff/tiling
- Cross‑database streaming with GPU: build hash may fall back to CPU; still safe and typically fast
- Threshold tuning for Fuzzy: start at 0.95; review samples before lowering

Best practices (production)
- Indices: Ensure indexes on id, birthdate, and name fields used for filtering; consider composite indexes for export/joins
- Partitioning: Use last‑initial or birthdate partitions for very large tables to bound comparisons
- Checkpoints: Enable checkpoints for long streaming runs to allow resume after interruptions
- Monitoring: Enable RUST_LOG=info; track progress updates (percent, ETA, mem, GPU stats)
- Resource planning: Size batch to keep RAM headroom; set GPU probe budget to avoid VRAM OOM; consider CPU pool sizing via NAME_MATCHER_POOL_* env vars
- Validation: Periodically sample matches; for Fuzzy, audit borderline scores and adjust threshold

---

## Performance Comparison (Qualitative)

| Algorithm | CPU In‑Memory | CPU Streaming | GPU (A1/A2 hash‑join; Fuzzy kernels) | Notes |
|---|---|---|---|---|
| A1 | Very fast (small/medium); near‑linear with threads | Excellent with large data | Highest throughput with large probe batches | Exact matches; lowest FP; watch DB I/O |
| A2 | Fast; slightly slower than A1 | Excellent | Same as A1 with small overhead | Requires middle name consistency |
| Fuzzy | Moderate; heavier per pair | Good with partitioning | Good‑to‑very‑good on large batches | CSV only; birthdate blocking recommended |

Throughput varies by hardware, string lengths, DB latency, and configuration. Use Estimate/Auto in GUI and adjust batch and probe VRAM budget iteratively.

---

## Appendix: Key Internals
- Deterministic checks: matches_algo1 / matches_algo2 (exact equality after normalization)
- Fuzzy metrics: Levenshtein (length‑normalized), Jaro‑Winkler, Double Metaphone code equality with rule set for CASE 1/2/3
- GPU hashing: FNV‑1a 64‑bit; tiled probe hashing honoring configured MB budget; OOM backoff and CPU fallback
- Verification: Deterministic verification remains CPU‑side to guarantee correctness after hash candidate selection
- Mode selection heuristic: Streaming if NAME_MATCHER_STREAMING=1 or (rows1+rows2) > 200k; Fuzzy CSV supported; GPU toggles available in GUI

