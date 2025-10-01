# Name Matcher - Business Process and Requirements Documentation

Generated from codebase analysis of this repository. All technical statements reflect the actual implementation as of this commit.

## Table of Contents
- Overview
- Matching Algorithms (Options 1–5)
- GPU Acceleration and Memory Management
- Database Integration, Streaming, and Checkpointing
- Configuration and Optimization Settings (GUI)
- Input/Output Formats and Data Flow
- Business Process & Requirements Analysis (BPRA)
- End-to-End Process Flow (Mermaid)
- Error Handling & Diagnostics
- Benchmarks/Performance Notes (from code paths)

---

## Overview
Name Matcher links person records across one or two MySQL databases using deterministic and fuzzy algorithms, with optional CUDA acceleration. It supports:
- In-memory and streaming modes (with resumable checkpoints in streaming + partitioned flows)
- GPU acceleration for hash-based joins (A1/A2 streaming) and fuzzy scoring/pre-pass (A3/A4)
- CSV/XLSX exports and summary statistics

Primary UI: Desktop GUI built with eframe/egui (src/bin/gui.rs). Matching core: src/matching/mod.rs and CSV/XLSX writers in src/export.

---

## Matching Algorithms (Options 1–5)
Enum: `matching::MatchingAlgorithm`
- Option 1: IdUuidYasIsMatchedInfnbd (deterministic)
  - Condition: birthdate equals AND first_name equals AND last_name equals
  - Normalization mode (optional): when `direct_norm_fuzzy` is ON, `normalize_simple` is applied before equality checks (lowercasing, dropping punctuation, etc.)
  - Matched fields reported: [id, uuid, first_name, last_name, birthdate]
  - Streaming hash-join supported; GPU build/probe hashing optional

- Option 2: IdUuidYasIsMatchedInfnmnbd (deterministic with middle)
  - Condition: birthdate equals AND first_name equals AND last_name equals AND middle_name equality (or both missing)
  - Normalization mode respected as in Option 1
  - Matched fields reported: [id, uuid, first_name, middle_name, last_name, birthdate]
  - Streaming hash-join supported; GPU build/probe hashing optional

- Option 3: Fuzzy (with middle)
  - Hard filter: exact birthdate equality required
  - Name score: composite of length-normalized Levenshtein, Jaro-Winkler, and Double Metaphone; returns label and 0–100 score
  - In-memory algorithm only (CSV output). Optional GPU: direct pre-pass hashing + metrics kernels

- Option 4: FuzzyNoMiddle (ignores middle)
  - Hard filter: exact birthdate equality required
  - Name score: same composite metrics, but computed on first+last only
  - In-memory algorithm only (CSV output). Optional GPU as above

- Option 5: HouseholdGpu
  - Leverages FuzzyNoMiddle semantics at the person level, then aggregates by (uuid from Table 1, hh_id from Table 2)
  - Produces household-level rows with `match_percentage`; filters to > 50%
  - GPU fuzzy metrics may be used heuristically; CPU fallback is implemented

Notes:
- All fuzzy algorithms require exact birthdate equality in their person-level comparisons.
- Streaming functions explicitly do not support fuzzy algorithms; fuzzy is handled in-memory or via partitioned CSV flow.

---

## GPU Acceleration and Memory Management
GPU usage (feature-gated by `--features gpu`):
- Deterministic A1/A2 (streaming): optional GPU hashing
  - Build-side hashing: `cfg.use_gpu_build_hash`
  - Probe-side hashing: `cfg.use_gpu_probe_hash` (kernel batch sizing via `cfg.gpu_probe_batch_mb`)
- Deterministic A1/A2 (in-memory): has a dedicated GPU hash path with self-budgeting; not controlled by probe batch UI value
- Fuzzy A3/A4 in-memory options:
  - GPU Fuzzy Pre-pass (hash-based candidate filtering): toggle via `set_gpu_fuzzy_direct_prep(true)` and budgeted by `set_gpu_fuzzy_prepass_budget_mb(mb)`; strategy commonly “last_initial” with exact birthdate requirement
  - GPU Fuzzy Metrics (Levenshtein/Jaro/JW): controlled by `set_gpu_fuzzy_metrics`, `set_gpu_fuzzy_force`, `set_gpu_fuzzy_disable` and used in in-memory and household flows; heuristics decide enablement unless forced

VRAM controls in the GUI (src/bin/gui.rs):
- GPU Mem Budget (MB): `gpu_mem_mb` → `MatchOptions.gpu.mem_budget_mb` for fuzzy kernels
- Probe GPU Mem (MB): `gpu_probe_mem_mb` → `StreamingConfig.gpu_probe_batch_mb` (A1/A2 streaming probe hashing batch sizing)
- Pre-pass VRAM (MB): `gpu_fuzzy_prep_mem_mb` → `matching::set_gpu_fuzzy_prepass_budget_mb` (A3/A4 GPU pre-pass hashing)

Optimization buttons auto-tune these values by inspecting CUDA free/total VRAM. Pre-pass VRAM uses ~25% (Auto) or ~40% (Aggressive) of available VRAM, clamped [128MB, 2048MB].

Isolation of memory pools:
- Hash-join (A1/A2) probe batch uses `gpu_probe_batch_mb` exclusively
- Fuzzy pre-pass uses `gpu_fuzzy_prep_mem_mb` exclusively
- Fuzzy metrics kernels use `gpu_mem_mb`

---

## Database Integration, Streaming, and Checkpointing
- DB: MySQL via sqlx; connection configured with `DatabaseConfig`
- Dual-database mode supported (two separate connections)
- Streaming mode (A1/A2 only):
  - Builds an inner index and streams the outer table in batches: `StreamingConfig.batch_size`
  - Adaptive batch throttling when free RAM < `memory_soft_min_mb`
  - Optional GPU hashing (build/probe); probe batches sized by `gpu_probe_batch_mb`
  - Checkpointing/resume supported: `.nmckpt` files via util/checkpoint helpers
- Partitioned streaming for fuzzy (CSV only) is implemented in dedicated paths that retain exact birthdate filtering and optional GPU pre-pass/metrics with CPU fallbacks

---

## Configuration and Optimization Settings (GUI)
Selectable in GUI:
- Algorithm (Options 1–5)
- Mode: Auto/Streaming/In-Memory
- Batch size (streaming), pool size, soft min free RAM
- GPU toggles (use GPU, use GPU hash join, build/probe), fuzzy GPU mode (Off/Auto/Force)
- GPU memory budgets: `gpu_mem_mb`, `gpu_probe_mem_mb`, `gpu_fuzzy_prep_mem_mb`
- Fuzzy pre-pass toggle (A3/A4 only) and threshold for keeping fuzzy matches
- Save/Load profile file `.nm_opt_profile` (now includes `gpu_fuzzy_prep_mem_mb`)
- Export Recommendations CSV (now includes `gpu_fuzzy_prep_mem_mb`)

Auto Optimize / Max Performance set pool, batch, thresholds, and GPU budgets using system RAM/VRAM detection. Status messages include selected GPU budgets, including pre-pass VRAM.

---

## Input/Output Formats and Data Flow
- Input (DB tables): expected columns include id, uuid, first_name, middle_name, last_name, birthdate; GUI provides mapping and WHERE filters in other panels
- Output:
  - CSV (A1/A2/A3/A4): headers vary by algorithm; fuzzy rows are filtered by user threshold
  - CSV Household (A5): [id, uuid, hh_id, match_percentage, region_code, poor_hat_0, poor_hat_10]
  - XLSX exports for deterministic summaries and household outputs
  - Summary CSV with aggregate counts and derived rates

Person-level CSV headers (see src/export/csv_export.rs):
- A1: Table1_ID, Table1_UUID, Table1_FirstName, Table1_LastName, Table1_Birthdate, Table2_..., is_matched_Infnbd, Confidence, MatchedFields
- A2: … + MiddleName fields and is_matched_Infnmnbd
- A3/A4: includes middle_name columns (empty when missing), is_matched_Fuzzy = true for kept rows

---

## Business Process & Requirements Analysis (BPRA)
- Objectives
  - Identify exact deterministic matches (A1/A2) and high-confidence fuzzy matches (A3/A4) under strict birthdate equality
  - Aggregate household matches (A5) with >50% member congruence
  - Provide tunable performance knobs and robust observability (progress, status, logs)

- Functional Requirements
  - Algorithm selection (1–5) with exact DOB equality constraint for all
  - Deterministic equality (A1/A2) with optional normalized equality
  - Fuzzy scoring (A3/A4) with composite similarity; threshold filtering
  - Household aggregation (A5) built on person-level fuzzy results
  - Modes: streaming (A1/A2) vs in-memory (A3/A4/A5)
  - GPU acceleration: available where implemented; CPU fallback with clear messaging
  - Save/Load profiles and export recommendations including all GPU budgets

- Non-functional Requirements
  - Scalability: streaming batches with backoff; partitioned flows for large datasets
  - Resilience: checkpointing and resume; explicit error categories for DB/connectivity/format/resource
  - Performance: GPU-accelerated hashing and kernels when beneficial; heuristics to avoid slow GPU paths
  - Observability: progress updates include memory and GPU activity

- User Roles
  - Data Operator (GUI): configures DBs/tables/columns, selects algorithm and mode, tunes memory/GPU, runs and exports results

- Data Constraints
  - Birthdate columns must be present for matching to succeed as intended
  - Name fields may be NULL; normalization paths handle empties explicitly

- Integration Requirements
  - MySQL connectivity via sqlx; dual-DB supported; WHERE filtering for subsets
  - Exports to CSV/XLSX; summary CSV for reporting

---

## End-to-End Process Flow (Mermaid)
```mermaid
flowchart TD
  A[Configure GUI: DB(s), tables, columns, algorithm, mode, GPU opts] --> B{Mode}
  B -- Streaming (A1/A2) --> C[Build inner index]
  C --> D[Optional GPU Build Hash]
  D --> E[Stream outer in batches]
  E --> F[Optional GPU Probe Hash
(gpu_probe_batch_mb)]
  F --> G[Equality check (A1/A2)]
  B -- In-Memory (A3/A4/A5) --> H[Optional Fuzzy GPU Pre-pass (gpu_fuzzy_prep_mem_mb)]
  H --> I[Fuzzy metrics (CPU/GPU; gpu_mem_mb)]
  I --> J[Threshold filter (CSV)]
  J --> K[Household aggregation (>50%) for A5]
  G --> L[Export CSV/XLSX and Summary]
  K --> L
```

---

## Error Handling & Diagnostics
- SQLSTATE and message inspection in GUI to categorize errors: connection, schema, data format, resource constraints, configuration, unknown
- Progress updates carry `stage` and memory/GPU usage; GUI displays running status, ETA, and throughput
- CUDA diagnostics dialog gathers device info and VRAM; Auto Optimize/Max Perf reflect VRAM in status

---

## Benchmarks/Performance Notes (from code paths)
- Heuristics enable GPU fuzzy metrics when candidate volume and name lengths suggest benefit; otherwise CPU typically wins
- Probe hashing batches sized by `gpu_probe_batch_mb` to control VRAM usage in streaming hash-join
- Fuzzy GPU pre-pass reduces candidate pairs via birthdate and last-initial blocking under budget `gpu_fuzzy_prep_mem_mb`

---

## Change Log (relevant features finalized here)
- Added Pre-pass VRAM (MB) to:
  - Save/Load profile (.nm_opt_profile)
  - Recommendations CSV export
  - Auto Optimize / Max Performance summaries
- Independent VRAM controls for A1/A2 probe hashing vs A3/A4 fuzzy pre-pass vs fuzzy metrics

