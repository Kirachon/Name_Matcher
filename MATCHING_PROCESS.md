## SRS-II Name Matching Application: Architecture, Algorithms, and Performance

### Backend Architecture Overview
- Components
  - Data Access: MySQL (via sqlx). Tables of Person rows are fetched into memory as Vec<Person>.
  - Matching Engine: Three algorithms (Algorithm 1, 2, 3) implemented in Rust.
  - Hybrid Compute: GPU kernels compute string metrics (Levenshtein, Jaro, Jaro-Winkler) in bulk; CPU performs authoritative business-rule classification.
  - Export: Matched pairs with confidence are written to CSV/XLSX.
- Execution Modes
  - CPU-only: All normalization, blocking, metrics, and classification on CPU.
  - GPU-accelerated: CPU handles normalization and blocking; GPU computes metrics for candidate tiles; CPU classifies results (authoritative).

### Data Flow
1. Database read
   - Connect to MySQL and fetch Person rows for the source and target tables.
2. Normalization
   - For each person: normalize names (remove punctuation/diacritics, case-fold), parse birthdate.
   - New: Per-person caches are built once for each side with normalized strings and Double Metaphone codes.
3. Blocking
   - Multi-field blocking to reduce the candidate space, using birth year, first/last initial, and last name soundex.
4. Candidate Metric Computation
   - GPU mode: Build tiles of candidate pairs; copy names into contiguous buffers; launch CUDA kernels for metrics.
   - CPU mode: Compute metrics directly during classification.
5. Authoritative Classification (CPU)
   - Apply deterministic business rules (DIRECT MATCH, CASE 1/2/3) over metrics and normalized fields.
6. Results Aggregation
   - Confidence scores are normalized (0–1), labels recorded, and pairs filtered by thresholds.
7. Export
   - Write selected matches to CSV/XLSX with relevant fields and confidence.

### Matching Algorithms
- Algorithm 1: Exact match on first + last + birthdate
  - Fast path for clear duplicates.
- Algorithm 2: Exact match on first + middle + last + birthdate
  - Stricter variant used when middle name must match.
- Algorithm 3: Fuzzy matching (applies CASE 1/2/3)
  - Metrics: Levenshtein percentage, Jaro, Jaro‑Winkler, and Double Metaphone equality.
  - Rules (high level):
    - DIRECT MATCH: Full normalized name equality (100%).
    - CASE 1: All three metrics strong (>=85%) and phonetic codes equal, average score reported.
    - CASE 2: Any 2 of 3 metrics strong (>=85%); average score reported.
    - CASE 3 (refinement on CASE 2): If average >=88 and per-field LD for first/middle/last <=2, upgrade to CASE 3.

### GPU vs CPU Processing
- GPU acceleration
  - Batched, coalesced computation of string metrics using CUDA kernels.
  - Max-of-three score used as a prefilter; thresholds reduce CPU work.
- CPU authoritative classification
  - Ensures byte-for-byte identical results with the CPU-only pipeline.
  - Applies domain rules, exact checks, and per-field refinements.

### Performance Optimizations
- Caching (New – Most Impactful)
  - Per-person caches built once:
    - simple_full: normalize_simple("first mid last")
    - simple_first/mid/last
    - phonetic_full: normalize_for_phonetic(simple_full)
    - dmeta_code: Double Metaphone code, with panic-guard; empty if unavailable
  - GPU post-processing now uses these caches to avoid recomputing normalization and phonetic codes for every pair.
  - Results remain byte-for-byte identical to prior CPU classification because the same normalization/encoding routines produce the cached values.
- Streaming and memory management
  - Tiling of candidate pairs to fit GPU memory budget.
  - Overlapped transfers/compute with multiple CUDA streams.
- Pre-filtering
  - GPU max-of-three metric prefilter (e.g., >=85) reduces CPU classification load significantly.

### Result Processing
- Confidence Score
  - Internally computed on a 0–100 scale; exported/confidence field normalized to 0–1.
  - Case-specific averages derived from metrics; DIRECT MATCH equals 1.0.
- Thresholding
  - Typical filter at 95% (0.95) for export; adjustable by downstream consumers.

### Export Pipeline
- CSV/XLSX writers format matched pairs, include:
  - Source/target identifiers and names
  - Birthdates
  - Confidence and case label (e.g., CASE 1/2/3)
- XLSX writing tested with a minimal smoke test.

### Benchmarks & Measurement Guidance
- Included binary: src/bin/gpu_audit.rs
  - Seeds two MySQL tables with ~1,100 rows each if missing.
  - Runs Fuzzy matching in GPU mode and CPU baseline, reporting elapsed time.
- Recommended runs
  - Dataset sizes: 1k–10k rows per table (adjust seeding loop for larger tests).
  - Compare elapsed times before and after caching optimization.
- Expected outcome
  - Significant reduction in CPU post-processing time under GPU mode, with identical match results.

### Backward Compatibility
- Public behavior and outputs are unchanged.
- Classification rules are untouched; only redundant computations are removed via caching.
- GPU prefilter threshold and business rules remain identical.

