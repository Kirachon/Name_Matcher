# Algorithm 5 (HouseholdGpu) vs Algorithm 6 (HouseholdGpuOpt6) - Comprehensive Analysis

**Date**: 2025-09-30  
**Analysis Type**: Code Implementation Comparison  
**Location**: `src/matching/mod.rs` (lines 243-538)

---

## Executive Summary

**Algorithm 5 (HouseholdGpu)** and **Algorithm 6 (HouseholdGpuOpt6)** are **role-swapped variants** of household matching that use **identical GPU acceleration** but differ in their **grouping strategy and denominator calculation**. Despite the "Gpu" naming, both algorithms use GPU acceleration **conditionally** based on heuristics and configuration.

**Key Finding**: The "Opt6" suffix refers to **Option 6** (the 6th matching algorithm variant), NOT an optimization level. It represents a **strategic role swap** for different use cases.

---

## 1. Code Implementation Comparison

### 1.1 Function Signatures

**Algorithm 5**:
```rust
pub fn match_households_gpu_inmemory<F>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    fuzzy_min_conf: f32,
    on_progress: F,
) -> Vec<HouseholdAggRow>
```

**Algorithm 6**:
```rust
pub fn match_households_gpu_inmemory_opt6<F>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    fuzzy_min_conf: f32,
    on_progress: F,
) -> Vec<HouseholdAggRow>
```

**Observation**: Identical signatures - both take the same inputs and return the same output type.

---

### 1.2 Person-Level Matching (Step 1)

**Both algorithms use IDENTICAL matching logic**:

```rust
// Algorithm 5 (lines 282-313)
let pairs: Vec<MatchPair> = {
    #[cfg(feature = "gpu")]
    {
        let mut use_gpu = false;
        if matches!(opts.backend, ComputeBackend::Gpu) && gpu_fuzzy_metrics_enabled() && !gpu_fuzzy_disable() {
            if gpu_fuzzy_force() { use_gpu = true; }
            else {
                let (ok, why) = should_enable_gpu_fuzzy_by_heuristic(t1, t2);
                use_gpu = ok;
            }
        }
        if use_gpu {
            match gpu::match_fuzzy_no_mid_gpu(t1, t2, opts, &on_progress) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("[Algo5] GPU fuzzy (no-mid) failed, falling back to CPU: {}", e);
                    match_all_progress(t1, t2, MatchingAlgorithm::FuzzyNoMiddle, opts.progress, &on_progress)
                }
            }
        } else {
            crate::matching::match_fuzzy_no_mid_blocked_cpu(t1, t2, &on_progress)
        }
    }
}

// Algorithm 6 (lines 446-474) - IDENTICAL LOGIC
```

**Key Points**:
- Both use `gpu::match_fuzzy_no_mid_gpu()` for GPU acceleration
- Both use `match_fuzzy_no_mid_blocked_cpu()` for CPU fallback
- Both apply the same heuristics for GPU activation
- Both use **FuzzyNoMiddle** matching semantics (exact birthdate, fuzzy names without middle name)

---

### 1.3 Household Grouping (Step 2) - **THE KEY DIFFERENCE**

#### **Algorithm 5: Table 1 as Source**

```rust
// Lines 322-328
// Precompute total members per uuid (Table 1). Skip rows without uuid.
let mut totals: HashMap<String, usize> = HashMap::new();
for p in t1.iter() {
    if let Some(u) = p.uuid.as_ref() {
        *totals.entry(u.clone()).or_insert(0) += 1;
    }
}
```

**Grouping Key**: `uuid` from **Table 1**  
**Denominator**: Total members in **Table 1 household** (by uuid)

#### **Algorithm 6: Table 2 as Source**

```rust
// Lines 482-487
// Precompute total members per Table 2 household (hh_id fallback to id)
let mut totals_t2: HashMap<i64, usize> = HashMap::new();
for p in t2.iter() {
    let hh_key = p.hh_id.unwrap_or(p.id);
    *totals_t2.entry(hh_key).or_insert(0) += 1;
}
```

**Grouping Key**: `hh_id` (or `id` fallback) from **Table 2**  
**Denominator**: Total members in **Table 2 household** (by hh_id)

---

### 1.4 Best Match Selection (Step 3) - **ROLE SWAP**

#### **Algorithm 5: Select Best Household for Each Table 1 Person**

```rust
// Lines 348-368
// For each person in Table 1, select a single best household (by highest confidence)
let mut best_for_p1: HashMap<i64, (String, i64, f32, bool)> = HashMap::new(); // p1.id -> (uuid, hh_id, conf, tie)
for p in pairs.into_iter() {
    if p.confidence < fuzzy_min_conf { continue; }
    let Some(uuid) = p.person1.uuid.clone() else { continue; };
    let key = p.person1.id;  // Table 1 person ID
    let cand_hh = p.person2.hh_id.unwrap_or(p.person2.id);  // Table 2 household
    // ... select best match ...
}
```

**Logic**: For each **Table 1 person**, find the best **Table 2 household** match.

#### **Algorithm 6: Select Best Household for Each Table 2 Person**

```rust
// Lines 489-504
// For each person in Table 2, select a single best Table 1 household (uuid)
let mut best_for_p2: HashMap<i64, (i64, String, f32, bool)> = HashMap::new();
for p in pairs.into_iter() {
    if p.confidence < fuzzy_min_conf { continue; }
    let Some(uuid) = p.person1.uuid.clone() else { continue; };
    let hh_key = p.person2.hh_id.unwrap_or(p.person2.id);
    let key = p.person2.id;  // Table 2 person ID
    // ... select best match ...
}
```

**Logic**: For each **Table 2 person**, find the best **Table 1 household** match.

---

### 1.5 Match Percentage Calculation (Step 4) - **DENOMINATOR DIFFERENCE**

#### **Algorithm 5: Denominator = Table 1 Household Size**

```rust
// Lines 376-401
for ((uuid, hh_id), members) in matched.into_iter() {
    let total = *totals.get(&uuid).unwrap_or(&0usize) as f32;  // Table 1 household size
    if total <= 0.0 { continue; }
    let pct = (members.len() as f32) / total * 100.0;  // % of Table 1 household matched
    
    if pct > 50.0 {
        out.push(HouseholdAggRow {
            row_id,
            uuid,
            hh_id,
            match_percentage: pct,
            // ...
        });
    }
}
```

**Formula**: `match_percentage = (matched_members / table1_household_size) * 100`

#### **Algorithm 6: Denominator = Table 2 Household Size**

```rust
// Lines 513-532
for ((hh_key, uuid), members) in matched.into_iter() {
    let total = *totals_t2.get(&hh_key).unwrap_or(&0usize) as f32;  // Table 2 household size
    if total <= 0.0 { continue; }
    let pct = (members.len() as f32) / total * 100.0;  // % of Table 2 household matched
    
    if pct > 50.0 {
        out.push(HouseholdAggRow {
            row_id,
            uuid,
            hh_id: hh_key,
            match_percentage: pct,
            // ...
        });
    }
}
```

**Formula**: `match_percentage = (matched_members / table2_household_size) * 100`

---

## 2. GPU Utilization Analysis

### 2.1 GPU Usage Pattern

**Both algorithms use GPU acceleration IDENTICALLY**:

1. **Conditional GPU Activation**:
   ```rust
   if matches!(opts.backend, ComputeBackend::Gpu) && gpu_fuzzy_metrics_enabled() && !gpu_fuzzy_disable()
   ```

2. **Heuristic-Based Decision**:
   ```rust
   let (ok, why) = should_enable_gpu_fuzzy_by_heuristic(t1, t2);
   ```

3. **GPU Kernel Used**: `gpu::match_fuzzy_no_mid_gpu()`
   - Calls GPU Levenshtein kernel (A1 optimized with shared memory)
   - Calls GPU Jaro-Winkler kernel
   - Uses GPU blocking/filtering for candidate reduction

4. **CPU Fallback**: Both fall back to `match_fuzzy_no_mid_blocked_cpu()` if GPU fails or is disabled

### 2.2 Why "Gpu" in the Name?

**Answer**: The "Gpu" suffix indicates that these algorithms **support GPU acceleration** for the person-level fuzzy matching phase. However:

- GPU usage is **conditional** (not always active)
- GPU is used for **person-level matching** (Step 1)
- Household aggregation (Steps 2-4) is **CPU-only** (HashMap operations)

**Naming Convention**:
- `HouseholdGpu` = Household matching with GPU-accelerated person matching
- `HouseholdGpuOpt6` = Option 6 variant with GPU-accelerated person matching

---

## 3. Performance Characteristics

### 3.1 Computational Complexity

**Both algorithms have IDENTICAL complexity**:

| Phase | Complexity | Notes |
|-------|------------|-------|
| Person-level matching | O(n × m) | GPU-accelerated fuzzy matching |
| Household grouping | O(n) or O(m) | HashMap construction |
| Best match selection | O(k) | k = number of person-level matches |
| Aggregation | O(h) | h = number of unique household pairs |

**Total**: O(n × m) dominated by person-level matching

### 3.2 Memory Usage

**Algorithm 5**:
- Stores `totals: HashMap<String, usize>` (Table 1 households by uuid)
- Stores `best_for_p1: HashMap<i64, (String, i64, f32, bool)>` (Table 1 person → best match)

**Algorithm 6**:
- Stores `totals_t2: HashMap<i64, usize>` (Table 2 households by hh_id)
- Stores `best_for_p2: HashMap<i64, (i64, String, f32, bool)>` (Table 2 person → best match)

**Memory Difference**: Negligible (both use similar HashMap structures)

### 3.3 Speed Comparison

**Expected Performance**: **IDENTICAL**

Both algorithms:
- Use the same GPU kernels
- Process the same number of person-level matches
- Perform similar HashMap operations
- Have the same O(n × m) complexity

**Benchmark Prediction**: Within 1-2% of each other (measurement noise)

---

## 4. Trade-offs and Use Cases

### 4.1 When to Use Algorithm 5 (HouseholdGpu)

**Use Case**: When **Table 1 is the authoritative source** and you want to measure:
> "What percentage of each Table 1 household has matches in Table 2?"

**Example Scenario**:
- Table 1 = Census data (authoritative household definitions)
- Table 2 = Survey data (may have incomplete households)
- **Question**: "For each census household, what % of members appear in the survey?"

**Output Interpretation**:
- `match_percentage = 75%` means **75% of the census household members** were found in the survey

**Advantages**:
- Denominator based on **authoritative source** (Table 1)
- Useful for **coverage analysis** (how well does Table 2 cover Table 1?)

---

### 4.2 When to Use Algorithm 6 (HouseholdGpuOpt6)

**Use Case**: When **Table 2 is the authoritative source** and you want to measure:
> "What percentage of each Table 2 household has matches in Table 1?"

**Example Scenario**:
- Table 1 = Survey data (may have incomplete households)
- Table 2 = Administrative records (authoritative household definitions)
- **Question**: "For each administrative household, what % of members appear in the survey?"

**Output Interpretation**:
- `match_percentage = 75%` means **75% of the administrative household members** were found in the survey

**Advantages**:
- Denominator based on **authoritative source** (Table 2)
- Useful for **reverse coverage analysis** (how well does Table 1 cover Table 2?)

---

### 4.3 Practical Example

**Scenario**: Census (Table 1) vs Survey (Table 2)

**Table 1 (Census) - Household A**:
- Person 1: John Doe (uuid=A)
- Person 2: Jane Doe (uuid=A)
- Person 3: Jimmy Doe (uuid=A)
- **Total**: 3 members

**Table 2 (Survey) - Household B**:
- Person 4: John Doe (hh_id=B)
- Person 5: Jane Doe (hh_id=B)
- **Total**: 2 members

**Person-Level Matches** (both algorithms find the same matches):
- John Doe (Table 1) ↔ John Doe (Table 2) - confidence 95%
- Jane Doe (Table 1) ↔ Jane Doe (Table 2) - confidence 95%

**Algorithm 5 Result**:
- Matched members: 2 (John, Jane)
- Total Table 1 household: 3
- **match_percentage = 2/3 × 100 = 66.7%**
- **Interpretation**: "66.7% of the census household was found in the survey"

**Algorithm 6 Result**:
- Matched members: 2 (John, Jane)
- Total Table 2 household: 2
- **match_percentage = 2/2 × 100 = 100%**
- **Interpretation**: "100% of the survey household was found in the census"

---

## 5. Implementation Details

### 5.1 Tie Handling

**Both algorithms handle ties identically**:

```rust
// If multiple households have the same top confidence, mark as tie and skip
if (p.confidence - *conf).abs() < f32::EPSILON && cand_hh != *hh {
    *tie = true;  // Ambiguous assignment
}

// Later: skip tied assignments
if tie { continue; }
```

**Rationale**: Prevents double-counting when a person could belong to multiple households with equal confidence.

### 5.2 Confidence Threshold

**Both algorithms apply the same threshold**:

```rust
if p.confidence < fuzzy_min_conf { continue; }
```

**Default**: `fuzzy_min_conf` is typically 85-95% (configurable)

### 5.3 Output Format

**Both algorithms produce identical output structure**:

```rust
pub struct HouseholdAggRow {
    pub row_id: i64,
    pub uuid: String,        // Table 1 household ID
    pub hh_id: i64,          // Table 2 household ID
    pub match_percentage: f32,
    pub region_code: Option<String>,
    pub poor_hat_0: Option<String>,
    pub poor_hat_10: Option<String>,
}
```

**Note**: The `uuid` and `hh_id` fields have the same meaning in both algorithms, but the `match_percentage` calculation differs.

---

## 6. Summary Table

| Aspect | Algorithm 5 (HouseholdGpu) | Algorithm 6 (HouseholdGpuOpt6) |
|--------|----------------------------|--------------------------------|
| **GPU Usage** | ✅ Conditional (person-level matching) | ✅ Conditional (person-level matching) |
| **GPU Kernels** | Levenshtein, Jaro-Winkler | Levenshtein, Jaro-Winkler |
| **Grouping Source** | Table 1 (uuid) | Table 2 (hh_id) |
| **Denominator** | Table 1 household size | Table 2 household size |
| **Best Match Selection** | For each Table 1 person | For each Table 2 person |
| **Match Percentage** | % of Table 1 household matched | % of Table 2 household matched |
| **Use Case** | Table 1 is authoritative | Table 2 is authoritative |
| **Complexity** | O(n × m) | O(n × m) |
| **Memory Usage** | ~O(n + m) | ~O(n + m) |
| **Speed** | Identical | Identical |

---

## 7. Conclusion

### Key Findings

1. **"Opt6" Meaning**: Refers to **Option 6** (the 6th algorithm variant), NOT an optimization level
2. **GPU Usage**: Both algorithms use GPU acceleration **identically** for person-level matching
3. **Core Difference**: **Role swap** - Algorithm 5 groups by Table 1, Algorithm 6 groups by Table 2
4. **Performance**: **Identical** speed and memory usage
5. **Use Case**: Choose based on which table is the **authoritative source** for household definitions

### Recommendations

**Use Algorithm 5 when**:
- Table 1 has authoritative household definitions
- You want to measure coverage of Table 1 in Table 2
- Example: Census → Survey matching

**Use Algorithm 6 when**:
- Table 2 has authoritative household definitions
- You want to measure coverage of Table 2 in Table 1
- Example: Survey → Administrative records matching

**Performance**: No performance difference - choose based on **semantic requirements**, not speed.

---

**Analysis Date**: 2025-09-30  
**Code Version**: Latest (with D1 optimization)  
**Analyst**: Augment Agent (Claude Sonnet 4.5)

