use crate::models::{NormalizedPerson, Person};
use crate::normalize::normalize_person;
use crate::metrics::memory_stats_mb;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use strsim::{levenshtein, jaro_winkler};
use rphonetic::{DoubleMetaphone, Encoder};
use unicode_normalization::UnicodeNormalization;

#[cfg(feature = "gpu")]
use chrono::Datelike;


fn normalize_simple(s: &str) -> String {
    let s = s.trim();
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '.' => { /* drop dot */ }
            '-' => out.push(' '),
            _ => {
                for lc in ch.to_lowercase() { out.push(lc); }
            }
        }
    }
    out
}

fn sim_levenshtein_pct(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 { return 100.0; }
    let dist = levenshtein(a, b);
    (1.0 - (dist as f64 / max_len as f64)) * 100.0
}

fn normalize_for_phonetic(s: &str) -> String {
    // Decompose diacritics, keep ASCII letters and single spaces; map a few common non-ASCII
    // Do lowercasing inline to avoid an intermediate allocation
    let s = s.trim();
    let mut out = String::with_capacity(s.len());
    for ch in s.nfd() {
        // Lowercase the codepoint; may yield 1..N chars
        for lc in ch.to_lowercase() {
            if lc.is_ascii_alphabetic() {
                out.push(lc);
            } else if lc.is_ascii_whitespace() {
                if !out.ends_with(' ') { out.push(' '); }
            } else {
                match lc {
                    'ß' => out.push_str("ss"),
                    'æ' | 'ǽ' => out.push_str("ae"),
                    'ø' => out.push('o'),
                    'đ' => out.push('d'),
                    _ => {}
                }
            }
        }
    }
    // Trim trailing space in-place
    let new_len = out.trim_end().len();
    out.truncate(new_len);
    out
}

fn metaphone_pct(a: &str, b: &str) -> f64 {
    let sa = normalize_for_phonetic(a);
    let sb = normalize_for_phonetic(b);
    if sa.is_empty() || sb.is_empty() { return 0.0; }

    // Protect against panics inside rphonetic by catching unwinds
    let ra = std::panic::catch_unwind(|| DoubleMetaphone::default().encode(&sa));
    let rb = std::panic::catch_unwind(|| DoubleMetaphone::default().encode(&sb));
    let (ra_s, rb_s) = match (ra, rb) {
        (Ok(ra), Ok(rb)) => (ra.to_string(), rb.to_string()),
        _ => {
            log::warn!("DoubleMetaphone panicked on inputs: {:?} / {:?}", a, b);
            return 0.0;
        }
    };
    if !ra_s.is_empty() && ra_s == rb_s { 100.0 } else { 0.0 }
}

fn fuzzy_compare_names_new(n1_first: Option<&str>, n1_mid: Option<&str>, n1_last: Option<&str>, n2_first: Option<&str>, n2_mid: Option<&str>, n2_last: Option<&str>) -> Option<(f64, String)> {
    let full1 = normalize_simple(&format!("{} {} {}", n1_first.unwrap_or(""), n1_mid.unwrap_or(""), n1_last.unwrap_or("")));
    let full2 = normalize_simple(&format!("{} {} {}", n2_first.unwrap_or(""), n2_mid.unwrap_or(""), n2_last.unwrap_or("")));
    if full1.trim().is_empty() || full2.trim().is_empty() { return None; }

    let lev = sim_levenshtein_pct(&full1, &full2);
    let jw = jaro_winkler(&full1, &full2) * 100.0;
    let mp = metaphone_pct(&full1, &full2);

    // Direct match
    if full1 == full2 {
        return Some((100.0, "DIRECT MATCH".to_string()));
    }

    // Case 1
    if lev >= 85.0 && jw >= 85.0 && (mp - 100.0).abs() < f64::EPSILON {
        let avg = (lev + jw + mp) / 3.0;
        return Some((avg, "CASE 1".to_string()));
    }

    // Case 2
    let mut pass = 0;
    if lev >= 85.0 { pass += 1; }
    if jw >= 85.0 { pass += 1; }
    if (mp - 100.0).abs() < f64::EPSILON { pass += 1; }
    if pass >= 2 {
        let avg = (lev + jw + mp) / 3.0;
        // Case 3 refinement
        if avg >= 88.0 {
            let ld_first = levenshtein(&normalize_simple(n1_first.unwrap_or("")), &normalize_simple(n2_first.unwrap_or(""))) as usize;
            let ld_last = levenshtein(&normalize_simple(n1_last.unwrap_or("")), &normalize_simple(n2_last.unwrap_or(""))) as usize;
            let ld_mid = levenshtein(&normalize_simple(n1_mid.unwrap_or("")), &normalize_simple(n2_mid.unwrap_or(""))) as usize;
            if ld_first <= 2 && ld_last <= 2 && ld_mid <= 2 {
                return Some((avg, "CASE 3".to_string()));
            }
        }
        return Some((avg, "CASE 2".to_string()));
    }

    None
}

#[allow(dead_code)]
fn compare_persons_new(p1: &Person, p2: &Person) -> Option<(f64, String)> {
    match (p1.birthdate.as_ref(), p2.birthdate.as_ref()) {
        (Some(d1), Some(d2)) if d1 == d2 => {}
        _ => return None,
    }
    fuzzy_compare_names_new(
        p1.first_name.as_deref(), p1.middle_name.as_deref(), p1.last_name.as_deref(),
        p2.first_name.as_deref(), p2.middle_name.as_deref(), p2.last_name.as_deref(),
    )
}

fn compare_persons_no_mid(p1: &Person, p2: &Person) -> Option<(f64, String)> {
    match (p1.birthdate.as_ref(), p2.birthdate.as_ref()) {
        (Some(d1), Some(d2)) if d1 == d2 => {}
        _ => return None,
    }
    fuzzy_compare_names_no_mid(
        p1.first_name.as_deref(), p1.last_name.as_deref(),
        p2.first_name.as_deref(), p2.last_name.as_deref(),
    )
}


fn fuzzy_compare_names_no_mid(n1_first: Option<&str>, n1_last: Option<&str>, n2_first: Option<&str>, n2_last: Option<&str>) -> Option<(f64, String)> {
    let full1 = normalize_simple(&format!("{} {}", n1_first.unwrap_or(""), n1_last.unwrap_or("")));
    let full2 = normalize_simple(&format!("{} {}", n2_first.unwrap_or(""), n2_last.unwrap_or("")));
    if full1.trim().is_empty() || full2.trim().is_empty() { return None; }

    let lev = sim_levenshtein_pct(&full1, &full2);
    let jw = jaro_winkler(&full1, &full2) * 100.0;
    let mp = metaphone_pct(&full1, &full2);

    if full1 == full2 { return Some((100.0, "DIRECT MATCH".to_string())); }

    if lev >= 85.0 && jw >= 85.0 && (mp - 100.0).abs() < f64::EPSILON {
        let avg = (lev + jw + mp) / 3.0; return Some((avg, "CASE 1".to_string()));
    }

    let mut pass = 0; if lev >= 85.0 { pass += 1; } if jw >= 85.0 { pass += 1; } if (mp - 100.0).abs() < f64::EPSILON { pass += 1; }
    if pass >= 2 {
        let avg = (lev + jw + mp) / 3.0;
        if avg >= 88.0 {
            let ld_first = levenshtein(&normalize_simple(n1_first.unwrap_or("")), &normalize_simple(n2_first.unwrap_or(""))) as usize;
            let ld_last  = levenshtein(&normalize_simple(n1_last.unwrap_or("")),  &normalize_simple(n2_last.unwrap_or("")))  as usize;
            if ld_first <= 2 && ld_last <= 2 { return Some((avg, "CASE 3".to_string())); }
        }
        return Some((avg, "CASE 2".to_string()));
    }
    None
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MatchingAlgorithm {
    IdUuidYasIsMatchedInfnbd,
    IdUuidYasIsMatchedInfnmnbd,
    Fuzzy,
    FuzzyNoMiddle,
    // New Option 5: GPU in-memory household matching
    HouseholdGpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HouseholdAggRow {
    pub row_id: i64,
    pub uuid: String,   // Table 1 household ID
    pub hh_id: i64,     // Table 2 household ID (mapped to `id`)
    pub match_percentage: f32,
    // Optional fields (if later fetched; currently None when unavailable)
    pub region_code: Option<String>,
    pub poor_hat_0: Option<String>,
    pub poor_hat_10: Option<String>,
}

/// Option 5: In-memory GPU household matching with birthdate hard filter and names-only similarity.
/// Produces aggregated household rows grouped by (uuid from Table 1, hh_id from Table 2).
pub fn match_households_gpu_inmemory<F>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    fuzzy_min_conf: f32,
    on_progress: F,
) -> Vec<HouseholdAggRow>
where
    F: Fn(ProgressUpdate) + Sync,
{
    use std::collections::{HashMap, HashSet};
    // 1) Generate person-level matches using existing FuzzyNoMiddle semantics with exact birthdate
    let pairs: Vec<MatchPair> = {
        #[cfg(feature = "gpu")]
        {
            match gpu::match_fuzzy_no_mid_gpu(t1, t2, opts, &on_progress) {
                Ok(v) => v,
                Err(_e) => {
                    // CPU fallback
                    match_all_progress(t1, t2, MatchingAlgorithm::FuzzyNoMiddle, opts.progress, &on_progress)
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            match_all_progress(t1, t2, MatchingAlgorithm::FuzzyNoMiddle, opts.progress, &on_progress)
        }
    };

    if pairs.is_empty() {
        return Vec::new();
    }

    // 2) Precompute total members per uuid (Table 1). Skip rows without uuid.
    let mut totals: HashMap<String, usize> = HashMap::new();
    for p in t1.iter() {
        if let Some(u) = p.uuid.as_ref() {
            *totals.entry(u.clone()).or_insert(0) += 1;
        }
    }

    // DEBUG: Log some household sizes and check for UUID truncation
    let mut total_count = 0;
    for (uuid, count) in totals.iter().take(5) {
        eprintln!("DEBUG: Table 1 household {} (len={}) has {} members", uuid, uuid.len(), count);
        total_count += 1;
    }
    eprintln!("DEBUG: Total unique households in Table 1: {}", totals.len());

    // DEBUG: Check first few raw persons to see their UUIDs
    for (i, p) in t1.iter().take(10).enumerate() {
        if let Some(uuid) = &p.uuid {
            eprintln!("DEBUG: Person {}: id={}, uuid={} (len={}), name={} {}",
                     i, p.id, uuid, uuid.len(),
                     p.first_name.as_deref().unwrap_or(""),
                     p.last_name.as_deref().unwrap_or(""));
        }
    }

    // 3) For each person in Table 1, select a single best household (by highest confidence) to avoid double counting across households.
    //    If there is a tie for top confidence across different households, skip counting that person to be conservative.
    let mut best_for_p1: HashMap<i64, (String, i64, f32, bool)> = HashMap::new(); // p1.id -> (uuid, hh_id, conf, tie)
    for p in pairs.into_iter() {
        if p.confidence < fuzzy_min_conf { continue; }
        let Some(uuid) = p.person1.uuid.clone() else { continue; };
        let key = p.person1.id;
        let cand_hh = p.person2.id; // Table 2 household id maps to `id`
        match best_for_p1.get_mut(&key) {
            None => { best_for_p1.insert(key, (uuid, cand_hh, p.confidence, false)); }
            Some((u, hh, conf, tie)) => {
                if p.confidence > *conf {
                    *u = uuid; *hh = cand_hh; *conf = p.confidence; *tie = false;
                } else if (p.confidence - *conf).abs() < f32::EPSILON && cand_hh != *hh {
                    // ambiguous top-1 across different households
                    *tie = true;
                }
            }
        }
    }
    let mut matched: HashMap<(String, i64), HashSet<i64>> = HashMap::new();
    for (p1_id, (uuid, hh_id, _conf, tie)) in best_for_p1.into_iter() {
        if tie { continue; } // skip ambiguous assignments
        matched.entry((uuid, hh_id)).or_default().insert(p1_id);
    }

    // 4) Compute match_percentage and filter > 50%
    let mut out: Vec<HouseholdAggRow> = Vec::new();
    let mut row_id: i64 = 1;
    for ((uuid, hh_id), members) in matched.into_iter() {
        let total = *totals.get(&uuid).unwrap_or(&0usize) as f32;
        if total <= 0.0 { continue; }
        let pct = (members.len() as f32) / total * 100.0;

        // DEBUG: Log the calculation details for first few households
        if row_id <= 5 {
            eprintln!("DEBUG: uuid={}, hh_id={}, matched_members={}, total_members={}, percentage={:.2}%",
                     uuid, hh_id, members.len(), total, pct);
        }

        if pct > 50.0 {
            out.push(HouseholdAggRow {
                row_id,
                uuid,
                hh_id,
                match_percentage: pct,
                region_code: None,
                poor_hat_0: None,
                poor_hat_10: None,
            });
            row_id += 1;
        }
    }

    // Sort for deterministic output (by uuid then hh_id)
    out.sort_by(|a, b| a.uuid.cmp(&b.uuid).then_with(|| a.hh_id.cmp(&b.hh_id)));
    out
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPair { pub person1: Person, pub person2: Person, pub confidence: f32, pub matched_fields: Vec<String>, pub is_matched_infnbd: bool, pub is_matched_infnmnbd: bool }

#[derive(Debug, Clone, Copy)]
pub struct ProgressConfig { pub update_every: usize, pub long_op_threshold: Duration, pub batch_size: Option<usize> }
impl Default for ProgressConfig { fn default() -> Self { Self { update_every: 1000, long_op_threshold: Duration::from_secs(30), batch_size: None } } }

#[derive(Debug, Clone, Copy)]
pub struct ProgressUpdate {
    pub processed: usize,
    pub total: usize,
    pub percent: f32,
    pub eta_secs: u64,
    pub mem_used_mb: u64,
    pub mem_avail_mb: u64,
    #[allow(dead_code)]
    pub stage: &'static str,
    #[allow(dead_code)]
    pub batch_size_current: Option<i64>,
    // GPU-related (0/false when CPU-only)
    #[allow(dead_code)]
    pub gpu_total_mb: u64,
    #[allow(dead_code)]
    pub gpu_free_mb: u64,
    #[allow(dead_code)]
    pub gpu_active: bool,
}


// --- Optional GPU backend abstraction ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ComputeBackend { Cpu, Gpu }

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct GpuConfig { pub device_id: Option<usize>, pub mem_budget_mb: u64 }

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MatchOptions { pub backend: ComputeBackend, pub gpu: Option<GpuConfig>, pub progress: ProgressConfig }

impl Default for MatchOptions {
    fn default() -> Self {
        Self { backend: ComputeBackend::Cpu, gpu: None, progress: ProgressConfig::default() }
    }
}

#[allow(dead_code)]
pub fn match_all_with_opts<F>(table1: &[Person], table2: &[Person], algo: MatchingAlgorithm, opts: MatchOptions, progress: F) -> Vec<MatchPair>
where F: Fn(ProgressUpdate) + Sync,
{
    // Fast path: deterministic A1/A2 with GPU-accelerated hashing in in-memory mode
    if matches!(algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd)
        && matches!(opts.backend, ComputeBackend::Gpu)
    {
        #[cfg(feature = "gpu")]
        {
            match gpu::det_match_gpu_hash_inmemory(table1, table2, algo, &opts, &progress) {
                Ok(v) => return v,
                Err(e) => { log::warn!("GPU in-memory hash (A1/A2) failed, falling back to CPU: {}", e); }
            }
        }
        // If feature disabled or GPU path failed, fall through to CPU
    }

    if matches!(algo, MatchingAlgorithm::Fuzzy) && matches!(opts.backend, ComputeBackend::Gpu) {
        // Optional in-memory GPU pre-pass before full fuzzy scoring
        #[cfg(feature = "gpu")]
        if gpu_fuzzy_prep_enabled() {
            progress(ProgressUpdate { processed: 0, total: table1.len().max(1), percent: 0.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_hash", batch_size_current: None, gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
            match gpu::fuzzy_direct_gpu_hash_prefilter_indices(table1, table2, "last_initial") {
                Ok(cand_lists) => {
                    progress(ProgressUpdate { processed: 0, total: table1.len().max(1), percent: 0.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_probe_hash", batch_size_current: None, gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
                    let n1: Vec<NormalizedPerson> = table1.par_iter().map(normalize_person).collect();
                    let n2: Vec<NormalizedPerson> = table2.par_iter().map(normalize_person).collect();
                    let mut out: Vec<MatchPair> = Vec::new();
                    for (i, p) in table1.iter().enumerate() {
                        let n = &n1[i];
                        for &j in cand_lists.get(i).map(|v| v.as_slice()).unwrap_or(&[]) {
                            let n2p = &n2[j];
                            if !n.birthdate.as_ref().zip(n2p.birthdate.as_ref()).map_or(false, |(a,b)| a==b) { continue; }
                            if let Some((score, label)) = fuzzy_compare_names_new(
                                n.first_name.as_deref(), n.middle_name.as_deref(), n.last_name.as_deref(),
                                n2p.first_name.as_deref(), n2p.middle_name.as_deref(), n2p.last_name.as_deref(),
                            ) {
                                let q = &table2[j];
                                let pair = MatchPair { person1: p.clone(), person2: q.clone(), confidence: (score/100.0) as f32, matched_fields: vec!["fuzzy".into(), label, "birthdate".into()], is_matched_infnbd: false, is_matched_infnmnbd: false };
                                out.push(pair);
                            }
                        }
                    }
                    progress(ProgressUpdate { processed: out.len(), total: table1.len().max(1), percent: 100.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_probe_hash_done", batch_size_current: None, gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
                    return out;
                }
                Err(e) => { log::warn!("GPU fuzzy direct pre-pass (in-memory) failed; proceeding with full GPU fuzzy: {}", e); }
            }
        }
        #[cfg(feature = "gpu")]
        {
            match gpu::match_fuzzy_gpu(table1, table2, opts, &progress) {
                Ok(v) => return v,
                Err(e) => { log::warn!("GPU fuzzy failed, falling back to CPU: {}", e); }
            }
        }
        // If feature disabled or GPU path failed, fall through to CPU
    }
    // GPU in-memory path for Option 4 (FuzzyNoMiddle)
    if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) && matches!(opts.backend, ComputeBackend::Gpu) {
        #[cfg(feature = "gpu")]
        if gpu_fuzzy_prep_enabled() {
            progress(ProgressUpdate { processed: 0, total: table1.len().max(1), percent: 0.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_hash", batch_size_current: None, gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
            match gpu::fuzzy_direct_gpu_hash_prefilter_indices(table1, table2, "last_initial") {
                Ok(cand_lists) => {
                    progress(ProgressUpdate { processed: 0, total: table1.len().max(1), percent: 0.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_probe_hash", batch_size_current: None, gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
                    let n1: Vec<NormalizedPerson> = table1.par_iter().map(normalize_person).collect();
                    let n2: Vec<NormalizedPerson> = table2.par_iter().map(normalize_person).collect();
                    let mut out: Vec<MatchPair> = Vec::new();
                    for (i, p) in table1.iter().enumerate() {
                        let n = &n1[i];
                        for &j in cand_lists.get(i).map(|v| v.as_slice()).unwrap_or(&[]) {
                            let n2p = &n2[j];
                            if !n.birthdate.as_ref().zip(n2p.birthdate.as_ref()).map_or(false, |(a,b)| a==b) { continue; }
                            if let Some((score, label)) = fuzzy_compare_names_no_mid(
                                n.first_name.as_deref(), n.last_name.as_deref(),
                                n2p.first_name.as_deref(), n2p.last_name.as_deref(),
                            ) {
                                let q = &table2[j];
                                let pair = MatchPair { person1: p.clone(), person2: q.clone(), confidence: (score/100.0) as f32, matched_fields: vec!["fuzzy".into(), label, "birthdate".into()], is_matched_infnbd: false, is_matched_infnmnbd: false };
                                out.push(pair);
                            }
                        }
                    }
                    progress(ProgressUpdate { processed: out.len(), total: table1.len().max(1), percent: 100.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_probe_hash_done", batch_size_current: None, gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
                    return out;
                }
                Err(e) => { log::warn!("GPU fuzzy direct pre-pass (in-memory; no-mid) failed; proceeding with full GPU fuzzy no-mid: {}", e); }
            }
        }
        #[cfg(feature = "gpu")]
        {
            match gpu::match_fuzzy_no_mid_gpu(table1, table2, opts, &progress) {
                Ok(v) => return v,
                Err(e) => { log::warn!("GPU fuzzy (no-mid) failed, falling back to CPU: {}", e); }
            }
        }
        // If feature disabled or GPU path failed, fall through to CPU
    }
    match_all_progress(table1, table2, algo, opts.progress, progress)
}

/// Global toggle: enable GPU fuzzy direct pre-pass in in-memory matching paths as well.
static GPU_FUZZY_PREPASS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_gpu_fuzzy_direct_prep(enabled: bool) { GPU_FUZZY_PREPASS.store(enabled, std::sync::atomic::Ordering::Relaxed); }
#[inline]
fn gpu_fuzzy_prep_enabled() -> bool { GPU_FUZZY_PREPASS.load(std::sync::atomic::Ordering::Relaxed) }

/// Global toggle: when true, Algorithms 1 & 2 use fuzzy-style normalization (drop periods, hyphens->spaces) before equality checks.
static DIRECT_NORM_FUZZY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_direct_normalization_fuzzy(enabled: bool) { DIRECT_NORM_FUZZY.store(enabled, std::sync::atomic::Ordering::Relaxed); }
#[inline]
fn direct_norm_fuzzy_enabled() -> bool { DIRECT_NORM_FUZZY.load(std::sync::atomic::Ordering::Relaxed) }


fn matches_algo1(p1: &NormalizedPerson, p2: &NormalizedPerson) -> bool {
    let date_ok  = p1.birthdate.as_ref().zip(p2.birthdate.as_ref()).map_or(false, |(a,b)| a == b);
    if !date_ok { return false; }
    if direct_norm_fuzzy_enabled() {
        let a_first = p1.first_name.as_deref().map(normalize_simple);
        let b_first = p2.first_name.as_deref().map(normalize_simple);
        let a_last  = p1.last_name.as_deref().map(normalize_simple);
        let b_last  = p2.last_name.as_deref().map(normalize_simple);
        a_first.as_deref() == b_first.as_deref() && a_last.as_deref() == b_last.as_deref()
    } else {
        let first_ok = p1.first_name.as_ref().zip(p2.first_name.as_ref()).map_or(false, |(a,b)| a == b);
        let last_ok  = p1.last_name.as_ref().zip(p2.last_name.as_ref()).map_or(false, |(a,b)| a == b);
        first_ok && last_ok
    }
}
fn matches_algo2(p1: &NormalizedPerson, p2: &NormalizedPerson) -> bool {
    let date_ok  = p1.birthdate.as_ref().zip(p2.birthdate.as_ref()).map_or(false, |(a,b)| a == b);
    if !date_ok { return false; }
    if direct_norm_fuzzy_enabled() {
        let a_first = p1.first_name.as_deref().map(normalize_simple);
        let b_first = p2.first_name.as_deref().map(normalize_simple);
        let a_last  = p1.last_name.as_deref().map(normalize_simple);
        let b_last  = p2.last_name.as_deref().map(normalize_simple);
        let a_mid   = p1.middle_name.as_deref().map(normalize_simple);
        let b_mid   = p2.middle_name.as_deref().map(normalize_simple);
        let middle_ok = match (a_mid.as_deref(), b_mid.as_deref()) { (Some(a), Some(b)) => a == b, (None, None) => true, _ => false };
        a_first.as_deref() == b_first.as_deref() && a_last.as_deref() == b_last.as_deref() && middle_ok
    } else {
        let first_ok = p1.first_name.as_ref().zip(p2.first_name.as_ref()).map_or(false, |(a,b)| a == b);
        let last_ok  = p1.last_name.as_ref().zip(p2.last_name.as_ref()).map_or(false, |(a,b)| a == b);
        let middle_ok = match (&p1.middle_name, &p2.middle_name) { (Some(a), Some(b)) => a == b, (None, None) => true, _ => false };
        first_ok && last_ok && middle_ok
    }
}

#[allow(dead_code)]
pub fn match_all<F>(table1: &[Person], table2: &[Person], algo: MatchingAlgorithm, progress: F) -> Vec<MatchPair>
where F: Fn(f32) + Sync,
{ match_all_progress(table1, table2, algo, ProgressConfig::default(), |u| progress(u.percent)) }

pub fn match_all_progress<F>(table1: &[Person], table2: &[Person], algo: MatchingAlgorithm, cfg: ProgressConfig, progress: F) -> Vec<MatchPair>
where F: Fn(ProgressUpdate) + Sync,
{
    let start = Instant::now();
    let norm1: Vec<NormalizedPerson> = table1.par_iter().map(normalize_person).collect();
    let norm2: Vec<NormalizedPerson> = table2.par_iter().map(normalize_person).collect();
    let total = norm1.len();
    if total == 0 || norm2.is_empty() { return Vec::new(); }
    let threads = rayon::current_num_threads().max(1);
    let auto_batch = (total / (threads * 4)).clamp(100, 10_000).max(1);
    let batch_size = cfg.batch_size.unwrap_or(auto_batch);

    let mut results: Vec<MatchPair> = Vec::new();
    let mut processed_outer = 0usize; let mut last_update = 0usize;

    for chunk in norm1.chunks(batch_size) {
        let chunk_start = Instant::now();
        let batch_res: Vec<MatchPair> = chunk.par_iter().flat_map(|p1| {
            norm2.par_iter().filter_map(|p2| {
                match algo {
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                        if matches_algo1(p1, p2) {
                            Some(MatchPair {
                                person1: to_original(p1, table1), person2: to_original(p2, table2), confidence: 1.0,
                                matched_fields: vec!["id","uuid","first_name","last_name","birthdate"].into_iter().map(String::from).collect(),
                                is_matched_infnbd: true, is_matched_infnmnbd: false,
                            })
                        } else { None }
                    }
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                        if matches_algo2(p1, p2) {
                            Some(MatchPair {
                                person1: to_original(p1, table1), person2: to_original(p2, table2), confidence: 1.0,
                                matched_fields: vec!["id","uuid","first_name","middle_name","last_name","birthdate"].into_iter().map(String::from).collect(),
                                is_matched_infnbd: false, is_matched_infnmnbd: true,
                            })
                        } else { None }
                    }
                    MatchingAlgorithm::Fuzzy => {
                        if p1.birthdate.as_ref().zip(p2.birthdate.as_ref()).map_or(false, |(a,b)| a==b) {
                            if let Some((score, label)) = fuzzy_compare_names_new(
                                p1.first_name.as_deref(),
                                p1.middle_name.as_deref(),
                                p1.last_name.as_deref(),
                                p2.first_name.as_deref(),
                                p2.middle_name.as_deref(),
                                p2.last_name.as_deref(),
                            ) {
                                Some(MatchPair {
                                    person1: to_original(p1, table1),
                                    person2: to_original(p2, table2),
                                    confidence: (score / 100.0) as f32,
                                    matched_fields: vec!["fuzzy".into(), label.into(), "birthdate".into()],
                                    is_matched_infnbd: false, is_matched_infnmnbd: false,
                                })
                            } else { None }
                        } else { None }
                    }
                    MatchingAlgorithm::FuzzyNoMiddle => {
                        if p1.birthdate.as_ref().zip(p2.birthdate.as_ref()).map_or(false, |(a,b)| a==b) {
                            if let Some((score, label)) = fuzzy_compare_names_no_mid(
                                p1.first_name.as_deref(),
                                p1.last_name.as_deref(),
                                p2.first_name.as_deref(),
                                p2.last_name.as_deref(),
                            ) {
                                Some(MatchPair {
                                    person1: to_original(p1, table1),
                                    person2: to_original(p2, table2),
                                    confidence: (score / 100.0) as f32,
                                    matched_fields: vec!["fuzzy".into(), label.into(), "birthdate".into()],
                                    is_matched_infnbd: false, is_matched_infnmnbd: false,
                                })
                            } else { None }
                        } else { None }
                    }
                        MatchingAlgorithm::HouseholdGpu => { None }

                }
            }).collect::<Vec<_>>()
        }).collect();
        results.extend(batch_res);
        processed_outer = (processed_outer + chunk.len()).min(total);
        if processed_outer - last_update >= cfg.update_every || processed_outer == total {
            let elapsed = start.elapsed();
            let frac = (processed_outer as f32 / total as f32).clamp(0.0, 1.0);
            let eta_secs = if frac > 0.0 { (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
            let mem = memory_stats_mb();
            progress(ProgressUpdate { processed: processed_outer, total, percent: frac * 100.0, eta_secs, mem_used_mb: mem.used_mb, mem_avail_mb: mem.avail_mb, stage: "matching", batch_size_current: None, gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
            last_update = processed_outer;
        }
        if chunk_start.elapsed() > cfg.long_op_threshold { let _m = memory_stats_mb(); }
    }

    results
}

// --- GPU module (feature-gated) ---
#[cfg(feature = "gpu")]
    // Helpers: memory info (best-effort) and simple Soundex for blocking
    #[inline]
    fn soundex4_ascii(s: &str) -> [u8;4] {
        let mut out = [b'0';4]; if s.is_empty() { return out; }
        let mut bytes = s.as_bytes().iter().copied().filter(|c| c.is_ascii_alphabetic());
        if let Some(f) = bytes.next() { out[0] = f.to_ascii_uppercase(); }
        let mut last = 0u8; let mut idx = 1usize;
        for c in bytes { if idx>=4 { break; }
            let d = match c.to_ascii_lowercase() { b'b'|b'f'|b'p'|b'v'=>1, b'c'|b'g'|b'j'|b'k'|b'q'|b's'|b'x'|b'z'=>2, b'd'|b't'=>3, b'l'=>4, b'm'|b'n'=>5, b'r'=>6, _=>0 };
            if d!=0 && d!=last { out[idx]=b'0'+d; idx+=1; }
            last=d;
        }
        out
    }

    #[cfg(feature = "gpu")]
    fn cuda_mem_info_mb(_ctx: &cudarc::driver::CudaContext) -> (u64,u64) {
        // Query using CUDA driver API; context must be current.
        unsafe {
            use cudarc::driver::sys::CUresult;
            let mut free: usize = 0; let mut total: usize = 0;
            let res = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _ as *mut _, &mut total as *mut _ as *mut _);
            if res == CUresult::CUDA_SUCCESS { ((total as u64)/1024/1024, (free as u64)/1024/1024) } else { (0,0) }
        }
    }

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use anyhow::{anyhow, Result};
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    const MAX_STR: usize = 64; // truncate names for GPU DP to keep registers/local mem bounded

    // CUDA kernel source for per-pair Levenshtein (two-row DP; lengths capped to MAX_STR)
    const LEV_KERNEL_SRC: &str = r#"
    __device__ __forceinline__ int max_i(int a, int b) { return a > b ? a : b; }
    __device__ __forceinline__ int min_i(int a, int b) { return a < b ? a : b; }

    extern "C" __global__ void lev_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int off_a = a_off[i]; int la = a_len[i]; if (la > (int)64) la = 64;
        const int off_b = b_off[i]; int lb = b_len[i]; if (lb > (int)64) lb = 64;
        const char* A = a_buf + off_a;
        const char* B = b_buf + off_b;
        int prev[65]; int curr[65];
        for (int j=0;j<=lb;++j) prev[j] = j;
        for (int ia=1; ia<=la; ++ia) {
            curr[0] = ia;
            char ca = A[ia-1];
            for (int jb=1; jb<=lb; ++jb) {
                int cost = (ca == B[jb-1]) ? 0 : 1;
                int del = prev[jb] + 1;
                int ins = curr[jb-1] + 1;
                int sub = prev[jb-1] + cost;
                int v = del < ins ? del : ins;
                curr[jb] = v < sub ? v : sub;
            }
            for (int jb=0; jb<=lb; ++jb) prev[jb] = curr[jb];
        }
        int dist = prev[lb];
        int ml = la > lb ? la : lb;
        float score = ml > 0 ? (1.0f - ((float)dist / (float)ml)) * 100.0f : 100.0f;
        out[i] = score;
    }

    __device__ float jaro_core(const char* A, int la, const char* B, int lb) {
        if (la == 0 && lb == 0) return 1.0f;
        int match_dist = max_i(0, max_i(la, lb) / 2 - 1);
        bool a_match[64]; bool b_match[64];
        for (int i=0;i<64;++i) { a_match[i]=false; b_match[i]=false; }
        int matches = 0;
        for (int i=0;i<la; ++i) {
            int start = max_i(0, i - match_dist);
            int end = min_i(i + match_dist + 1, lb);
            for (int j=start; j<end; ++j) {
                if (b_match[j]) continue;
                if (A[i] != B[j]) continue;
                a_match[i] = true; b_match[j] = true; ++matches; break;
            }
        }
        if (matches == 0) return 0.0f;
        int k = 0; int trans = 0;
        for (int i=0;i<la; ++i) {
            if (!a_match[i]) continue;
            while (k < lb && !b_match[k]) ++k;
            if (k < lb && A[i] != B[k]) ++trans;
            ++k;
        }
        float m = (float)matches;
        float j1 = m / la;
        float j2 = m / lb;
        float j3 = (m - trans/2.0f) / m;
        return (j1 + j2 + j3) / 3.0f;
    }

    extern "C" __global__ void jaro_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int la = a_len[i]; if (la > 64) la = 64;
        int lb = b_len[i]; if (lb > 64) lb = 64;
        const char* A = a_buf + a_off[i];
        const char* B = b_buf + b_off[i];
        float j = jaro_core(A, la, B, lb);
        out[i] = j * 100.0f;
    }

    extern "C" __global__ void jw_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int la = a_len[i]; if (la > 64) la = 64;
        int lb = b_len[i]; if (lb > 64) lb = 64;
        const char* A = a_buf + a_off[i];
        const char* B = b_buf + b_off[i];
        float j = jaro_core(A, la, B, lb);
        int l = 0; int maxp = 4;
        for (int k=0; k<min_i(min_i(la, lb), maxp); ++k) { if (A[k] == B[k]) ++l; else break; }
        float p = 0.1f;
        float jw = j + l * p * (1.0f - j);
        out[i] = jw * 100.0f;
    }

    extern "C" __global__ void max3_kernel(const float* a, const float* b, const float* c, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float m = a[i];
            if (b[i] > m) m = b[i];
            if (c[i] > m) m = c[i];
            out[i] = m;
        }
    }
    "#;
    // Per-person cache to avoid repeated normalization and metaphone encoding during GPU post-processing
    #[derive(Clone)]
    struct FuzzyCache {
        simple_full: String,
        simple_first: String,
        simple_mid: String,
        simple_last: String,
        phonetic_full: String,
        dmeta_code: String, // empty if encode failed/panicked/empty
    }

    fn build_cache_from_person(p: &Person) -> FuzzyCache {
        let simple_first = normalize_simple(p.first_name.as_deref().unwrap_or(""));
        let simple_mid = normalize_simple(p.middle_name.as_deref().unwrap_or(""));
        let simple_last = normalize_simple(p.last_name.as_deref().unwrap_or(""));
        let simple_full = normalize_simple(&format!("{} {} {}",
            p.first_name.as_deref().unwrap_or(""),
            p.middle_name.as_deref().unwrap_or(""),
            p.last_name.as_deref().unwrap_or("")));
        // match metaphone_pct() path: normalize_for_phonetic on the full name string
        let phonetic_full = normalize_for_phonetic(&simple_full);
        let dmeta_code = if phonetic_full.is_empty() {
            String::new()
        } else {
            // Protect against panics as in metaphone_pct
            match std::panic::catch_unwind(|| DoubleMetaphone::default().encode(&phonetic_full)) {
                Ok(code) => code.to_string(),
                Err(_) => String::new(),
            }
        };
        FuzzyCache { simple_full, simple_first, simple_mid, simple_last, phonetic_full, dmeta_code }
    }

    // Authoritative CPU classification using cached strings/codes to eliminate recomputation
    fn classify_pair_cached(c1: &FuzzyCache, c2: &FuzzyCache) -> Option<(f64, String)> {
        // Direct match
        if c1.simple_full == c2.simple_full { return Some((100.0, "DIRECT MATCH".to_string())); }
        // Metrics
        let lev = sim_levenshtein_pct(&c1.simple_full, &c2.simple_full);
        let jw = jaro_winkler(&c1.simple_full, &c2.simple_full) * 100.0;
        let mp = if !c1.dmeta_code.is_empty() && !c2.dmeta_code.is_empty() && c1.dmeta_code == c2.dmeta_code { 100.0 } else { 0.0 };
        // Case 1
        if lev >= 85.0 && jw >= 85.0 && mp == 100.0 {
            let avg = (lev + jw + mp) / 3.0; return Some((avg, "CASE 1".to_string()));
        }
        // Case 2 (+ Case 3 refinement)
        let mut pass = 0; if lev >= 85.0 { pass += 1; } if jw >= 85.0 { pass += 1; } if mp == 100.0 { pass += 1; }
        if pass >= 2 {
            let avg = (lev + jw + mp) / 3.0;
            if avg >= 88.0 {
                let ld_first = levenshtein(&c1.simple_first, &c2.simple_first) as usize;
                let ld_last  = levenshtein(&c1.simple_last,  &c2.simple_last)  as usize;
                let ld_mid   = levenshtein(&c1.simple_mid,   &c2.simple_mid)   as usize;
                if ld_first <= 2 && ld_last <= 2 && ld_mid <= 2 { return Some((avg, "CASE 3".to_string())); }
            }
            return Some((avg, "CASE 2".to_string()));
        }
        None
    }


    // --- GPU FNV-1a 64-bit hash kernel and hashing helpers (module scope) ---
    const FNV_KERNEL_SRC: &str = r#"
    extern "C" __global__ void fnv1a64_kernel(
        const char* buf, const int* off, const int* len,
        unsigned long long* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        unsigned long long hash = 0xcbf29ce484222325ULL;
        const unsigned long long prime = 0x100000001b3ULL;
        const char* s = buf + off[i];
        int L = len[i];
        #pragma unroll 1
        for (int j = 0; j < L; ++j) {
            hash ^= (unsigned long long)(unsigned char)s[j];
            hash *= prime;
        }
        out[i] = hash;
    }
    "#;

    pub struct GpuHashContext {
        ctx: std::sync::Arc<CudaContext>,
        module: std::sync::Arc<cudarc::driver::CudaModule>,
        func_hash: std::sync::Arc<cudarc::driver::CudaFunction>,
    }
    impl GpuHashContext {
        pub fn new() -> Result<Self> {
            let dev_id = 0usize;
            let ctx = CudaContext::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;

            // Query device details (name, compute capability, driver version)
            let (gpu_name, cc_major, cc_minor, drv_major, drv_minor) = unsafe {
                use std::ffi::CStr;
                use std::os::raw::{c_char, c_int};
                let mut cu_dev: cudarc::driver::sys::CUdevice = 0;
                let mut driver_ver: c_int = 0;
                // Best-effort queries; ignore non-zero return codes for logging-only paths
                let _ = cudarc::driver::sys::cuDeviceGet(&mut cu_dev as *mut _, dev_id as c_int);
                let _ = cudarc::driver::sys::cuDriverGetVersion(&mut driver_ver as *mut _);
                let mut maj: c_int = 0;
                let mut min: c_int = 0;
                let _ = cudarc::driver::sys::cuDeviceComputeCapability(&mut maj as *mut _, &mut min as *mut _, cu_dev);
                let mut name_buf: [c_char; 128] = [0; 128];
                let _ = cudarc::driver::sys::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as c_int, cu_dev);
                let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }.to_string_lossy().into_owned();
                let drv_major = driver_ver / 1000;
                let drv_minor = (driver_ver % 1000) / 10;
                (name, maj as i32, min as i32, drv_major as i32, drv_minor as i32)
            };

            // Memory snapshot and activation logs
            let (tot_mb, free_mb) = super::cuda_mem_info_mb(&ctx);
            let used_mb = tot_mb.saturating_sub(free_mb);
            log::info!(
                "[GPU] CUDA context initialized: {name} (dev {dev}, compute {cc_major}.{cc_minor}) | Driver {drv_major}.{drv_minor} | Mem: used={used}/{tot} MB, free={free} MB",
                name = gpu_name,
                dev = dev_id,
                cc_major = cc_major,
                cc_minor = cc_minor,
                drv_major = drv_major,
                drv_minor = drv_minor,
                used = used_mb,
                tot = tot_mb,
                free = free_mb
            );
            log::info!(
                "[GPU] GPU acceleration ACTIVE: {name} (dev {dev}, compute {cc_major}.{cc_minor}) | Memory: {used}/{tot} MB",
                name = gpu_name,
                dev = dev_id,
                cc_major = cc_major,
                cc_minor = cc_minor,
                used = used_mb,
                tot = tot_mb
            );

            let ptx = compile_ptx(FNV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
            let module = ctx.load_module(ptx).map_err(|e| anyhow!("Load PTX failed: {e}"))?;
            let func_hash = module.load_function("fnv1a64_kernel").map_err(|e| anyhow!("Get fnv1a64 func failed: {e}"))?;
            Ok(Self { ctx, module, func_hash: func_hash.into() })
        }
    }

    impl GpuHashContext {
        pub fn mem_info_mb(&self) -> (u64, u64) {
            super::cuda_mem_info_mb(&self.ctx)
        }
    }


    pub fn hash_fnv1a64_batch(hctx: &GpuHashContext, strings: &[String]) -> Result<Vec<u64>> {
        let n = strings.len();
        if n == 0 { return Ok(Vec::new()); }
        let mut offsets: Vec<i32> = Vec::with_capacity(n);
        let mut lengths: Vec<i32> = Vec::with_capacity(n);
        let mut flat: Vec<u8> = Vec::new();
        flat.reserve(strings.iter().map(|s| s.len()).sum());
        let mut cur = 0i32;
        for s in strings {
            offsets.push(cur);
            let bytes = s.as_bytes();
            lengths.push(bytes.len() as i32);
            flat.extend_from_slice(bytes);
            cur += bytes.len() as i32;
        }
        let stream = hctx.ctx.default_stream();
        let d_buf = stream.memcpy_stod(flat.as_slice())?;
        let d_off = stream.memcpy_stod(offsets.as_slice())?;
        let d_len = stream.memcpy_stod(lengths.as_slice())?;
        let mut d_out = stream.alloc_zeros::<u64>(n)?;
        let bs: u32 = 256;
        let grid: u32 = ((n as u32 + bs - 1) / bs).max(1);
        let cfg = LaunchConfig { grid_dim: (grid,1,1), block_dim: (bs,1,1), shared_mem_bytes: 0 };
        let n_i32 = n as i32;
        let (tot_mb, free_mb) = super::cuda_mem_info_mb(&hctx.ctx);
        log::debug!("[GPU] Launching fnv1a64_kernel for {} strings (grid={}, block={}) | mem: total={} MB free={} MB", n, grid, bs, tot_mb, free_mb);
        let mut b = stream.launch_builder(&hctx.func_hash);
        b.arg(&d_buf).arg(&d_off).arg(&d_len).arg(&mut d_out).arg(&n_i32);
        unsafe { b.launch(cfg)?; }
        stream.synchronize()?;
        let out: Vec<u64> = stream.memcpy_dtov(&d_out)?;
        Ok(out)
    }

    /// VRAM-aware tiled hashing for probe/build keys. Tiles the input to respect
    /// both current free VRAM and a user-provided budget. On CUDA OOM, halves the
    /// tile size and retries; finally falls back to CPU hashing for that tile.
    pub fn hash_fnv1a64_batch_tiled(
        hctx: &GpuHashContext,
        strings: &[String],
        budget_mb: u64,
        reserve_mb: u64,
    ) -> Result<Vec<u64>> {
        fn is_cuda_oom(e: &anyhow::Error) -> bool {
            let s = e.to_string().to_ascii_lowercase();
            s.contains("cuda_error_out_of_memory") || s.contains("out of memory") || s.contains("oom")
        }

        if strings.is_empty() { return Ok(Vec::new()); }
        let mut out: Vec<u64> = Vec::with_capacity(strings.len());
        let mut i = 0usize;
        let min_tile = 512usize;
        while i < strings.len() {
            // Determine per-tile target bytes from current free VRAM and user budget
            let (_tot_mb, free_mb) = super::cuda_mem_info_mb(&hctx.ctx);
            let target_mb = free_mb.min(budget_mb.max(64)).saturating_sub(reserve_mb.max(64));
            let target_bytes: usize = (target_mb as usize).saturating_mul(1024*1024).max(256*1024);

            // Greedy grow tile until budget
            let mut est_bytes = 0usize;
            let mut j = i;
            while j < strings.len() && est_bytes < target_bytes {
                // Rough per-key bytes: offsets(4)+len(4)+out(8)+string bytes
                est_bytes += 16 + strings[j].len();
                j += 1;
            }
            if j == i { j = (i + min_tile).min(strings.len()); }

            // Attempt with backoff on OOM
            let mut lo = i;
            let mut hi = j;
            let mut done = false;
            while !done {
                let tile = &strings[lo..hi];
                match hash_fnv1a64_batch(hctx, tile) {
                    Ok(mut v) => { out.extend(v.drain(..)); done = true; }
                    Err(e) if is_cuda_oom(&e) && tile.len() > min_tile => {
                        let new_hi = lo + (tile.len()/2).max(min_tile);
                        log::warn!("[GPU] OOM during probe hashing ({} keys); shrinking tile to {}", tile.len(), new_hi - lo);
                        hi = new_hi;
                    }
                    Err(e) => {
                        // CPU fallback for this tile
                        log::warn!("[GPU] Probe hashing fallback to CPU for {} keys: {}", tile.len(), e);
                        let mut v_cpu: Vec<u64> = Vec::with_capacity(tile.len());
                        for s in tile { v_cpu.push(super::fnv1a64_bytes(s.as_bytes())); }
                        out.extend(v_cpu);
                        done = true;
                    }
                }
            }
            i = j;
        }
        Ok(out)
    }
    /// GPU-accelerated hashing for in-memory deterministic algorithms (A1/A2).

    /// GPU hash pre-pass for Fuzzy direct phase (candidate filtering only).
    /// Returns, for each outer row i (people1), a Vec of indices j into people2 that share the blocking key.
    /// Key policy: always exact birthdate; optionally include last initial when partition strategy is 'last_initial'.
    pub fn fuzzy_direct_gpu_hash_prefilter_indices(
        people1: &[super::Person],
        people2: &[super::Person],
        part_strategy: &str,
    ) -> Result<Vec<Vec<usize>>> {
        let ctx = GpuHashContext::new()?;
        // Build keys for inner (people2)
        let mut keys2: Vec<String> = Vec::with_capacity(people2.len());
        let mut idx2: Vec<usize> = Vec::with_capacity(people2.len());
        for (j, p) in people2.iter().enumerate() {
            if let Some(d) = p.birthdate {
                let mut k = d.to_string();
                if part_strategy == "last_initial" {
                    let li = super::normalize_simple(p.last_name.as_deref().unwrap_or(""))
                        .chars().next().unwrap_or('\0').to_ascii_uppercase();
                    k.push('|'); k.push(li);
                }
                keys2.push(k); idx2.push(j);
            }
        }
        let (_tot_mb, free_mb) = ctx.mem_info_mb();
        let budget_mb = (free_mb / 2).max(64);
        let h2 = hash_fnv1a64_batch_tiled(&ctx, &keys2, budget_mb, 64)?;
        use std::collections::HashMap as Map;
        let mut map: Map<u64, Vec<usize>> = Map::with_capacity(h2.len());
        for (k, &h) in h2.iter().enumerate() { map.entry(h).or_default().push(idx2[k]); }
        // Probe from people1
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); people1.len()];
        let mut keys1: Vec<String> = Vec::with_capacity(people1.len());
        let mut idx1: Vec<usize> = Vec::with_capacity(people1.len());
        for (i, p) in people1.iter().enumerate() {
            if let Some(d) = p.birthdate {
                let mut k = d.to_string();
                if part_strategy == "last_initial" {
                    let li = super::normalize_simple(p.last_name.as_deref().unwrap_or(""))
                        .chars().next().unwrap_or('\0').to_ascii_uppercase();
                    k.push('|'); k.push(li);
                }
                keys1.push(k); idx1.push(i);
            }
        }
        let h1 = hash_fnv1a64_batch_tiled(&ctx, &keys1, budget_mb, 64)?;
        for (pos, &h) in h1.iter().enumerate() {
            let i = idx1[pos];
            if let Some(cands) = map.get(&h) { out[i] = cands.clone(); }
        }
        Ok(out)
    }

    /// Uses GPU only to compute FNV-1a 64-bit hashes; join/verification stays on CPU.
    pub fn det_match_gpu_hash_inmemory<F>(
        t1: &[Person], t2: &[Person], algo: MatchingAlgorithm, opts: &MatchOptions, on_progress: &F
    ) -> Result<Vec<MatchPair>>
    where F: Fn(ProgressUpdate) + Sync
    {
        use rayon::prelude::*;
        if matches!(algo, MatchingAlgorithm::Fuzzy) { return Err(anyhow!("Fuzzy not supported")); }
        let ctx = GpuHashContext::new()?;
        let (gt, gf) = ctx.mem_info_mb();
        // Normalize both tables (CPU)
        let n1: Vec<NormalizedPerson> = t1.par_iter().map(|p| normalize_person(p)).collect();
        let n2: Vec<NormalizedPerson> = t2.par_iter().map(|p| normalize_person(p)).collect();
        if n1.is_empty() || n2.is_empty() { return Ok(vec![]); }
        // Choose inner (smaller)
        let (inner_t, inner_n, outer_t, outer_n, inner_is_t2) = if n2.len() < n1.len() { (t2, n2.as_slice(), t1, n1.as_slice(), true) } else { (t1, n1.as_slice(), t2, n2.as_slice(), false) };
        // Build inner keys
        let mut key_idx: Vec<usize> = Vec::with_capacity(inner_n.len());
        let mut key_strs: Vec<String> = Vec::new();
        for (i, n) in inner_n.iter().enumerate() { if let Some(k) = super::key_for(algo, n) { key_idx.push(i); key_strs.push(k); } }
        // Hash inner keys
        let (_tmb, fmb) = ctx.mem_info_mb();
        let budget_mb = (fmb / 2).max(128);
        on_progress(ProgressUpdate { processed: 0, total: key_strs.len(), percent: 0.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "inmem_gpu_hash_build", batch_size_current: opts.progress.batch_size.map(|v| v as i64), gpu_total_mb: gt, gpu_free_mb: gf, gpu_active: true });
        let inner_hashes = hash_fnv1a64_batch_tiled(&ctx, &key_strs, budget_mb, 64)?;
        let mut index: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
        for (j, &h) in inner_hashes.iter().enumerate() { index.entry(h).or_default().push(key_idx[j]); }
        // Probe and verify
        let mut matches: Vec<MatchPair> = Vec::new();
        let mut processed: usize = 0; let total = outer_n.len(); let bs = opts.progress.batch_size.unwrap_or(100_000).max(10_000);
        let mut start = 0usize; while start < total {
            let end = (start + bs).min(total);
            let slice = &outer_n[start..end];
            let mut pkeys: Vec<String> = Vec::new(); let mut pidx: Vec<usize> = Vec::new();
            for (i, n) in slice.iter().enumerate() { if let Some(k) = super::key_for(algo, n) { pidx.push(i); pkeys.push(k); } }
            if !pkeys.is_empty() {
                let (gt2, gf2) = ctx.mem_info_mb(); let memx = memory_stats_mb();
                on_progress(ProgressUpdate { processed, total, percent: (processed as f32 / total.max(1) as f32) * 100.0, eta_secs: 0, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "inmem_gpu_probe", batch_size_current: Some(bs as i64), gpu_total_mb: gt2, gpu_free_mb: gf2, gpu_active: true });
                let phashes = hash_fnv1a64_batch_tiled(&ctx, &pkeys, budget_mb, 64)?;
                for (k, &h) in phashes.iter().enumerate() {
                    let np = &outer_n[pidx[k]];
                    if let Some(cands) = index.get(&h) {
                        for &ii in cands {
                            let q = &inner_t[ii]; let nq = &inner_n[ii];
                            let ok = match algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => super::matches_algo1(np, nq), MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => super::matches_algo2(np, nq), MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => false };
                            if ok {
                                let (a, b, na, nb) = if inner_is_t2 { (&outer_t[pidx[k] + (start - start)], q, np, nq) } else { (q, &outer_t[pidx[k] + (start - start)], nq, np) };
                                let pair = super::to_pair(a, b, algo, na, nb);
                                matches.push(pair);
                            }
                        }
                    }
                }
            }
            processed = end; start = end;
        }
        Ok(matches)
    }

    pub fn match_fuzzy_gpu<F>(t1: &[Person], t2: &[Person], opts: MatchOptions, on_progress: &F) -> Result<Vec<MatchPair>>
    where F: Fn(ProgressUpdate) + Sync
    {
        // 1) Normalize on CPU (reuse existing)
        let n1: Vec<NormalizedPerson> = t1.par_iter().map(normalize_person).collect();
        let n2: Vec<NormalizedPerson> = t2.par_iter().map(normalize_person).collect();
        if n1.is_empty() || n2.is_empty() { return Ok(vec![]); }

        // 2) Sophisticated multi-field blocking to reduce candidate pairs
        use std::collections::{HashMap, HashSet};
        #[derive(Hash, Eq, PartialEq)]

        struct BKey(u16, u8, u8, [u8;4]); // (birth year, first initial, last initial, last name soundex)
        let mut block: HashMap<BKey, Vec<usize>> = HashMap::new();
        for (j, p) in n2.iter().enumerate() {
            let (Some(d), Some(fn_str), Some(ln_str)) = (p.birthdate.as_ref(), p.first_name.as_deref(), p.last_name.as_deref()) else { continue; };
            let year = d.year() as u16;
            let fi = fn_str.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let li = ln_str.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let sx = super::soundex4_ascii(ln_str);
            block.entry(BKey(year, fi, li, sx)).or_default().push(j);
        }

        // 3) Prepare CUDA context & streams
        let dev_id = opts.gpu.and_then(|g| g.device_id).unwrap_or(0);
        // Build per-person caches once (used by GPU tiling and CPU post-processing)
        let cache1: Vec<FuzzyCache> = t1.par_iter().map(build_cache_from_person).collect();
        let cache2: Vec<FuzzyCache> = t2.par_iter().map(build_cache_from_person).collect();

        let ctx = CudaContext::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;
        let stream = ctx.default_stream();
        // Second stream for overlapping next-tile transfers/compute
        let stream2 = ctx.new_stream().map_err(|e| anyhow!("CUDA stream create failed: {e}"))?;
        let ptx = compile_ptx(LEV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| anyhow!("Load PTX failed: {e}"))?;
        let func = module.load_function("lev_kernel").map_err(|e| anyhow!("Get func failed: {e}"))?;
        let func_jaro = module.load_function("jaro_kernel").map_err(|e| anyhow!("Get jaro func failed: {e}"))?;
        let func_jw = module.load_function("jw_kernel").map_err(|e| anyhow!("Get jw func failed: {e}"))?;
        let func_max3 = module.load_function("max3_kernel").map_err(|e| anyhow!("Get max3 func failed: {e}"))?;

        // Report GPU init and memory info
        let (gpu_total_mb, mut gpu_free_mb) = cuda_mem_info_mb(&ctx);
        let mem0 = memory_stats_mb();
        on_progress(ProgressUpdate { processed: 0, total: n1.len(), percent: 0.0, eta_secs: 0, mem_used_mb: mem0.used_mb, mem_avail_mb: mem0.avail_mb, stage: "gpu_init", batch_size_current: None, gpu_total_mb, gpu_free_mb, gpu_active: true });


        // 4) Tile candidates to respect memory budget
        let mem_budget_mb = opts.gpu.map(|g| g.mem_budget_mb).unwrap_or(512);
        // Rough bytes per pair: two strings up to 64 bytes + offsets/len + output ~ 256 bytes
        let approx_bpp: usize = 256;


        let mut tile_max = ((mem_budget_mb as usize * 1024 * 1024) / approx_bpp).max(1024);
        tile_max = tile_max.min(200_000); // upper bound to keep launch sizes sane

        let mut results: Vec<MatchPair> = Vec::new();
        let total: usize = n1.len();

        for (i, p1) in n1.iter().enumerate() {
            let (Some(d), Some(fn_str), Some(ln_str)) = (p1.birthdate.as_ref(), p1.first_name.as_deref(), p1.last_name.as_deref()) else { continue; };
            let year = d.year() as u16;
            let fi = fn_str.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let li = ln_str.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let sx = super::soundex4_ascii(ln_str);
            let mut set: HashSet<usize> = HashSet::new();
            if let Some(v) = block.get(&BKey(year, fi, li, sx)) { set.extend(v.iter().copied()); }
            // fallback: drop first initial
            if set.is_empty() { if let Some(v) = block.get(&BKey(year, b'?', li, sx)) { set.extend(v.iter().copied()); } }
            // fallback: drop soundex precision (use only first 2 digits)
            if set.is_empty() { let mut sx2 = sx; sx2[2]=b'0'; sx2[3]=b'0'; if let Some(v) = block.get(&BKey(year, fi, li, sx2)) { set.extend(v.iter().copied()); } }
            if set.is_empty() { continue; }
            let cands_vec: Vec<usize> = set.into_iter().collect();
            // Build candidate pairs for this outer p1 in tiles of tile_max
            let mut start = 0;
                // Reuse host buffers across tiles to reduce allocations
                let mut a_offsets: Vec<i32> = Vec::new();
                let mut a_lengths: Vec<i32> = Vec::new();
                let mut b_offsets: Vec<i32> = Vec::new();
                let mut b_lengths: Vec<i32> = Vec::new();
                let mut a_bytes: Vec<u8> = Vec::new();
                let mut b_bytes: Vec<u8> = Vec::new();

            while start < cands_vec.len() {
                // Dynamic tile sizing BEFORE slicing: compute tile_len based on VRAM and mem_budget
                let remaining = cands_vec.len() - start;
                if remaining == 0 { break; }
                let (_tot_mb, free_mb_now) = cuda_mem_info_mb(&ctx);
                let budget_mb = opts.gpu.map(|g| g.mem_budget_mb).unwrap_or(512);
                let target_mb = free_mb_now.min(budget_mb).saturating_sub(64);
                let bytes_per_pair_est: u64 = 128; // conservative estimate
                let mut suggested_pairs = ((target_mb as u64 * 1024 * 1024) / bytes_per_pair_est) as usize;
                if suggested_pairs == 0 { suggested_pairs = 1; }
                let min_pairs = remaining.min(1024).max(1);
                let max_pairs = remaining.max(1);
                suggested_pairs = suggested_pairs.clamp(min_pairs, max_pairs);
                let tile_len = suggested_pairs.min(tile_max).min(remaining).max(1);
                let end = start + tile_len;
                let cur = &cands_vec[start..end];
                // Prepare/clear reusable buffers (reuse across tiles)
                a_offsets.clear(); a_lengths.clear(); b_offsets.clear(); b_lengths.clear();
                a_bytes.clear(); b_bytes.clear();
                a_offsets.reserve_exact(cur.len());
                a_lengths.reserve_exact(cur.len());
                b_offsets.reserve_exact(cur.len());
                b_lengths.reserve_exact(cur.len());
                a_bytes.reserve(cur.len() * 32);
                b_bytes.reserve(cur.len() * 32);
                for &j_idx in cur {
                    let s1 = &cache1[i].simple_full;
                    let s2 = &cache2[j_idx].simple_full;
                    let s1b = s1.as_bytes(); let s2b = s2.as_bytes();
                    let a_off = a_bytes.len() as i32; a_offsets.push(a_off);
                    let la = s1b.len().min(MAX_STR); a_lengths.push(la as i32);
                    a_bytes.extend_from_slice(&s1b[..la]);
                    let b_off = b_bytes.len() as i32; b_offsets.push(b_off);
                    let lb = s2b.len().min(MAX_STR); b_lengths.push(lb as i32);
                    b_bytes.extend_from_slice(&s2b[..lb]);
                }
                let n_pairs = cur.len();
                if n_pairs == 0 { start = end; continue; }

                // Tile length already chosen before slicing; n_pairs = cur.len()

                // Adaptive attempt with backoff on OOM
                let mut attempt_ok = false;
                loop {
                    // refresh GPU free mem for display
                    let (_tot, free_now) = cuda_mem_info_mb(&ctx); gpu_free_mb = free_now.max(gpu_free_mb);
                    let try_run: anyhow::Result<Vec<f32>> = (|| {
                        // Pick stream alternating to enable overlap
                        let use_stream2 = (start / n_pairs) % 2 == 1;
                        let s = if use_stream2 { &stream2 } else { &stream };
                        // Device buffers
                        let d_a = s.memcpy_stod(a_bytes.as_slice())?;
                        let d_a_off = s.memcpy_stod(a_offsets.as_slice())?;
                        let d_a_len = s.memcpy_stod(a_lengths.as_slice())?;
                        let d_b = s.memcpy_stod(b_bytes.as_slice())?;
                        let d_b_off = s.memcpy_stod(b_offsets.as_slice())?;
                        let d_b_len = s.memcpy_stod(b_lengths.as_slice())?;
                        let mut d_lev = s.alloc_zeros::<f32>(n_pairs)?;
                        let mut d_j = s.alloc_zeros::<f32>(n_pairs)?;
                        let mut d_w = s.alloc_zeros::<f32>(n_pairs)?;
                        let mut d_final = s.alloc_zeros::<f32>(n_pairs)?;
                        // Launch kernels (on chosen stream) with smaller block size to reduce per-SM resource pressure
                        let bs: u32 = 64; // was 256 by default; 64 lowers register pressure per block
                        let grid: u32 = ((n_pairs as u32 + bs - 1) / bs).max(1);
                        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (bs, 1, 1), shared_mem_bytes: 0 };
                        let n_i32 = n_pairs as i32;
                        // lev
                        let mut b1 = s.launch_builder(&func);
                        b1.arg(&d_a).arg(&d_a_off).arg(&d_a_len)
                          .arg(&d_b).arg(&d_b_off).arg(&d_b_len)
                          .arg(&mut d_lev).arg(&n_i32);
                        unsafe { b1.launch(cfg)?; }
                        // jaro
                        let mut b2 = s.launch_builder(&func_jaro);
                        b2.arg(&d_a).arg(&d_a_off).arg(&d_a_len)
                          .arg(&d_b).arg(&d_b_off).arg(&d_b_len)
                          .arg(&mut d_j).arg(&n_i32);
                        unsafe { b2.launch(cfg)?; }
                        // jw
                        let mut b3 = s.launch_builder(&func_jw);
                        b3.arg(&d_a).arg(&d_a_off).arg(&d_a_len)
                          .arg(&d_b).arg(&d_b_off).arg(&d_b_len)
                          .arg(&mut d_w).arg(&n_i32);
                        unsafe { b3.launch(cfg)?; }
                        // max3
                        let mut b4 = s.launch_builder(&func_max3);
                        b4.arg(&d_lev).arg(&d_j).arg(&d_w).arg(&mut d_final).arg(&n_i32);
                        unsafe { b4.launch(cfg)?; }
                        // Read back only final
                        let final_scores: Vec<f32> = s.memcpy_dtov(&d_final)?;
                        Ok(final_scores)
                    })();

                    match try_run {
                        Ok(final_scores) => {
                            // Use GPU scores only as a prefilter. Authoritative classification is CPU-equivalent via cached strings.
                            for (k, &j_idx) in cur.iter().enumerate() {
                                let gpu_pref = final_scores[k] as f64;
                                if gpu_pref < 85.0 { continue; }
                                // Restore algorithm requirement: birthdates must match before fuzzy classification
                                if !t1[i].birthdate.as_ref().zip(t2[j_idx].birthdate.as_ref()).map_or(false, |(a,b)| a==b) { continue; }
                                if let Some((score, label)) = classify_pair_cached(&cache1[i], &cache2[j_idx]) {
                                    results.push(MatchPair {
                                        person1: t1[i].clone(),
                                        person2: t2[j_idx].clone(),
                                        confidence: (score / 100.0) as f32, // normalize to 0..1 like CPU
                                        matched_fields: vec!["fuzzy".into(), label, "birthdate".into()],
                                        is_matched_infnbd: false,
                                        is_matched_infnmnbd: false,
                                    });
                                }
                            }
                            attempt_ok = true;
                            break;
                        }
                        Err(e) => {
                            // back off tile size and retry
                            log::warn!("GPU tile failed ({}); reducing tile_max from {}", e, tile_max);
                            if tile_max <= 512 { return Err(anyhow!("GPU processing failed even at minimal tile size: {}", e)); }
                            tile_max = (tile_max / 2).max(512);
                            // rebuild slice bounds at new tile size
                            let new_end = (start + tile_max).min(cands_vec.len());
                            if new_end == end { // cannot shrink further without progress
                                return Err(anyhow!("GPU processing failed and could not shrink tile further"));
                            }
                            // adjust cur to smaller slice
                            // Note: restart outer while-loop iteration with smaller end
                            // by resetting end and continue
                            // We break out to while start<... with the smaller tile_max
                            break;
                        }
                    }
                }
                if !attempt_ok { continue; }

                let total_pairs_est = total.max(1) * 1; // rough; we still show percent over outer loop
                let frac = (i as f32 / total as f32).clamp(0.0, 1.0);
                let mem = memory_stats_mb();
                on_progress(ProgressUpdate { processed: i+1, total, percent: frac*100.0, eta_secs: 0, mem_used_mb: mem.used_mb, mem_avail_mb: mem.avail_mb, stage: "gpu_kernel", batch_size_current: Some(cur.len() as i64), gpu_total_mb: gpu_total_mb, gpu_free_mb: gpu_free_mb, gpu_active: true });



                start = end;
            }
        }
        Ok(results)
    }

    pub fn match_fuzzy_no_mid_gpu<F>(t1: &[Person], t2: &[Person], opts: MatchOptions, on_progress: &F) -> Result<Vec<MatchPair>>
    where F: Fn(ProgressUpdate) + Sync
    {
        // Reuse full fuzzy GPU pipeline to generate candidate pairs, then reclassify
        // to Option 4 semantics (first+last only, birthdate equality required).
        let pairs = match_fuzzy_gpu(t1, t2, opts, on_progress)?;
        let mut out = Vec::with_capacity(pairs.len());
        for mut p in pairs.into_iter() {
            if let Some((score, label)) = super::compare_persons_no_mid(&p.person1, &p.person2) {
                p.confidence = (score / 100.0) as f32;
                p.matched_fields = vec!["fuzzy".into(), label, "birthdate".into()];
                p.is_matched_infnbd = false;
                p.is_matched_infnmnbd = false;
                out.push(p);
            }
        }
        Ok(out)
    }

}



fn to_original<'a>(np: &NormalizedPerson, originals: &'a [Person]) -> Person {
    originals.iter().find(|p| p.id == np.id).cloned().unwrap_or_else(|| Person {
        id: np.id,
        uuid: Some(np.uuid.clone()),
        first_name: np.first_name.clone(),
        middle_name: np.middle_name.clone(),
        last_name: np.last_name.clone(),
        birthdate: np.birthdate,
    })
}

#[cfg(test)]
mod tests {
    use super::*; use chrono::NaiveDate; use std::sync::{Arc, Mutex};
    fn p(id: i64, f: &str, m: Option<&str>, l: &str, d: (i32, u32, u32)) -> Person { Person { id, uuid: Some(format!("u{}", id)), first_name: Some(f.into()), middle_name: m.map(|s| s.to_string()), last_name: Some(l.into()), birthdate: NaiveDate::from_ymd_opt(d.0, d.1, d.2) } }
    #[test] fn algo1_basic() { let a = vec![p(1, "José", None, "García", (1990,1,1))]; let b = vec![p(2, "Jose", None, "Garcia", (1990,1,1))]; let r = match_all(&a,&b,MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |_| {}); assert_eq!(r.len(),1); assert!(r[0].is_matched_infnbd); }
    #[test] fn algo2_middle_required() { let a = vec![p(1, "Ann", Some("B"), "Lee", (1990,1,1))]; let b = vec![p(2, "Ann", None, "Lee", (1990,1,1))]; let r = match_all(&a,&b,MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |_| {}); assert_eq!(r.len(),0); }
    #[test] fn progress_updates() {
        let a = (0..10).map(|i| p(i, "A", None, "Z", (2000,1,1))).collect::<Vec<_>>();
        let b = (0..10).map(|i| p(100+i as i64, "A", None, "Z", (2000,1,1))).collect::<Vec<_>>();
        let updates: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::new(Mutex::new(vec![]));
        let cfg = ProgressConfig { update_every: 3, batch_size: Some(2), ..Default::default() };
        let u2 = updates.clone();
        let _ = match_all_progress(&a, &b, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, cfg, |u| { u2.lock().unwrap().push(u); });
        let v = updates.lock().unwrap();
        assert!(v.len() >= 3); // at least a few updates including final
    }

    #[test]
    fn hash_key_for_np_basic() {
        use chrono::NaiveDate;
        let a = Person { id: 1, uuid: None, first_name: Some("Ann".into()), middle_name: None, last_name: Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) };
        let n = normalize_person(&a);
        let h1 = hash_key_for_np(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &n);
        assert!(h1.is_some());
        let b = Person { id: 2, uuid: None, first_name: Some("Ann".into()), middle_name: None, last_name: Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) };
        let n2 = normalize_person(&b);
        let h2 = hash_key_for_np(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &n2);
        assert_eq!(h1, h2, "same normalized inputs must hash equal");
        // Missing field -> None
        let c = Person { id: 3, uuid: None, first_name: Some("Ann".into()), middle_name: None, last_name: Some("Lee".into()), birthdate: None };
        let n3 = normalize_person(&c);
        assert!(hash_key_for_np(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &n3).is_none());
    }

    fn hash_join_in_memory(algo: MatchingAlgorithm, t1: &[Person], t2: &[Person]) -> Vec<(i64,i64)> {
        use std::collections::HashMap;
        let mut map: HashMap<u64, Vec<Person>> = HashMap::new();
        for p in t1 {
            let n = normalize_person(p);
            if let Some(h) = hash_key_for_np(algo, &n) { map.entry(h).or_default().push(p.clone()); }
        }
        let mut out = Vec::new();
        for q in t2 {
            let nq = normalize_person(q);
            if let Some(hq) = hash_key_for_np(algo, &nq) {
                if let Some(cands) = map.get(&hq) {
                    for p in cands {
                        let np = normalize_person(p);
                        let ok = match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&np, &nq),
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(&np, &nq),
                            MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => false,
                        };
                        if ok { out.push((p.id, q.id)); }
                    }
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn hash_join_equivalence_algo1_and_2() {
        use chrono::NaiveDate;
        let t1 = vec![
            Person { id: 1, uuid: None, first_name: Some("José".into()), middle_name: None, last_name: Some("García".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) },
            Person { id: 2, uuid: None, first_name: Some("Ann".into()), middle_name: Some("B".into()), last_name: Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1985,5,5) },
        ];
        let t2 = vec![
            Person { id: 10, uuid: None, first_name: Some("Jose".into()), middle_name: None, last_name: Some("Garcia".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) },
            Person { id: 20, uuid: None, first_name: Some("Ann".into()), middle_name: Some("B".into()), last_name: Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1985,5,5) },
            Person { id: 21, uuid: None, first_name: Some("Ann".into()), middle_name: Some("C".into()), last_name: Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1985,5,5) },
        ];
        // Algorithm 1 (no middle name)
        let hj1 = hash_join_in_memory(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &t1, &t2);
        let mut direct1 = match_all(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |_|{}).into_iter().map(|m| (m.person1.id, m.person2.id)).collect::<Vec<_>>();
        direct1.sort();
        assert_eq!(hj1, direct1, "hash-join prefilter + exact verify must equal direct matches (algo1)");
        // Algorithm 2 (with middle name)
        let hj2 = hash_join_in_memory(MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, &t1, &t2);
        let mut direct2 = match_all(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |_|{}).into_iter().map(|m| (m.person1.id, m.person2.id)).collect::<Vec<_>>();
        direct2.sort();
        assert_eq!(hj2, direct2, "hash-join prefilter + exact verify must equal direct matches (algo2)");
    }


    }
    #[test]
    fn fuzzy_basic() {
        use chrono::NaiveDate;
        let a = vec![Person{ id:1, uuid:Some("u1".into()), first_name:Some("Jon".into()), middle_name:None, last_name:Some("Smith".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let b = vec![Person{ id:2, uuid:Some("u2".into()), first_name:Some("John".into()), middle_name:None, last_name:Some("Smith".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let r = match_all(&a,&b, MatchingAlgorithm::Fuzzy, |_| {});
        assert_eq!(r.len(), 1);
        assert!(r[0].confidence > 0.85);
    }
    #[test]
    fn metaphone_handles_unicode_without_panic() {
        let _ = metaphone_pct("JO\u{2229}N", "JOHN");
        let _ = metaphone_pct("Jos\u{00e9}", "Jose");
        let _ = metaphone_pct("M\u{00fc}ller", "Muller");
        let _ = metaphone_pct("\u{738b}\u{5c0f}\u{660e}", "Wang Xiaoming");
    }

    #[test]
    fn compute_stream_cfg_bounds_and_flush() {
        let c1 = compute_stream_cfg(512);
        assert!(c1.batch_size >= 5_000);
        assert!(c1.flush_every >= 1000);
        let c2 = compute_stream_cfg(32_768);
        assert!(c2.batch_size <= 100_000);
        assert_eq!(c2.flush_every, (c2.batch_size as usize / 10).max(1000));
    }

    #[test]
    fn optional_fields_algo1_requirements() {
        use chrono::NaiveDate;
        // Same names but missing birthdate -> no match
        let a = vec![Person{ id:1, uuid:Some("u1".into()), first_name:Some("Ann".into()), middle_name:None, last_name:Some("Lee".into()), birthdate: None }];
        let b = vec![Person{ id:2, uuid:Some("u2".into()), first_name:Some("Ann".into()), middle_name:None, last_name:Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let r = match_all(&a,&b, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |_| {});
        assert_eq!(r.len(), 0);
        // Missing first name -> no match
        let a2 = vec![Person{ id:3, uuid:Some("u3".into()), first_name:None, middle_name:None, last_name:Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let b2 = vec![Person{ id:4, uuid:Some("u4".into()), first_name:Some("Ann".into()), middle_name:None, last_name:Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let r2 = match_all(&a2,&b2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |_| {});
        assert_eq!(r2.len(), 0);
    }

    #[test]
    fn optional_fields_algo2_middle_none_allowed() {
        use chrono::NaiveDate;
        // Both middle None but names and birthdate equal -> match
        let a = vec![Person{ id:10, uuid:Some("u10".into()), first_name:Some("Ann".into()), middle_name:None, last_name:Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let b = vec![Person{ id:20, uuid:Some("u20".into()), first_name:Some("Ann".into()), middle_name:None, last_name:Some("Lee".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let r = match_all(&a,&b, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |_| {});
        assert_eq!(r.len(), 1);
        assert!(r[0].is_matched_infnmnbd);
    }

    #[test]
    fn fuzzy_requires_birthdate_and_some_name_content() {
        use chrono::NaiveDate;
        // Missing birthdate -> no match
        let a = vec![Person{ id:30, uuid:Some("u30".into()), first_name:Some("Jon".into()), middle_name:None, last_name:Some("Smith".into()), birthdate: None }];
        let b = vec![Person{ id:40, uuid:Some("u40".into()), first_name:Some("John".into()), middle_name:None, last_name:Some("Smith".into()), birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let r = match_all(&a,&b, MatchingAlgorithm::Fuzzy, |_| {});
        assert_eq!(r.len(), 0);
        // Empty names (all None) even with same birthdate -> no match
        let a2 = vec![Person{ id:31, uuid:Some("u31".into()), first_name:None, middle_name:None, last_name:None, birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let b2 = vec![Person{ id:41, uuid:Some("u41".into()), first_name:None, middle_name:None, last_name:None, birthdate: NaiveDate::from_ymd_opt(1990,1,1) }];
        let r2 = match_all(&a2,&b2, MatchingAlgorithm::Fuzzy, |_| {});
        assert_eq!(r2.len(), 0);
    }



    #[tokio::test]
    async fn prefetch_pipeline_sim() {
        use tokio::task::JoinHandle;
        let total_chunks = 5usize;
        let batch = 10i32;
        let mut next: Option<JoinHandle<Vec<i32>>> = None;
        let mut out: Vec<i32> = Vec::new();
        let mut offset = 0i32;
        while (offset as usize) < total_chunks * batch as usize {
            let cur: Vec<i32> = if let Some(h) = next.take() { h.await.unwrap() } else {
                let start = offset; let b = batch;
                tokio::spawn(async move { (start..start+b).collect::<Vec<_>>() }).await.unwrap()
            };
            out.extend(cur.iter());
            offset += batch;
            if (offset as usize) < total_chunks * batch as usize {
                let start = offset; let b = batch;
                next = Some(tokio::spawn(async move { (start..start+b).collect::<Vec<_>>() }));
            }
        }
        assert_eq!(out.len(), total_chunks * batch as usize);
        assert_eq!(out[0], 0);
        assert_eq!(*out.last().unwrap(), (total_chunks as i32 * batch) - 1);
    }

    #[test]
    fn checkpoint_roundtrip() {
        use crate::util::checkpoint::{save_checkpoint, load_checkpoint, remove_checkpoint, StreamCheckpoint};
        let path = format!("{}\\nmckpt_test_{}.ckpt", std::env::temp_dir().display(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
        let cp = StreamCheckpoint { db: "db".into(), table_inner: "t1".into(), table_outer: "t2".into(), algorithm: "Algo".into(), batch_size: 123, next_offset: 456, total_outer: 789, partition_idx: 0, partition_name: "p".into(), updated_utc: "now".into() };
        save_checkpoint(&path, &cp).unwrap();
        let loaded = load_checkpoint(&path).expect("should load");
        assert_eq!(loaded.table_inner, cp.table_inner);
        assert_eq!(loaded.next_offset, cp.next_offset);
        remove_checkpoint(&path);
        assert!(load_checkpoint(&path).is_none());
    }
// --- Streaming matching and export for large datasets ---
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use crate::db::{get_person_count, fetch_person_rows_chunk};
use sqlx::MySqlPool;


fn key_for(algo: MatchingAlgorithm, p: &NormalizedPerson) -> Option<String> {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (p.last_name.as_deref(), p.first_name.as_deref(), p.birthdate.as_ref()) else { return None; };
            if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln);
                let fn2 = normalize_simple(fnm);
                Some(format!("{}\x1F{}\x1F{}", ln2, fn2, d))
            } else {
                Some(format!("{}\x1F{}\x1F{}", ln, fnm, d))
            }
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (p.last_name.as_deref(), p.first_name.as_deref(), p.birthdate.as_ref()) else { return None; };
            let mid = p.middle_name.clone().unwrap_or_default();
            if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln);
                let fn2 = normalize_simple(fnm);
                let mid2 = normalize_simple(&mid);
                Some(format!("{}\x1F{}\x1F{}\x1F{}", ln2, fn2, mid2, d))
            } else {
                Some(format!("{}\x1F{}\x1F{}\x1F{}", ln, fnm, mid, d))
            }
        }
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => None,
    }
}

fn to_pair(orig1: &Person, orig2: &Person, algo: MatchingAlgorithm, _np1: &NormalizedPerson, _np2: &NormalizedPerson) -> MatchPair {
    let (im1, im2) = (
        matches!(algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd),
        matches!(algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd),
    );
    let matched_fields = match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => vec!["id","uuid","first_name","last_name","birthdate"].into_iter().map(String::from).collect(),
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => vec!["id","uuid","first_name","middle_name","last_name","birthdate"].into_iter().map(String::from).collect(),
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle => vec!["fuzzy".into(), "birthdate".into()],
        MatchingAlgorithm::HouseholdGpu => vec!["household".into()],
    };
    MatchPair { person1: orig1.clone(), person2: orig2.clone(), confidence: 1.0, matched_fields, is_matched_infnbd: im1, is_matched_infnmnbd: im2 }
}

async fn build_index(pool: &MySqlPool, table: &str, algo: MatchingAlgorithm, mut batch: i64) -> Result<HashMap<String, Vec<Person>>> {
    let total = get_person_count(pool, table).await?;
    if batch <= 0 { batch = 50_000; }
    let mut map: HashMap<String, Vec<Person>> = HashMap::with_capacity((total as f64 * 0.8) as usize);
    let mut offset: i64 = 0;
    while offset < total {
        let rows = fetch_person_rows_chunk(pool, table, offset, batch).await?;
        for p in rows.iter() {
            let n = normalize_person(p);
            if let Some(k) = key_for(algo, &n) {
                map.entry(k).or_default().push(p.clone());
            }
        }
        offset += batch;
    }
    Ok(map)
}

#[derive(Clone, Debug)]
pub struct StreamingConfig {
    pub batch_size: i64,
    pub memory_soft_min_mb: u64,
    // new fields for robustness on millions of records
    pub flush_every: usize,            // flush output every N matches
    pub resume: bool,                  // resume from checkpoint if exists
    pub retry_max: u32,                // DB retry attempts per chunk
    pub retry_backoff_ms: u64,         // base backoff between retries
    pub checkpoint_path: Option<String>,
    // GPU hash join acceleration (Algorithms 1/2 only)
    pub use_gpu_hash_join: bool,       // legacy switch enabling GPU hash-join path
    pub use_gpu_build_hash: bool,      // GPU for index build-side hashing
    pub use_gpu_probe_hash: bool,      // GPU for probe-side hashing
    pub gpu_probe_batch_mb: u64,       // advisory memory budget (MB) for probe GPU batches
    // New: GPU pre-pass for fuzzy direct phase
    pub use_gpu_fuzzy_direct_hash: bool,
    // New: apply fuzzy-style normalization to direct algorithms (A1/A2)
    pub direct_use_fuzzy_normalization: bool,
}
impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 50_000,
            memory_soft_min_mb: 800,
            flush_every: 10_000,
            resume: true,
            retry_max: 5,
            retry_backoff_ms: 500,
            checkpoint_path: None,
            use_gpu_hash_join: false,
            use_gpu_build_hash: false,
            use_gpu_probe_hash: false,
            gpu_probe_batch_mb: 256,
            use_gpu_fuzzy_direct_hash: false,
            direct_use_fuzzy_normalization: false,
        }
    }
}

/// Compute an adaptive StreamingConfig based on available memory (MB).
/// Conservative defaults with clamped batch sizes for stability across machines.
#[allow(dead_code)]
pub fn compute_stream_cfg(avail_mb: u64) -> StreamingConfig {
    let mut cfg = StreamingConfig::default();
    // Start with a conservative estimate: roughly a quarter of free RAM in rows, with clamps
    let mut b = (avail_mb as i64 - 1024).max(256) / 4;
    if b < 5_000 { b = 5_000; }
    if b > 100_000 { b = 100_000; }
    cfg.batch_size = b;
    cfg.flush_every = (cfg.batch_size as usize / 10).max(1000);
    cfg
}

// --- Hash helpers for GPU hash join (CPU fallback) ---
#[inline]
fn fnv1a64_bytes(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    let prime: u64 = 0x100000001b3;         // FNV prime
    for &b in data { hash ^= b as u64; hash = hash.wrapping_mul(prime); }
    hash
}

#[inline]
fn hash_key_for_np(algo: MatchingAlgorithm, p: &NormalizedPerson) -> Option<u64> {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (p.last_name.as_deref(), p.first_name.as_deref(), p.birthdate.as_ref()) else { return None; };
            let s = if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln); let fn2 = normalize_simple(fnm);
                format!("{}\x1F{}\x1F{}", ln2, fn2, d)
            } else { format!("{}\x1F{}\x1F{}", ln, fnm, d) };
            Some(fnv1a64_bytes(s.as_bytes()))
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (p.last_name.as_deref(), p.first_name.as_deref(), p.birthdate.as_ref()) else { return None; };
            let mid = p.middle_name.clone().unwrap_or_default();
            let s = if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln); let fn2 = normalize_simple(fnm); let mid2 = normalize_simple(&mid);
                format!("{}\x1F{}\x1F{}\x1F{}", ln2, fn2, mid2, d)
            } else { format!("{}\x1F{}\x1F{}\x1F{}", ln, fnm, mid, d) };
            Some(fnv1a64_bytes(s.as_bytes()))
        }
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => None,
    }
}

#[inline]
fn concat_key_for_np(algo: MatchingAlgorithm, p: &NormalizedPerson) -> Option<String> {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (p.last_name.as_deref(), p.first_name.as_deref(), p.birthdate.as_ref()) else { return None; };
            Some(format!("{}\x1F{}\x1F{}", ln, fnm, d))
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (p.last_name.as_deref(), p.first_name.as_deref(), p.birthdate.as_ref()) else { return None; };
            let mid = p.middle_name.clone().unwrap_or_default();
            Some(format!("{}\x1F{}\x1F{}\x1F{}", ln, fnm, mid, d))
        }
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => None,
    }
}

/// GPU-accelerated (hash-join) streaming path for Algorithms 1 & 2.
/// Falls back to CPU hashing if GPU is unavailable or feature is disabled.
pub async fn stream_match_gpu_hash_join<F>(
    pool: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    mut on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where F: FnMut(&MatchPair) -> Result<()>
{
    use crate::util::checkpoint::{load_checkpoint, save_checkpoint, StreamCheckpoint};
    if matches!(algo, MatchingAlgorithm::Fuzzy) {

        anyhow::bail!("GPU hash join applies only to Algorithms 1/2 (deterministic)");
    }
    // Decide inner/outer by row count
    let c1 = get_person_count(pool, table1).await?;
    let c2 = get_person_count(pool, table2).await?;
    // Try GPU hash computation context (optional)
    #[cfg(feature = "gpu")]
    let gpu_hash_ctx: Option<gpu::GpuHashContext> = if cfg.use_gpu_hash_join || cfg.use_gpu_build_hash || cfg.use_gpu_probe_hash {
        match gpu::GpuHashContext::new() {
            Ok(ctx) => { log::info!("[GPU] Hash context ready"); Some(ctx) },
            Err(e) => {
                if cfg.use_gpu_probe_hash {
                    // Explicit probe-on-GPU requested: do not silently fall back
                    return Err(anyhow!("GPU probe hashing requested but CUDA unavailable: {}", e));
                } else {
                    log::warn!("[GPU] Hash context init failed: {}. Falling back to CPU hashing for build.", e);
                    None
                }
            }
        }
    } else { None };
    #[cfg(not(feature = "gpu"))]
    let _gpu_hash_ctx: Option<()> = None;

    let (inner_table, outer_table, total_outer) = if c2 <= c1 { (table2, table1, c1) } else { (table1, table2, c2) };

    // Resume support
    let mut offset: i64 = 0;
    if cfg.resume { if let Some(path) = cfg.checkpoint_path.as_ref() { if let Some(cp) = load_checkpoint(path) { if cp.table_inner == inner_table && cp.table_outer == outer_table && cp.batch_size == cfg.batch_size { offset = cp.next_offset; } } } }

    // Build inner hash index (CPU hash; can be replaced with GPU hash in future)
    let mut index: std::collections::HashMap<u64, Vec<Person>> = std::collections::HashMap::new();
    let mut inner_off: i64 = 0;
    let batch = cfg.batch_size.max(10_000);
    on_progress(ProgressUpdate{ processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "indexing_hash", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
    let mut gpu_logged_once = false;
    loop {
        let rows = fetch_person_rows_chunk(pool, inner_table, inner_off, batch).await?;
        if rows.is_empty() { break; }
        // Prepare normalized key strings for hashing (only valid keys)
        let mut key_strs: Vec<String> = Vec::new();
        let mut key_idx: Vec<usize> = Vec::new();
        let mut norm_cache: Vec<Option<NormalizedPerson>> = Vec::with_capacity(rows.len());
        for (i, p) in rows.iter().enumerate() {
            let n = normalize_person(p);
            let k = concat_key_for_np(algo, &n);
            norm_cache.push(Some(n));
            if let Some(s) = k { key_idx.push(i); key_strs.push(s); }
        }
        let mut hashed: Option<Vec<u64>> = None;
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_build_hash {
            if let Some(ctx) = gpu_hash_ctx.as_ref() {
                if !key_strs.is_empty() {
                    // Progress: starting GPU hash for this batch (build side)
                    let (gt, gf) = ctx.mem_info_mb();
                    let memx = memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed: inner_off as usize,
                        total: total_outer as usize,
                        percent: 0.0,
                        eta_secs: 0,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "gpu_hash",
                        batch_size_current: Some(batch),
                        gpu_total_mb: gt,
                        gpu_free_mb: gf,
                        gpu_active: true,
                    });
                    match gpu::hash_fnv1a64_batch(ctx, &key_strs) {
                        Ok(v) => {
                            if !gpu_logged_once { log::info!("[GPU] Using GPU hashing for inner index (first batch: {} keys)", v.len()); gpu_logged_once = true; }
                            // Progress: after kernel execution
                            let (gt2, gf2) = ctx.mem_info_mb();
                            let mem2 = memory_stats_mb();
                            on_progress(ProgressUpdate {
                                processed: inner_off as usize,
                                total: total_outer as usize,
                                percent: 0.0,
                                eta_secs: 0,
                                mem_used_mb: mem2.used_mb,
                                mem_avail_mb: mem2.avail_mb,
                                stage: "gpu_hash_done",
                                batch_size_current: Some(batch),
                                gpu_total_mb: gt2,
                                gpu_free_mb: gf2,
                                gpu_active: true,
                            });
                            hashed = Some(v);
                        }
                        Err(e) => { log::warn!("GPU hash failed, falling back to CPU: {}", e); }
                    }
                }
            }
        }
        if let Some(hs) = hashed.as_ref() {
            for (j, &h) in hs.iter().enumerate() {
                let i = key_idx[j];
                index.entry(h).or_default().push(rows[i].clone());
            }
        } else {
            // CPU fallback hashing
            for (i, p) in rows.iter().enumerate() {
                if let Some(ref n) = norm_cache[i] {
                    if let Some(h) = hash_key_for_np(algo, n) {
                        index.entry(h).or_default().push(p.clone());
                    }
                }
            }
        }
        inner_off += batch;
        if (inner_off as usize) >= (if inner_table==table2 { c2 } else { c1 }) as usize { break; }
    }

    // Stream outer table and probe
    let start = Instant::now();
    let mut written = 0usize;
    while offset < total_outer {
        if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
        let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
        if rows.is_empty() { break; }
        // Progress update
        let elapsed = start.elapsed();
        let processed = (offset as usize).min(total_outer as usize);
        let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
        let eta_secs = if frac>0.0 { (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs, mem_used_mb: mem.used_mb, mem_avail_mb: mem.avail_mb, stage: "probing_hash", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

        // Prepare probe batch normalization and keys
        let mut probe_norms: Vec<NormalizedPerson> = Vec::with_capacity(rows.len());
        let mut probe_keys: Vec<String> = Vec::new();
        let mut probe_idx: Vec<usize> = Vec::new();
        for (i, p) in rows.iter().enumerate() {
            let n = normalize_person(p);
            if concat_key_for_np(algo, &n).is_some() {
                if let Some(k) = concat_key_for_np(algo, &n) { probe_keys.push(k); probe_idx.push(i); }
            }
            probe_norms.push(n);
        }
        // Compute probe hashes (GPU if enabled)
        let mut probe_hashes_opt: Option<Vec<u64>> = None;
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_probe_hash {
            if let Some(ctx) = gpu_hash_ctx.as_ref() {
                if !probe_keys.is_empty() {
                    let (gt, gf) = ctx.mem_info_mb();
                    let memx = memory_stats_mb();
                    log::info!("[GPU] Using GPU hashing for probe (batch: {} keys)", probe_keys.len());
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "gpu_probe_hash", batch_size_current: Some(batch), gpu_total_mb: gt, gpu_free_mb: gf, gpu_active: true });
                    match gpu::hash_fnv1a64_batch_tiled(ctx, &probe_keys, cfg.gpu_probe_batch_mb, 64) {
                        Ok(hs) => {
                            let (gt2, gf2) = ctx.mem_info_mb();
                            let mem2 = memory_stats_mb();
                            on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs, mem_used_mb: mem2.used_mb, mem_avail_mb: mem2.avail_mb, stage: "gpu_probe_hash_done", batch_size_current: Some(batch), gpu_total_mb: gt2, gpu_free_mb: gf2, gpu_active: true });
                            probe_hashes_opt = Some(hs);
                        }
                        Err(e) => { return Err(anyhow!("GPU probe hash failed: {}", e)); }
                    }
                }
            } else {
                return Err(anyhow!("GPU probe hashing requested but no CUDA context available"));
            }
        }
        if let Some(probe_hashes) = probe_hashes_opt.as_ref() {
            for (j, &h) in probe_hashes.iter().enumerate() {
                let i = probe_idx[j];
                let p = &rows[i];
                let n = &probe_norms[i];
                if let Some(cands) = index.get(&h) {
                    for q in cands {
                        let n2 = normalize_person(q);
                        let ok = match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(n, &n2),
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(n, &n2),
                            MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => false,
                        };
                        if ok {
                            let pair = if inner_table == table2 { to_pair(p, q, algo, n, &n2) } else { to_pair(q, p, algo, &n2, n) };
                            on_match(&pair)?; written += 1;
                        }
                    }
                }
            }
        } else {
            // CPU hashing for probe
            for (i, p) in rows.iter().enumerate() {
                let n = &probe_norms[i];
                if let Some(h) = hash_key_for_np(algo, n) {
                    if let Some(cands) = index.get(&h) {
                        for q in cands {
                            let n2 = normalize_person(q);
                            let ok = match algo {
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(n, &n2),
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(n, &n2),
                                MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => false,
                            };
                            if ok {
                                let pair = if inner_table == table2 { to_pair(p, q, algo, n, &n2) } else { to_pair(q, p, algo, &n2, n) };
                                on_match(&pair)?; written += 1;
                            }
                        }
                    }
                }
            }
        }

        offset += batch;
        if cfg.resume { if let Some(path) = cfg.checkpoint_path.as_ref() {
            let cp = StreamCheckpoint { db: String::new(), table_inner: inner_table.into(), table_outer: outer_table.into(), algorithm: format!("{:?}", algo), batch_size: batch, next_offset: offset, total_outer, partition_idx: 0, partition_name: String::new(), updated_utc: chrono::Utc::now().to_rfc3339() };
            let _ = save_checkpoint(path, &cp);
        }}
    }

    Ok(written)
}


#[derive(Clone)]
pub struct StreamControl { pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>, pub pause: std::sync::Arc<std::sync::atomic::AtomicBool> }

#[allow(dead_code)]
pub async fn stream_match_csv<F>(pool: &MySqlPool, table1: &str, table2: &str, algo: MatchingAlgorithm, mut on_match: F, cfg: StreamingConfig, on_progress: impl Fn(ProgressUpdate) + Sync, ctrl: Option<StreamControl>) -> Result<usize>
where F: FnMut(&MatchPair) -> Result<()>
{
    use crate::util::checkpoint::{load_checkpoint, save_checkpoint, remove_checkpoint, StreamCheckpoint};
    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
        anyhow::bail!("Fuzzy algorithms are supported only in in-memory or partitioned mode (CSV). Use algorithm=3/4 with CSV in-memory or partitioned streaming.");
    }
    #[cfg(not(feature = "gpu"))]
    { if cfg.use_gpu_hash_join { log::warn!("GPU hash-join requested but GPU feature not compiled; proceeding with CPU"); } }
    // Optional accelerated path: GPU hash-join (with CPU hashing fallback inside)
    if cfg.use_gpu_hash_join && !matches!(algo, MatchingAlgorithm::Fuzzy) {
        log::info!("GPU hash-join path selected (algo={:?}, tables: {} vs {})", algo, table1, table2);
        return stream_match_gpu_hash_join(pool, table1, table2, algo, on_match, cfg, on_progress, ctrl).await;
    }

    let c1 = get_person_count(pool, table1).await?;
    let c2 = get_person_count(pool, table2).await?;
    // index smaller table
    let (inner_table, outer_table, total) = if c2 <= c1 { (table2, table1, c1) } else { (table1, table2, c2) };
    let mut batch = cfg.batch_size.max(10_000);
    let start = Instant::now();

    // Progress: indexing start
    let mems = memory_stats_mb();
    on_progress(ProgressUpdate { processed: 0, total: total as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems.used_mb, mem_avail_mb: mems.avail_mb, stage: "indexing", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
    let index = build_index(pool, inner_table, algo, batch).await?;
    let mems2 = memory_stats_mb();
    on_progress(ProgressUpdate { processed: 0, total: total as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems2.used_mb, mem_avail_mb: mems2.avail_mb, stage: "indexing_done", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

    // Resume support: detect checkpoint
    let mut offset: i64 = 0;
    if cfg.resume {
        if let Some(p) = cfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(p) {
                if cp.db == "" || (cp.table_inner == inner_table && cp.table_outer == outer_table && cp.algorithm == format!("{:?}", algo)) {
                    offset = cp.next_offset.min(total);
                    batch = cp.batch_size.max(10_000);
                }
            }
        }
    }

    let mut written = 0usize; let mut processed = 0usize;
    let mut last_chunk_start = Instant::now();

    let mut next_rows_task: Option<tokio::task::JoinHandle<anyhow::Result<Vec<Person>>>> = None;

    while offset < total {
        if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(100)).await; } }
        // adaptive batch: memory based decrease, throughput-based increase
        let mem = memory_stats_mb();
        if mem.avail_mb < cfg.memory_soft_min_mb && batch > 10_000 { batch = (batch / 2).max(10_000); }

        // obtain rows: use prefetched task if available, else fetch with retry
        let rows: Vec<Person> = if let Some(handle) = next_rows_task.take() {
            // await the prefetched result
            match handle.await {
                Ok(res) => res?,
                Err(_join_err) => {
                    // fall back to direct fetch with retry if join failed
                    let mut tries = 0u32;
                    loop {
                        match fetch_person_rows_chunk(pool, outer_table, offset, batch).await {
                            Ok(v) => break v,
                            Err(e) => {
                                tries += 1;
                                if tries > cfg.retry_max { return Err(e); }
                                let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5)-1));
                                tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                            }
                        }
                    }
                }
            }
        } else {
            let mut tries = 0u32;
            loop {
                match fetch_person_rows_chunk(pool, outer_table, offset, batch).await {
                    Ok(v) => break v,
                    Err(e) => {
                        tries += 1;
                        if tries > cfg.retry_max { return Err(e); }
                        let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5)-1));
                        tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                    }
                }
            }
        };

        for p in rows.iter() {
            if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
            let n = normalize_person(p);
            if let Some(k) = key_for(algo, &n) {
                if let Some(cands) = index.get(&k) {
                    for q in cands {
                        let n2 = normalize_person(q);
                        let ok = match algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&n, &n2), MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(&n, &n2), MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => false };
                        if ok { let pair = if inner_table == table2 { to_pair(p, q, algo, &n, &n2) } else { to_pair(q, p, algo, &n2, &n) }; on_match(&pair)?; written += 1; }
                    }
                }
            }
        }

        offset += batch; processed = (processed + rows.len()).min(total as usize);

        // save checkpoint
        if let Some(p) = cfg.checkpoint_path.as_ref() {
            let _ = save_checkpoint(p, &StreamCheckpoint {
                db: String::new(),
                table_inner: inner_table.to_string(),
                table_outer: outer_table.to_string(),
                algorithm: format!("{:?}", algo),
                batch_size: batch,
                next_offset: offset,
                total_outer: total,
                partition_idx: 0,
                partition_name: "all".into(),
                updated_utc: chrono::Utc::now().to_rfc3339(),
            });
        }

        // progress update
        let frac = (processed as f32 / total as f32).clamp(0.0, 1.0);
        let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
        let memx = memory_stats_mb();
        on_progress(ProgressUpdate { processed, total: total as usize, percent: frac * 100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

        // adaptive increase if fast

        // prefetch next chunk while we process current one
        if offset < total {
            let pool_cloned = pool.clone();
            let table = outer_table.to_string();
            let next_off = offset;
            let next_batch = batch;
            let retry_max = cfg.retry_max;
            let backoff_ms = cfg.retry_backoff_ms;
            next_rows_task = Some(tokio::spawn(async move {
                let mut tries = 0u32;
                loop {
                    match fetch_person_rows_chunk(&pool_cloned, &table, next_off, next_batch).await {
                        Ok(v) => break Ok(v),
                        Err(e) => {
                            tries += 1;
                            if tries > retry_max { break Err(e); }
                            let backoff = backoff_ms * (1u64 << (tries.min(5)-1));
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }));
        }

        let dur = last_chunk_start.elapsed();
        if dur.as_millis() > 0 { // if chunk was quick and memory is plentiful, increase
            if memx.avail_mb > cfg.memory_soft_min_mb * 2 && dur < std::time::Duration::from_secs(1) {
                let new_batch = (batch as f64 * 1.5) as i64;
                batch = new_batch.min(200_000).max(10_000);
            }
        }
        last_chunk_start = Instant::now();
        // allow runtime to schedule
        tokio::task::yield_now().await;
    }

    if let Some(p) = cfg.checkpoint_path.as_ref() { remove_checkpoint(p); }
    Ok(written)
}

// New: dual-pool variant to support cross-database streaming
pub async fn stream_match_csv_dual<F>(
    pool1: &MySqlPool,
    pool2: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    mut on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where F: FnMut(&MatchPair) -> Result<()>
{
    use crate::util::checkpoint::{load_checkpoint, save_checkpoint, remove_checkpoint, StreamCheckpoint};
    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) { anyhow::bail!("Fuzzy algorithms are supported only in in-memory or partitioned mode (CSV). Use algorithm=3/4 with CSV in-memory or partitioned streaming."); }
    let c1 = get_person_count(pool1, table1).await?;
    let c2 = get_person_count(pool2, table2).await?;
    // Decide inner/outer and corresponding pools
    let inner_is_t2 = c2 <= c1;
    let (inner_table, inner_pool, outer_table, outer_pool, total) = if inner_is_t2 {
        (table2, pool2, table1, pool1, c1)
    } else {
        (table1, pool1, table2, pool2, c2)
    };
    let mut batch = cfg.batch_size.max(10_000);
    let start = Instant::now();

    // Progress: indexing start
    let mems = memory_stats_mb();
    on_progress(ProgressUpdate { processed: 0, total: total as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems.used_mb, mem_avail_mb: mems.avail_mb, stage: "indexing", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
    let index = build_index(inner_pool, inner_table, algo, batch).await?;
    let mems2 = memory_stats_mb();
    on_progress(ProgressUpdate { processed: 0, total: total as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems2.used_mb, mem_avail_mb: mems2.avail_mb, stage: "indexing_done", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

    // Resume support: detect checkpoint
    let mut offset: i64 = 0;
    if cfg.resume {
        if let Some(p) = cfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(p) {
                if cp.db == "" || (cp.table_inner == inner_table && cp.table_outer == outer_table && cp.algorithm == format!("{:?}", algo)) {
                    offset = cp.next_offset.min(total);
                    batch = cp.batch_size.max(10_000);
                }
            }
        }
    }

    let mut written = 0usize; let mut processed = 0usize;
    let mut last_chunk_start = Instant::now();

    let mut next_rows_task_dual: Option<tokio::task::JoinHandle<anyhow::Result<Vec<Person>>>> = None;

    while offset < total {
        if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(100)).await; } }
        // adaptive batch: memory based decrease, throughput-based increase
        let mem = memory_stats_mb();
        if mem.avail_mb < cfg.memory_soft_min_mb && batch > 10_000 { batch = (batch / 2).max(10_000); }

        // obtain rows: use prefetched task if available, else fetch with retry from OUTER pool
        let rows: Vec<Person> = if let Some(handle) = next_rows_task_dual.take() {
            match handle.await {
                Ok(res) => res?,
                Err(_join_err) => {
                    let mut tries = 0u32;
                    loop {
                        match fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await {
                            Ok(v) => break v,
                            Err(e) => {
                                tries += 1;
                                if tries > cfg.retry_max { return Err(e); }
                                let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5)-1));
                                tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                            }
                        }
                    }
                }
            }
        } else {
            let mut tries = 0u32;
            loop {
                match fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await {
                    Ok(v) => break v,
                    Err(e) => {
                        tries += 1;
                        if tries > cfg.retry_max { return Err(e); }
                        let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5)-1));
                        tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                    }


                }
            }
        };

        for p in rows.iter() {
            if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
            let n = normalize_person(p);
            if let Some(k) = key_for(algo, &n) {
                if let Some(cands) = index.get(&k) {
                for q in cands {
                    let n2 = normalize_person(q);
                    let ok = match algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&n, &n2), MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(&n, &n2), MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu => false };
                    if ok {
                        let pair = if inner_is_t2 { to_pair(p, q, algo, &n, &n2) } else { to_pair(q, p, algo, &n2, &n) };
                        on_match(&pair)?; written += 1;
                    }
                }
            }
            }

        }

        offset += batch; processed = (processed + rows.len()).min(total as usize);

        // save checkpoint
        if let Some(p) = cfg.checkpoint_path.as_ref() {
            let _ = save_checkpoint(p, &StreamCheckpoint {
                db: String::new(),
                table_inner: inner_table.to_string(),
                table_outer: outer_table.to_string(),
                algorithm: format!("{:?}", algo),
                batch_size: batch,
                next_offset: offset,
                total_outer: total,
                partition_idx: 0,
                partition_name: "all".into(),
                updated_utc: chrono::Utc::now().to_rfc3339(),
            });
        }

        // progress update
        let frac = (processed as f32 / total as f32).clamp(0.0, 1.0);
        let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };

#[cfg(test)]
mod tests {
    use super::*;

    fn np(first: &str, mid: Option<&str>, last: &str, date: &str) -> NormalizedPerson {
        NormalizedPerson {
            id: 0,
            uuid: String::new(),
            first_name: Some(first.to_string()),
            middle_name: mid.map(|s| s.to_string()),
            last_name: Some(last.to_string()),
            birthdate: Some(chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap()),
        }
    }

    #[test]
    fn direct_normalization_alignment_algo1() {
        let a = np("Ann.", None, "Smith-Jones", "2000-01-02");
        let b = np("ann", None, "smith jones", "2000-01-02");
        // Default strict behavior: should not match
        set_direct_normalization_fuzzy(false);
        assert!(!matches_algo1(&a, &b));
        // Fuzzy-style normalization enabled: should match
        set_direct_normalization_fuzzy(true);
        assert!(matches_algo1(&a, &b));
        // Reset
        set_direct_normalization_fuzzy(false);
    }

    #[test]
    fn direct_normalization_alignment_algo2_middle_optional() {
        let a = np("Jo-hn", Some("P."), "Doe", "1999-12-31");
        let b = np("john", None, "doe", "1999-12-31");
        set_direct_normalization_fuzzy(true);
        // Middle name missing on one side is acceptable
        assert!(matches_algo2(&a, &b));
        set_direct_normalization_fuzzy(false);
    }
}

        let memx = memory_stats_mb();
        on_progress(ProgressUpdate { processed, total: total as usize, percent: frac * 100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });


        // prefetch next chunk (dual) while processing current one
        if offset < total {
            let pool_cloned = outer_pool.clone();
            let table = outer_table.to_string();
            let next_off = offset;
            let next_batch = batch;
            let retry_max = cfg.retry_max;
            let backoff_ms = cfg.retry_backoff_ms;
            next_rows_task_dual = Some(tokio::spawn(async move {
                let mut tries = 0u32;
                loop {
                    match fetch_person_rows_chunk(&pool_cloned, &table, next_off, next_batch).await {
                        Ok(v) => break Ok(v),
                        Err(e) => {
                            tries += 1;
                            if tries > retry_max { break Err(e); }
                            let backoff = backoff_ms * (1u64 << (tries.min(5)-1));
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }));
        }

        // adaptive increase if fast
        let dur = last_chunk_start.elapsed();
        if dur.as_millis() > 0 { if memx.avail_mb > cfg.memory_soft_min_mb * 2 && dur < std::time::Duration::from_secs(1) { let new_batch = (batch as f64 * 1.5) as i64; batch = new_batch.min(200_000).max(10_000); } }
        last_chunk_start = Instant::now();
        tokio::task::yield_now().await;
    }

    if let Some(p) = cfg.checkpoint_path.as_ref() { remove_checkpoint(p); }
    Ok(written)
}

// Backward-compatible single-pool API can continue to be used alongside the dual-pool API



// --- Partitioned streaming (multi-pass) ---
use crate::util::partition::{PartitionStrategy, DefaultPartition};
use crate::db::schema::{get_person_count_where, get_person_rows_where, fetch_person_rows_chunk_where};
use crate::models::ColumnMapping;

#[derive(Clone, Debug)]
pub struct PartitioningConfig {
    #[allow(dead_code)]
    pub enabled: bool,
    pub strategy: String, // e.g., "last_initial" | "birthyear5"
}
impl Default for PartitioningConfig { fn default() -> Self { Self { enabled: false, strategy: "last_initial".into() } } }

pub async fn stream_match_csv_partitioned<F>(
    pool: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    mut on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
    mapping1: Option<&ColumnMapping>,
    mapping2: Option<&ColumnMapping>,
    part_cfg: PartitioningConfig,
) -> Result<usize>
where F: FnMut(&MatchPair) -> Result<()>
{
    use crate::util::checkpoint::{load_checkpoint, save_checkpoint, remove_checkpoint, StreamCheckpoint};
    let strat: Box<dyn PartitionStrategy + Send + Sync> = match part_cfg.strategy.as_str() {
        "birthyear5" => DefaultPartition::BirthYear5.build(),
        _ => DefaultPartition::LastInitial.build(),
    };
    let parts1 = strat.partitions(mapping1);
    let parts2 = strat.partitions(mapping2);
    if parts1.len() != parts2.len() { anyhow::bail!("Partition strategy produced mismatched partition counts for the two tables"); }

    // resume state
    let mut start_part: usize = 0;
    let mut offset: i64 = 0;
    let mut batch = cfg.batch_size.max(10_000);
    if cfg.resume {
        if let Some(pth) = cfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(pth) {
                start_part = (cp.partition_idx as isize).max(0) as usize;
                offset = cp.next_offset;
                batch = cp.batch_size.max(10_000);
            }
        }
    }

    let total_parts = parts1.len();
    let mut total_written = 0usize;
    for pi in start_part..total_parts {
        let p1 = &parts1[pi];
        let p2 = &parts2[pi];
        // Index inner table for this partition
        let c1 = get_person_count_where(pool, table1, &p1.where_sql, &p1.binds).await?;
        let c2 = get_person_count_where(pool, table2, &p2.where_sql, &p2.binds).await?;
        let inner_is_t2 = c2 <= c1;
        let (inner_table, inner_where, inner_binds, inner_map) = if inner_is_t2 { (table2, &p2.where_sql, &p2.binds, mapping2) } else { (table1, &p1.where_sql, &p1.binds, mapping1) };
        let (outer_table, outer_where, outer_binds, outer_map) = if inner_is_t2 { (table1, &p1.where_sql, &p1.binds, mapping1) } else { (table2, &p2.where_sql, &p2.binds, mapping2) };
        let total_outer = if inner_is_t2 { c1 } else { c2 };


        let mems = memory_stats_mb();
        on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems.used_mb, mem_avail_mb: mems.avail_mb, stage: "indexing", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
        let inner_rows = get_person_rows_where(pool, inner_table, inner_where, inner_binds, inner_map).await?;
        let mems2 = memory_stats_mb();
        on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems2.used_mb, mem_avail_mb: mems2.avail_mb, stage: "indexing_done", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

        // Precompute normalized inner and group by birthdate to bound comparisons
        use std::collections::HashMap as Map;
        let norm_inner: Vec<NormalizedPerson> = inner_rows.iter().map(normalize_person).collect();
        let mut by_date: Map<chrono::NaiveDate, Vec<usize>> = Map::new();
        for (i, n) in norm_inner.iter().enumerate() {
            if let Some(d) = n.birthdate.as_ref() { by_date.entry(*d).or_default().push(i); }
        }

        // Optional GPU: we'll attempt it when feature is enabled; else CPU fallback

        let start_time = Instant::now();
        let mut processed = 0usize;
        if pi != start_part { offset = 0; }
        while offset < total_outer {
            if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(100)).await; } }
            let mem = memory_stats_mb();
            if mem.avail_mb < cfg.memory_soft_min_mb && batch > 10_000 { batch = (batch / 2).max(10_000); }

            // fetch chunk with WHERE
            let mut tries = 0u32;
            let rows: Vec<Person> = loop {
                match fetch_person_rows_chunk_where(pool, outer_table, offset, batch, outer_where, outer_binds, outer_map).await {
                    Ok(v) => break v,
                    Err(e) => { tries += 1; if tries > cfg.retry_max { return Err(e); } let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5)-1)); tokio::time::sleep(std::time::Duration::from_millis(backoff)).await; }
                }
            };


            // Optional GPU pre-pass: hash-based candidate filtering for Fuzzy direct phase
            #[allow(unused_mut)]
            let mut gpu_done = false;
            #[cfg(feature = "gpu")]
            if cfg.use_gpu_fuzzy_direct_hash {
                // Indicate GPU build/probe hashing activity for GUI status lights
                on_progress(ProgressUpdate { processed: processed.min(total_outer as usize), total: total_outer as usize, percent: ((processed as f32)/(total_outer.max(1) as f32))*100.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_hash", batch_size_current: Some(batch), gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
                match gpu::fuzzy_direct_gpu_hash_prefilter_indices(&rows, &inner_rows, &part_cfg.strategy) {
                    Ok(cand_lists) => {
                        on_progress(ProgressUpdate { processed: processed.min(total_outer as usize), total: total_outer as usize, percent: ((processed as f32)/(total_outer.max(1) as f32))*100.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_probe_hash", batch_size_current: Some(batch), gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
                        for (i, p) in rows.iter().enumerate() {
                            let n = normalize_person(p);
                            for &i2 in cand_lists.get(i).map(|v| v.as_slice()).unwrap_or(&[]) {
                                let n2 = &norm_inner[i2];
                                // Enforce exact birthdate equality (algorithm requirement)
                                if !n.birthdate.as_ref().zip(n2.birthdate.as_ref()).map_or(false, |(a,b)| a==b) { continue; }
                                // If strategy does not enforce last initial, keep it permissive
                                if part_cfg.strategy != "last_initial" {
                                    let li1 = n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase();
                                    let li2 = n2.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase();
                                    if li1 != li2 { continue; }
                                }
                                let comp = if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) {
                                    fuzzy_compare_names_no_mid(
                                        n.first_name.as_deref(), n.last_name.as_deref(),
                                        n2.first_name.as_deref(), n2.last_name.as_deref(),
                                    )
                                } else {
                                    fuzzy_compare_names_new(
                                        n.first_name.as_deref(), n.middle_name.as_deref(), n.last_name.as_deref(),
                                        n2.first_name.as_deref(), n2.middle_name.as_deref(), n2.last_name.as_deref(),
                                    )
                                };
                                if let Some((score, label)) = comp {
                                    let q = &inner_rows[i2];
                                    let pair = MatchPair { person1: if inner_is_t2 { p.clone() } else { q.clone() }, person2: if inner_is_t2 { q.clone() } else { p.clone() }, confidence: (score/100.0) as f32, matched_fields: vec!["fuzzy".into(), label, "birthdate".into()], is_matched_infnbd: false, is_matched_infnmnbd: false };
                                    on_match(&pair)?; total_written += 1;
                                }
                            }
                        }
                        on_progress(ProgressUpdate { processed: processed.min(total_outer as usize), total: total_outer as usize, percent: ((processed as f32)/(total_outer.max(1) as f32))*100.0, eta_secs: 0, mem_used_mb: memory_stats_mb().used_mb, mem_avail_mb: memory_stats_mb().avail_mb, stage: "gpu_probe_hash_done", batch_size_current: Some(batch), gpu_total_mb: 1, gpu_free_mb: 0, gpu_active: true });
                        gpu_done = true;
                    }
                    Err(e) => { log::warn!("GPU fuzzy direct pre-pass failed; continuing to full scoring: {}", e); }
                }
            }

            // Try GPU first (per-chunk) if available; fallback to CPU if disabled/failed
            #[cfg(feature = "gpu")]
            {
                let opts = MatchOptions { backend: ComputeBackend::Gpu, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: 512 }), progress: ProgressConfig::default() };
                let gpu_try = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu::match_fuzzy_gpu(&rows, &inner_rows, opts, &on_progress)));
                match gpu_try {
                    Ok(Ok(mut vec_pairs)) => {
                        for pair in vec_pairs.drain(..) {
                            let comp = if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) {
                                compare_persons_no_mid(&pair.person1, &pair.person2)
                            } else {
                                compare_persons_new(&pair.person1, &pair.person2)
                            };
                            if let Some((score, label)) = comp {
                                let mut updated = pair;
                                updated.confidence = (score/100.0) as f32;
                                updated.matched_fields = vec!["fuzzy".into(), label, "birthdate".into()];
                                on_match(&updated)?; total_written += 1;
                            }
                        }
                        gpu_done = true;
                    }
                    Ok(Err(e)) => { log::warn!("GPU fuzzy failed in partition; falling back to CPU: {}", e); }
                    Err(_) => { log::warn!("GPU fuzzy panicked; falling back to CPU for this chunk"); }
                }
            }

                if !gpu_done {
                    // CPU fallback: candidate window by exact birthdate
                    for p in rows.iter() {
                        let n = normalize_person(p);
                        if let Some(d) = n.birthdate.as_ref() {
                            if let Some(cand_idx) = by_date.get(d) {
                                for &i2 in cand_idx {
                                    let n2 = &norm_inner[i2];
                                    // quick initial filter: last initial must match if not enforced by strategy
                                    if part_cfg.strategy != "last_initial" {
                                        let li1 = n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase();
                                        let li2 = n2.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase();
                                        if li1 != li2 { continue; }
                                    }
                                    let comp = if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) {
                                        fuzzy_compare_names_no_mid(
                                            n.first_name.as_deref(), n.last_name.as_deref(),
                                            n2.first_name.as_deref(), n2.last_name.as_deref(),
                                        )
                                    } else {
                                        fuzzy_compare_names_new(
                                            n.first_name.as_deref(), n.middle_name.as_deref(), n.last_name.as_deref(),
                                            n2.first_name.as_deref(), n2.middle_name.as_deref(), n2.last_name.as_deref(),
                                        )
                                    };
                                    if let Some((score, label)) = comp {
                                        let q = &inner_rows[i2];
                                        let pair = MatchPair { person1: if inner_is_t2 { p.clone() } else { q.clone() }, person2: if inner_is_t2 { q.clone() } else { p.clone() }, confidence: (score/100.0) as f32, matched_fields: vec!["fuzzy".into(), label, "birthdate".into()], is_matched_infnbd: false, is_matched_infnmnbd: false };
                                        on_match(&pair)?; total_written += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                // Iteration tail: update checkpoint and progress, yield
                offset += batch; processed = (processed + rows.len()).min(total_outer as usize);
                if let Some(pth) = cfg.checkpoint_path.as_ref() {
                    let _ = save_checkpoint(pth, &StreamCheckpoint { db: String::new(), table_inner: inner_table.to_string(), table_outer: outer_table.to_string(), algorithm: format!("{:?}", algo), batch_size: batch, next_offset: offset, total_outer, partition_idx: pi as i32, partition_name: p1.name.clone(), updated_utc: chrono::Utc::now().to_rfc3339() });
                }
                let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                let eta = if frac > 0.0 { (start_time.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                let memx = memory_stats_mb();
                on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac * 100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: if gpu_done { "gpu_kernel" } else { "streaming" }, batch_size_current: Some(batch), gpu_total_mb: if gpu_done { 1 } else { 0 }, gpu_free_mb: 0, gpu_active: gpu_done });
                tokio::task::yield_now().await;
            }

    }
    if let Some(pth) = cfg.checkpoint_path.as_ref() { remove_checkpoint(pth); }
    Ok(total_written)
}

