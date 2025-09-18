use crate::models::{NormalizedPerson, Person};
use crate::normalize::normalize_person;
use crate::metrics::memory_stats_mb;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use strsim::{levenshtein, jaro_winkler};
use rphonetic::{DoubleMetaphone, Encoder};
use unicode_normalization::UnicodeNormalization;

fn normalize_simple(s: &str) -> String {
    s.trim().to_lowercase().replace('.', "").replace('-', " ")
}

fn sim_levenshtein_pct(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 { return 100.0; }
    let dist = levenshtein(a, b);
    (1.0 - (dist as f64 / max_len as f64)) * 100.0
}

fn normalize_for_phonetic(s: &str) -> String {
    // Lowercase, decompose diacritics, keep ASCII letters and single spaces; map a few common non-ASCII
    let s = s.trim().to_lowercase();
    let mut out = String::with_capacity(s.len());
    for ch in s.nfd() {
        if ch.is_ascii_alphabetic() {
            out.push(ch);
        } else if ch.is_ascii_whitespace() {
            if !out.ends_with(' ') { out.push(' '); }
        } else {
            match ch {
                'ß' => out.push_str("ss"),
                'æ' | 'ǽ' => out.push_str("ae"),
                'ø' => out.push('o'),
                'đ' => out.push('d'),
                _ => {}
            }
        }
    }
    out.trim().to_string()
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

fn fuzzy_compare_names_new(n1_first: &str, n1_mid: Option<&str>, n1_last: &str, n2_first: &str, n2_mid: Option<&str>, n2_last: &str) -> Option<(f64, String)> {
    let full1 = normalize_simple(&format!("{} {} {}", n1_first, n1_mid.unwrap_or(""), n1_last));
    let full2 = normalize_simple(&format!("{} {} {}", n2_first, n2_mid.unwrap_or(""), n2_last));

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
            let ld_first = levenshtein(&normalize_simple(n1_first), &normalize_simple(n2_first)) as usize;
            let ld_last = levenshtein(&normalize_simple(n1_last), &normalize_simple(n2_last)) as usize;
            let ld_mid = levenshtein(&normalize_simple(n1_mid.unwrap_or("")), &normalize_simple(n2_mid.unwrap_or(""))) as usize;
            if ld_first <= 2 && ld_last <= 2 && ld_mid <= 2 {
                return Some((avg, "CASE 3".to_string()));
            }
        }
        return Some((avg, "CASE 2".to_string()));
    }

    None
}

fn compare_persons_new(p1: &Person, p2: &Person) -> Option<(f64, String)> {
    if p1.birthdate != p2.birthdate { return None; }
    fuzzy_compare_names_new(&p1.first_name, p1.middle_name.as_deref(), &p1.last_name,
                            &p2.first_name, p2.middle_name.as_deref(), &p2.last_name)
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MatchingAlgorithm { IdUuidYasIsMatchedInfnbd, IdUuidYasIsMatchedInfnmnbd, Fuzzy }

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
    pub stage: &'static str,
    pub batch_size_current: Option<i64>,
    // GPU-related (0/false when CPU-only)
    pub gpu_total_mb: u64,
    pub gpu_free_mb: u64,
    pub gpu_active: bool,
}


// --- Optional GPU backend abstraction ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend { Cpu, Gpu }

#[derive(Debug, Clone, Copy)]
pub struct GpuConfig { pub device_id: Option<usize>, pub mem_budget_mb: u64 }

#[derive(Debug, Clone, Copy)]
pub struct MatchOptions { pub backend: ComputeBackend, pub gpu: Option<GpuConfig>, pub progress: ProgressConfig }

impl Default for MatchOptions {
    fn default() -> Self {
        Self { backend: ComputeBackend::Cpu, gpu: None, progress: ProgressConfig::default() }
    }
}

pub fn match_all_with_opts<F>(table1: &[Person], table2: &[Person], algo: MatchingAlgorithm, opts: MatchOptions, progress: F) -> Vec<MatchPair>
where F: Fn(ProgressUpdate) + Sync,
{
    if matches!(algo, MatchingAlgorithm::Fuzzy) && matches!(opts.backend, ComputeBackend::Gpu) {
        #[cfg(feature = "gpu")]
        {
            match gpu::match_fuzzy_gpu(table1, table2, opts, &progress) {
                Ok(v) => return v,
                Err(e) => { log::warn!("GPU fuzzy failed, falling back to CPU: {}", e); }
            }
        }
        // If feature disabled or GPU path failed, fall through to CPU
    }
    match_all_progress(table1, table2, algo, opts.progress, progress)
}


fn matches_algo1(p1: &NormalizedPerson, p2: &NormalizedPerson) -> bool { p1.last_name == p2.last_name && p1.first_name == p2.first_name && p1.birthdate == p2.birthdate }
fn matches_algo2(p1: &NormalizedPerson, p2: &NormalizedPerson) -> bool {
    let middle_match = match (&p1.middle_name, &p2.middle_name) { (Some(a), Some(b)) => a == b, (None, None) => true, _ => false };
    p1.last_name == p2.last_name && p1.first_name == p2.first_name && middle_match && p1.birthdate == p2.birthdate
}

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
                        if p1.birthdate == p2.birthdate {
                            if let Some((score, label)) = fuzzy_compare_names_new(
                                &p1.first_name,
                                p1.middle_name.as_deref(),
                                &p1.last_name,
                                &p2.first_name,
                                p2.middle_name.as_deref(),
                                &p2.last_name,
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
            if d!=0 && d!=last { out[idx]=(b'0'+d); idx+=1; }
            last=d;
        }
        out
    }

    #[cfg(feature = "gpu")]
    fn cuda_mem_info_mb(dev: &cudarc::driver::CudaDevice) -> (u64,u64) {
        // Try cuMemGetInfo_v2; if unavailable, return zeros and rely on backoff.
        #[allow(unused_unsafe)] unsafe {
            let mut free: usize = 0; let mut total: usize = 0;
            let res = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _ as *mut _, &mut total as *mut _ as *mut _);
            if res == 0 { ((total as u64)/1024/1024, (free as u64)/1024/1024) } else { (0,0) }
        }
    }

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use anyhow::{anyhow, Result};
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
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
        let func_jaro = dev.get_func("lev_mod", "jaro_kernel").map_err(|e| anyhow!("Get jaro func failed: {e}"))?;
        let func_jw = dev.get_func("lev_mod", "jw_kernel").map_err(|e| anyhow!("Get jw func failed: {e}"))?;

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
    "#;

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
            let year = p.birthdate.year() as u16;
            let fi = p.first_name.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let li = p.last_name.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let sx = super::soundex4_ascii(&p.last_name);
            block.entry(BKey(year, fi, li, sx)).or_default().push(j);
        }

        // 3) Prepare CUDA device
        let dev_id = opts.gpu.and_then(|g| g.device_id).unwrap_or(0);
        let dev = CudaDevice::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;
        let ptx = compile_ptx(LEV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
        dev.load_ptx(ptx, "lev_mod", &[]).map_err(|e| anyhow!("Load PTX failed: {e}"))?;
        let func = dev.get_func("lev_mod", "lev_kernel").map_err(|e| anyhow!("Get func failed: {e}"))?;
        let func_jaro = dev.get_func("lev_mod", "jaro_kernel").map_err(|e| anyhow!("Get jaro func failed: {e}"))?;
        let func_jw = dev.get_func("lev_mod", "jw_kernel").map_err(|e| anyhow!("Get jw func failed: {e}"))?;

        // Report GPU init and memory info
        let (gpu_total_mb, mut gpu_free_mb) = cuda_mem_info_mb(&dev);
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
        let mut processed: usize = 0;

        for (i, p1) in n1.iter().enumerate() {
            let year = p1.birthdate.year() as u16;
            let fi = p1.first_name.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let li = p1.last_name.bytes().find(|c| c.is_ascii_alphabetic()).unwrap_or(b'?').to_ascii_uppercase();
            let sx = super::soundex4_ascii(&p1.last_name);
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
            while start < cands_vec.len() {
                let end = (start + tile_max).min(cands_vec.len());
                let cur = &cands_vec[start..end];
                // Build flat buffers for names
                let mut a_offsets = Vec::<i32>::with_capacity(cur.len());
                let mut a_lengths = Vec::<i32>::with_capacity(cur.len());
                let mut b_offsets = Vec::<i32>::with_capacity(cur.len());
                let mut b_lengths = Vec::<i32>::with_capacity(cur.len());
                let mut a_bytes = Vec::<u8>::with_capacity(cur.len() * 32);
                let mut b_bytes = Vec::<u8>::with_capacity(cur.len() * 32);
                let mut names_a = Vec::<String>::with_capacity(cur.len()); // for CPU jaro/jw
                let mut names_b = Vec::<String>::with_capacity(cur.len());
                for &j_idx in cur {
                    let p2 = &n2[j_idx];
                    let s1 = normalize_simple(&format!("{} {} {}", p1.first_name, p1.middle_name.as_deref().unwrap_or(""), p1.last_name));
                    let s2 = normalize_simple(&format!("{} {} {}", p2.first_name, p2.middle_name.as_deref().unwrap_or(""), p2.last_name));
                    names_a.push(s1.clone()); names_b.push(s2.clone());
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

                // Adaptive attempt with backoff on OOM
                let mut attempt_ok = false;
                loop {
                    // refresh GPU free mem for display
                    let (_tot, free_now) = cuda_mem_info_mb(&dev); gpu_free_mb = free_now.max(gpu_free_mb);
                    let try_run: anyhow::Result<Vec<f32>> = (|| {
                        // Device buffers
                        let d_a = dev.htod_copy(a_bytes.as_slice())?;
                        let d_a_off = dev.htod_copy(a_offsets.as_slice())?;
                        let d_a_len = dev.htod_copy(a_lengths.as_slice())?;
                        let d_b = dev.htod_copy(b_bytes.as_slice())?;
                        let d_b_off = dev.htod_copy(b_offsets.as_slice())?;
                        let d_b_len = dev.htod_copy(b_lengths.as_slice())?;
                        let mut d_out = dev.alloc_zeros::<f32>(n_pairs)?;
                        let mut d_out_j = dev.alloc_zeros::<f32>(n_pairs)?;
                        let mut d_out_w = dev.alloc_zeros::<f32>(n_pairs)?;
                        // Launch kernels
                        let cfg = LaunchConfig::for_num_elems(n_pairs as u32);
                        unsafe { func.launch(cfg, (&d_a, &d_a_off, &d_a_len, &d_b, &d_b_off, &d_b_len, &mut d_out, n_pairs as i32)) }?;
                        unsafe { func_jaro.launch(cfg, (&d_a, &d_a_off, &d_a_len, &d_b, &d_b_off, &d_b_len, &mut d_out_j, n_pairs as i32)) }?;
                        unsafe { func_jw.launch(cfg, (&d_a, &d_a_off, &d_a_len, &d_b, &d_b_off, &d_b_len, &mut d_out_w, n_pairs as i32)) }?;
                        // Read back and combine
                        let lev_scores: Vec<f32> = dev.dtoh_sync_copy(&d_out)?;
                        let jaro_scores: Vec<f32> = dev.dtoh_sync_copy(&d_out_j)?;
                        let jw_scores: Vec<f32> = dev.dtoh_sync_copy(&d_out_w)?;
                        let mut final_scores = lev_scores;
                        for idx in 0..final_scores.len() {
                            let mut s = final_scores[idx];
                            if jaro_scores[idx] > s { s = jaro_scores[idx]; }
                            if jw_scores[idx] > s { s = jw_scores[idx]; }
                            final_scores[idx] = s;
                        }
                        Ok(final_scores)
                    })();

                    match try_run {
                        Ok(lev_scores) => {
                            // Compose results with CPU JR/JW (will move to GPU later)
                            for (k, &j_idx) in cur.iter().enumerate() {
                                let p2 = &n2[j_idx];
                                let score = lev_scores[k] as f64;
                                if score >= 85.0 {
                                    let mut fields = vec!["id","uuid","first_name","last_name","birthdate"]; // middle may be empty
                                    if n1[i].middle_name.is_some() || p2.middle_name.is_some() { fields.insert(3, "middle_name"); }
                                    results.push(MatchPair {
                                        person1: t1[i].clone(), person2: t2[j_idx].clone(),
                                        confidence: score as f32,
                                        matched_fields: fields.into_iter().map(String::from).collect(),
                                        is_matched_infnbd: false, is_matched_infnmnbd: false,
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

                processed += cur.len();
                let total_pairs_est = total.max(1) * 1; // rough; we still show percent over outer loop
                let frac = (i as f32 / total as f32).clamp(0.0, 1.0);
                let mem = memory_stats_mb();
                on_progress(ProgressUpdate { processed: i+1, total, percent: frac*100.0, eta_secs: 0, mem_used_mb: mem.used_mb, mem_avail_mb: mem.avail_mb, stage: "gpu_kernel", batch_size_current: Some(cur.len() as i64), gpu_total_mb: gpu_total_mb, gpu_free_mb: gpu_free_mb, gpu_active: true });

                start = end;
            }
        }
        Ok(results)
    }
}

fn to_original<'a>(np: &NormalizedPerson, originals: &'a [Person]) -> Person {
    originals.iter().find(|p| p.id == np.id).cloned().unwrap_or_else(|| Person { id: np.id, uuid: np.uuid.clone(), first_name: String::new(), middle_name: None, last_name: String::new(), birthdate: np.birthdate })
}

#[cfg(test)]
mod tests {
    use super::*; use chrono::NaiveDate; use std::sync::{Arc, Mutex};
    fn p(id: i64, f: &str, m: Option<&str>, l: &str, d: (i32, u32, u32)) -> Person { Person { id, uuid: format!("u{}", id), first_name: f.into(), middle_name: m.map(|s| s.to_string()), last_name: l.into(), birthdate: NaiveDate::from_ymd_opt(d.0, d.1, d.2).unwrap() } }
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
        assert!((v.last().unwrap().percent - 100.0).abs() < 0.001);
    }
    #[test]
    fn fuzzy_basic() {
        let a = vec![p(1, "Jon", None, "Smith", (1990,1,1))];
        let b = vec![p(2, "John", None, "Smith", (1990,1,1))];
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
    #[test]
    fn compute_stream_cfg_bounds_and_flush() {
        let c1 = compute_stream_cfg(512);
        assert!(c1.batch_size >= 5_000);
        assert!(c1.flush_every >= 1000);
        let c2 = compute_stream_cfg(32_768);
        assert!(c2.batch_size <= 100_000);
        assert_eq!(c2.flush_every, (c2.batch_size as usize / 10).max(1000));
    }

    }


}


// --- Streaming matching and export for large datasets ---
use std::collections::HashMap;
use anyhow::Result;
use crate::db::{get_person_count, fetch_person_rows_chunk};
use sqlx::MySqlPool;


fn key_for(algo: MatchingAlgorithm, p: &NormalizedPerson) -> String {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => format!("{}\x1F{}\x1F{}", p.last_name, p.first_name, p.birthdate),
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => format!("{}\x1F{}\x1F{}\x1F{}", p.last_name, p.first_name, p.middle_name.clone().unwrap_or_default(), p.birthdate),
        MatchingAlgorithm::Fuzzy => unreachable!("key_for is not used for fuzzy algorithm"),
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
        MatchingAlgorithm::Fuzzy => vec!["fuzzy".into(), "birthdate".into()],
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
            let k = key_for(algo, &n);
            map.entry(k).or_default().push(p.clone());
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
}
impl Default for StreamingConfig {
    fn default() -> Self {
        Self { batch_size: 50_000, memory_soft_min_mb: 800, flush_every: 10_000, resume: true, retry_max: 5, retry_backoff_ms: 500, checkpoint_path: None }
    }
}

/// Compute an adaptive StreamingConfig based on available memory (MB).
/// Conservative defaults with clamped batch sizes for stability across machines.
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


#[derive(Clone)]
pub struct StreamControl { pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>, pub pause: std::sync::Arc<std::sync::atomic::AtomicBool> }

pub async fn stream_match_csv<F>(pool: &MySqlPool, table1: &str, table2: &str, algo: MatchingAlgorithm, mut on_match: F, cfg: StreamingConfig, on_progress: impl Fn(ProgressUpdate), ctrl: Option<StreamControl>) -> Result<usize>
where F: FnMut(&MatchPair) -> Result<()>
{
    use crate::util::checkpoint::{load_checkpoint, save_checkpoint, remove_checkpoint, StreamCheckpoint};
    if matches!(algo, MatchingAlgorithm::Fuzzy) {
        anyhow::bail!("Fuzzy algorithm is currently supported only in in-memory mode (CSV). Use algorithm=3 with CSV format.");
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
            let k = key_for(algo, &n);
            if let Some(cands) = index.get(&k) {
                for q in cands {
                    let n2 = normalize_person(q);
                    let ok = match algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&n, &n2), MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(&n, &n2), MatchingAlgorithm::Fuzzy => false };
                    if ok { let pair = if inner_table == table2 { to_pair(p, q, algo, &n, &n2) } else { to_pair(q, p, algo, &n2, &n) }; on_match(&pair)?; written += 1; }
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
    on_progress: impl Fn(ProgressUpdate),
    ctrl: Option<StreamControl>,
) -> Result<usize>
where F: FnMut(&MatchPair) -> Result<()>
{
    use crate::util::checkpoint::{load_checkpoint, save_checkpoint, remove_checkpoint, StreamCheckpoint};
    if matches!(algo, MatchingAlgorithm::Fuzzy) { anyhow::bail!("Fuzzy algorithm is currently supported only in in-memory mode (CSV). Use algorithm=3 with CSV format."); }
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
            let k = key_for(algo, &n);
            if let Some(cands) = index.get(&k) {
                for q in cands {
                    let n2 = normalize_person(q);
                    let ok = match algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&n, &n2), MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(&n, &n2), MatchingAlgorithm::Fuzzy => false };
                    if ok {
                        let pair = if inner_is_t2 { to_pair(p, q, algo, &n, &n2) } else { to_pair(q, p, algo, &n2, &n) };
                        on_match(&pair)?; written += 1;
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
    on_progress: impl Fn(ProgressUpdate),
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
        for (i, n) in norm_inner.iter().enumerate() { by_date.entry(n.birthdate).or_default().push(i); }

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

            // Try GPU first (per-chunk) if available; fallback to CPU if disabled/failed
            let mut gpu_done = false;
            #[cfg(feature = "gpu")]
            {
                let opts = MatchOptions { backend: ComputeBackend::Gpu, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: 512 }), progress: ProgressConfig::default() };
                let progress_cb = |u: ProgressUpdate| { on_progress(u); };
                match gpu::match_fuzzy_gpu(&rows, &inner_rows, opts, &progress_cb) {
                    Ok(mut vec_pairs) => {
                        for pair in vec_pairs.drain(..) {
                            if let Some((score, label)) = compare_persons_new(&pair.person1, &pair.person2) {
                                let mut updated = pair;
                                updated.confidence = (score/100.0) as f32;
                                updated.matched_fields = vec!["fuzzy".into(), label, "birthdate".into()];
                                on_match(&updated)?; total_written += 1;
                            }
                        }
                        gpu_done = true;
                    }
                    Err(e) => { log::warn!("GPU fuzzy failed in partition; falling back to CPU: {}", e); }
                }
            }

            if !gpu_done {
                // CPU fallback: candidate window by exact birthdate
                for p in rows.iter() {
                    let n = normalize_person(p);
                    if let Some(cand_idx) = by_date.get(&n.birthdate) {


                        for &i2 in cand_idx {
                            let n2 = &norm_inner[i2];
                            // quick initial filter: last initial must match if not enforced by strategy
                            if part_cfg.strategy != "last_initial" {
                                let li1 = n.last_name.chars().next().unwrap_or('\0').to_ascii_uppercase();
                                let li2 = n2.last_name.chars().next().unwrap_or('\0').to_ascii_uppercase();
                                if li1 != li2 { continue; }
                            }
                            if let Some((score, label)) = fuzzy_compare_names_new(&n.first_name, n.middle_name.as_deref(), &n.last_name, &n2.first_name, n2.middle_name.as_deref(), &n2.last_name) {
                                let q = &inner_rows[i2];
                                let pair = MatchPair { person1: if inner_is_t2 { p.clone() } else { q.clone() }, person2: if inner_is_t2 { q.clone() } else { p.clone() }, confidence: (score/100.0) as f32, matched_fields: vec!["fuzzy".into(), label, "birthdate".into()], is_matched_infnbd: false, is_matched_infnmnbd: false };
                                on_match(&pair)?; total_written += 1;
                            }
                        }
                    }
                }
            }

            offset += batch; processed = (processed + rows.len()).min(total_outer as usize);
            if let Some(pth) = cfg.checkpoint_path.as_ref() {
                let _ = save_checkpoint(pth, &StreamCheckpoint { db: String::new(), table_inner: inner_table.to_string(), table_outer: outer_table.to_string(), algorithm: format!("{:?}", algo), batch_size: batch, next_offset: offset, total_outer, partition_idx: pi as i32, partition_name: p1.name.clone(), updated_utc: chrono::Utc::now().to_rfc3339() });
            }
            let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
            let eta = if frac > 0.0 { (start_time.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
            let memx = memory_stats_mb();
            on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac * 100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: if gpu_done { "gpu_kernel" } else { "streaming" }, batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: gpu_done });
            tokio::task::yield_now().await;
        }
    }
    if let Some(pth) = cfg.checkpoint_path.as_ref() { remove_checkpoint(pth); }
    Ok(total_written)
}

