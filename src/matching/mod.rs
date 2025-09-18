use crate::models::{NormalizedPerson, Person};
use crate::normalize::normalize_person;
use crate::metrics::memory_stats_mb;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use strsim::{levenshtein, jaro, jaro_winkler};

fn normalize_simple(s: &str) -> String {
    s.trim().to_lowercase().replace('.', "").replace('-', " ")
}

fn sim_levenshtein_pct(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 { return 100.0; }
    let dist = levenshtein(a, b);
    (1.0 - (dist as f64 / max_len as f64)) * 100.0
}

fn fuzzy_score_names(n1_first: &str, n1_mid: Option<&str>, n1_last: &str, n2_first: &str, n2_mid: Option<&str>, n2_last: &str) -> (f64, &'static str) {
    let name1 = normalize_simple(&format!("{} {} {}", n1_first, n1_mid.unwrap_or(""), n1_last));
    let name2 = normalize_simple(&format!("{} {} {}", n2_first, n2_mid.unwrap_or(""), n2_last));
    let lev = sim_levenshtein_pct(&name1, &name2);
    let jr = jaro(&name1, &name2) * 100.0;
    let jw = jaro_winkler(&name1, &name2) * 100.0;
    let score = lev.max(jr).max(jw);
    let label = if score >= 95.0 { "Auto-Match" } else if score >= 85.0 { "Review" } else { "Reject" };
    (score, label)
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
                            let (score, label) = fuzzy_score_names(
                                &p1.first_name,
                                p1.middle_name.as_deref(),
                                &p1.last_name,
                                &p2.first_name,
                                p2.middle_name.as_deref(),
                                &p2.last_name,
                            );
                            if score >= 85.0 {
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
            progress(ProgressUpdate { processed: processed_outer, total, percent: frac * 100.0, eta_secs, mem_used_mb: mem.used_mb, mem_avail_mb: mem.avail_mb, stage: "matching", batch_size_current: None });
            last_update = processed_outer;
        }
        if chunk_start.elapsed() > cfg.long_op_threshold { let _m = memory_stats_mb(); }
    }
    results
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

#[derive(Clone, Copy)]
pub struct StreamingConfig { pub batch_size: i64, pub memory_soft_min_mb: u64 }
impl Default for StreamingConfig { fn default() -> Self { Self { batch_size: 50_000, memory_soft_min_mb: 800 } } }

#[derive(Clone)]
pub struct StreamControl { pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>, pub pause: std::sync::Arc<std::sync::atomic::AtomicBool> }

pub async fn stream_match_csv<F>(pool: &MySqlPool, table1: &str, table2: &str, algo: MatchingAlgorithm, mut on_match: F, cfg: StreamingConfig, on_progress: impl Fn(ProgressUpdate), ctrl: Option<StreamControl>) -> Result<usize>
where F: FnMut(&MatchPair) -> Result<()>
{
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
    on_progress(ProgressUpdate { processed: 0, total: total as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems.used_mb, mem_avail_mb: mems.avail_mb, stage: "indexing", batch_size_current: Some(batch) });
    let index = build_index(pool, inner_table, algo, batch).await?;
    let mems2 = memory_stats_mb();
    on_progress(ProgressUpdate { processed: 0, total: total as usize, percent: 0.0, eta_secs: 0, mem_used_mb: mems2.used_mb, mem_avail_mb: mems2.avail_mb, stage: "indexing_done", batch_size_current: Some(batch) });

    let mut offset: i64 = 0; let mut written = 0usize; let mut processed = 0usize;
    let mem0 = memory_stats_mb();
    while offset < total {
        if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(100)).await; } }
        // adjust batch under memory pressure
        let mem = memory_stats_mb();
        if mem.avail_mb < cfg.memory_soft_min_mb && batch > 10_000 { batch = (batch / 2).max(10_000); }
        let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
        for p in rows.iter() {
            if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
            let n = normalize_person(p);
            let k = key_for(algo, &n);
            if let Some(cands) = index.get(&k) {
                for q in cands {
                    let n2 = normalize_person(q);
                    let ok = match algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&n, &n2), MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(&n, &n2), MatchingAlgorithm::Fuzzy => false /* unreachable due to early bail */ };
                    if ok { let pair = if inner_table == table2 { to_pair(p, q, algo, &n, &n2) } else { to_pair(q, p, algo, &n2, &n) }; on_match(&pair)?; written += 1; }
                }
            }
        }
        offset += batch; processed = (processed + rows.len()).min(total as usize);
        let frac = (processed as f32 / total as f32).clamp(0.0, 1.0);
        let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
        let memx = memory_stats_mb();
        on_progress(ProgressUpdate { processed, total: total as usize, percent: frac * 100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch) });
    }
    let _mem_end = memory_stats_mb(); let _ = mem0; // could log delta
    Ok(written)
}
