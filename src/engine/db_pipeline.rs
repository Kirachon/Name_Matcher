#[cfg(feature = "new_engine")]
use anyhow::Result;
#[cfg(feature = "new_engine")]
use sqlx::MySqlPool;

#[cfg(feature = "new_engine")]
use crate::matching::{MatchingAlgorithm, StreamingConfig, ProgressUpdate, StreamControl};
#[cfg(feature = "new_engine")]
use crate::matching::MatchPair;
#[cfg(feature = "new_engine")]
use crate::models::ColumnMapping;

/// Database-backed streaming facades for the new engine.
///
/// Notes:
/// - To preserve GPU acceleration, performance, and exact parity, these
///   facades currently delegate to the legacy streaming implementations.
/// - They provide a stable API surface under `engine::db_pipeline` so we can
///   incrementally replace internals with the new trait-based engine without
///   changing call sites.
/// - All functions are compiled only when the `new_engine` feature is enabled.
#[cfg(feature = "new_engine")]
pub mod db_pipeline {
    use super::*;
        use chrono::Datelike;


    /// Single-database streaming (auto-picks inner/outer by row count) using the trait-based StreamEngine.
    pub async fn stream_new_engine_single<F>(
        pool: &MySqlPool,
        table1: &str,
        table2: &str,
        algo: MatchingAlgorithm,
        mut on_match: F,
        cfg: StreamingConfig,
        on_progress: impl Fn(ProgressUpdate) + Sync,
        ctrl: Option<StreamControl>,
    ) -> Result<usize>
    where
        F: FnMut(&MatchPair) -> Result<()>,
    {
        use crate::db::schema::{get_person_count, fetch_person_rows_chunk};
        use crate::engine::{StreamEngine, FnPartitioner};
        use crate::engine::file_checkpointer::FileCheckpointer;
        use crate::engine::legacy_adapters::{LegacyAdapterAlgo1, LegacyAdapterAlgo2, LegacyAdapterFuzzy, LegacyAdapterFuzzyNoMiddle};
        use crate::normalize::normalize_person;
        use std::time::Instant;

        // Preserve GPU accelerated single-DB path by delegating only when requested
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_hash_join {
            return crate::matching::stream_match_csv(pool, table1, table2, algo, on_match, cfg, on_progress, ctrl).await;
        }

        let c1 = get_person_count(pool, table1).await?;
        let c2 = get_person_count(pool, table2).await?;
        let inner_is_t2 = c2 <= c1;
        let inner_table = if inner_is_t2 { table2 } else { table1 };
        let outer_table = if inner_is_t2 { table1 } else { table2 };
        let total_outer = if inner_is_t2 { c1 } else { c2 };

        // Load entire inner side into memory (same as legacy index build)
        let batch = cfg.batch_size.max(10_000);
        let mut inner_rows: Vec<crate::models::Person> = Vec::new();
        let mut inner_off: i64 = 0;
        on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: crate::metrics::memory_stats_mb().used_mb, mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb, stage: "indexing", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
        loop {
            let rows = fetch_person_rows_chunk(pool, inner_table, inner_off, batch).await?;
            if rows.is_empty() { break; }
            inner_off += rows.len() as i64;
            inner_rows.extend(rows);
        }
        on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: crate::metrics::memory_stats_mb().used_mb, mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb, stage: "indexing_done", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

        // Build partitioner factory
        let key_algo = algo;
        let part_make = || FnPartitioner::<crate::models::Person, _>(move |p| {
            crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default()
        }, std::marker::PhantomData);

        let ck_path = cfg.checkpoint_path.clone().unwrap_or_else(|| "engine_ck.db".into());
        let job = format!("single:{}->{}", inner_table, outer_table);
        let mut total_written = 0usize;
        match algo {
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterAlgo2, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::Fuzzy => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterFuzzy, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::FuzzyNoMiddle => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterFuzzyNoMiddle, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::FuzzyBirthdate => { anyhow::bail!("Algorithm 7 (FuzzyBirthdate) is deprecated and not supported in streaming mode.") }
        }
        Ok(total_written)
    }

    /// Cross-database streaming (different pools for table1 and table2) using trait-based engine.
    pub async fn stream_new_engine_dual<F>(
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
    where
        F: FnMut(&MatchPair) -> Result<()>,
    {
        use crate::db::schema::{get_person_count, fetch_person_rows_chunk};
        use crate::engine::{StreamEngine, FnPartitioner};
        use crate::engine::file_checkpointer::FileCheckpointer;
        use crate::engine::legacy_adapters::{LegacyAdapterAlgo1, LegacyAdapterAlgo2, LegacyAdapterFuzzy, LegacyAdapterFuzzyNoMiddle};
        use crate::normalize::normalize_person;
        use std::time::Instant;

        let c1 = get_person_count(pool1, table1).await?;
        let c2 = get_person_count(pool2, table2).await?;
        let inner_is_t2 = c2 <= c1;
        let inner_pool = if inner_is_t2 { pool2 } else { pool1 };
        let outer_pool = if inner_is_t2 { pool1 } else { pool2 };
        let inner_table = if inner_is_t2 { table2 } else { table1 };
        let outer_table = if inner_is_t2 { table1 } else { table2 };
        let total_outer = if inner_is_t2 { c1 } else { c2 };

        let batch = cfg.batch_size.max(10_000);
        let mut inner_rows: Vec<crate::models::Person> = Vec::new();
        let mut inner_off: i64 = 0;
        on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: crate::metrics::memory_stats_mb().used_mb, mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb, stage: "indexing", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
        loop {
            let rows = fetch_person_rows_chunk(inner_pool, inner_table, inner_off, batch).await?;
            if rows.is_empty() { break; }
            inner_off += rows.len() as i64;
            inner_rows.extend(rows);
        }
        on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: crate::metrics::memory_stats_mb().used_mb, mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb, stage: "indexing_done", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

        // Build partitioner factory
        let key_algo = algo;
        let part_make = || FnPartitioner::<crate::models::Person, _>(move |p| {
            crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default()
        }, std::marker::PhantomData);

        let ck_path = cfg.checkpoint_path.clone().unwrap_or_else(|| "engine_ck.db".into());
        let job = format!("dual:{}->{}", inner_table, outer_table);
        let mut total_written = 0usize;
        match algo {
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterAlgo2, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::Fuzzy => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterFuzzy, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::FuzzyNoMiddle => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterFuzzyNoMiddle, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
                let part_a = part_make(); let part_b = part_make();
                let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                let mut offset: i64 = 0; let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                    let rows = fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() { break; }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _expl| {
                        let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                        let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                    })?; total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::FuzzyBirthdate => { anyhow::bail!("Algorithm 7 (FuzzyBirthdate) is deprecated and not supported in streaming mode.") }
        }
        Ok(total_written)
    }

    /// Partition-aware streaming with optional column mappings using trait-based engine.
    pub async fn stream_new_engine_partitioned<F>(
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
        part_cfg: crate::matching::PartitioningConfig,
    ) -> Result<usize>
    where
        F: FnMut(&MatchPair) -> Result<()>,
    {
        use crate::util::partition::{DefaultPartition, PartitionStrategy};
        use crate::db::schema::{get_person_count_where, get_person_rows_where, fetch_person_rows_chunk_where};
        use crate::engine::{StreamEngine, FnPartitioner};
        use crate::engine::file_checkpointer::FileCheckpointer;
        use crate::engine::legacy_adapters::{LegacyAdapterAlgo1, LegacyAdapterAlgo2, LegacyAdapterFuzzy, LegacyAdapterFuzzyNoMiddle};
        use crate::normalize::normalize_person;
        use std::collections::HashMap;
        use std::time::Instant;

        let strat: Box<dyn PartitionStrategy + Send + Sync> = match part_cfg.strategy.as_str() {
            "birthyear5" => DefaultPartition::BirthYear5.build(),
            _ => DefaultPartition::LastInitial.build(),
        };
        let parts1 = strat.partitions(mapping1);
        let parts2 = strat.partitions(mapping2);
        if parts1.len() != parts2.len() { anyhow::bail!("Partition strategy produced mismatched partition counts for the two tables"); }

        let matcher_direct = matches!(algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd | MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6);
        let mut total_written = 0usize;
        let mut start_part: usize = 0;
        let mut offset: i64 = 0;
        let mut batch = cfg.batch_size.max(10_000);
        if cfg.resume {
            if let Some(pth) = cfg.checkpoint_path.as_ref() {
                if let Some(cp) = crate::util::checkpoint::load_checkpoint(pth) {
                    start_part = (cp.partition_idx as isize).max(0) as usize;
                    offset = cp.next_offset; batch = cp.batch_size.max(10_000);
                }
            }
        }

        for pi in start_part..parts1.len() {
            let p1 = &parts1[pi];
            let p2 = &parts2[pi];
            let c1 = get_person_count_where(pool, table1, &p1.where_sql, &p1.binds).await?;
            let c2 = get_person_count_where(pool, table2, &p2.where_sql, &p2.binds).await?;
            let inner_is_t2 = c2 <= c1;
            let (inner_table, inner_where, inner_binds, inner_map) = if inner_is_t2 { (table2, &p2.where_sql, &p2.binds, mapping2) } else { (table1, &p1.where_sql, &p1.binds, mapping1) };
            let (outer_table, outer_where, outer_binds, outer_map) = if inner_is_t2 { (table1, &p1.where_sql, &p1.binds, mapping1) } else { (table2, &p2.where_sql, &p2.binds, mapping2) };
            let total_outer = if inner_is_t2 { c1 } else { c2 };

            // Load inner rows for this partition
            on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: crate::metrics::memory_stats_mb().used_mb, mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb, stage: "indexing", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
            let inner_rows = get_person_rows_where(pool, inner_table, inner_where, inner_binds, inner_map).await?;
            on_progress(ProgressUpdate { processed: 0, total: total_outer as usize, percent: 0.0, eta_secs: 0, mem_used_mb: crate::metrics::memory_stats_mb().used_mb, mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb, stage: "indexing_done", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });

            // Prepare partitioners
            let ck_path = cfg.checkpoint_path.clone().unwrap_or_else(|| "engine_ck.db".into());
            let last_initial = !matcher_direct && part_cfg.strategy == "last_initial";
            let key_algo = algo;
            let part = FnPartitioner::<crate::models::Person, _>(move |p| {
                if last_initial {
                    let n = normalize_person(p);
                    n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string()
                } else {
                    crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default()
                }
            }, std::marker::PhantomData);

            // Build engine per algorithm at use-site to avoid heterogeneous generic types

            // Group inner set based on algorithm needs to bound candidate explosion
            let mut by_date: Option<HashMap<chrono::NaiveDate, Vec<crate::models::Person>>> = None;
            let mut by_year: Option<HashMap<i32, Vec<crate::models::Person>>> = None;
            if !matcher_direct {
                // Group by exact birthdate for fuzzy 3/4; Algorithm 7 is deprecated
                let mut map: HashMap<chrono::NaiveDate, Vec<crate::models::Person>> = HashMap::new();
                for p in inner_rows.iter() {
                    if let Some(d) = p.birthdate { map.entry(d).or_default().push(p.clone()); }
                }
                by_date = Some(map);
            }

            if pi != start_part { offset = 0; }
            let start = Instant::now();
            while offset < total_outer {
                if let Some(c) = &ctrl { if c.cancel.load(std::sync::atomic::Ordering::Relaxed) { break; } while c.pause.load(std::sync::atomic::Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } }
                let rows = fetch_person_rows_chunk_where(pool, outer_table, offset, batch, outer_where, outer_binds, outer_map).await?;
                if rows.is_empty() { break; }
                offset += rows.len() as i64;
                let processed = (offset as usize).min(total_outer as usize);
                let job = format!("part:{}:{}->{}", pi, inner_table, outer_table);
                let wrote = if matcher_direct {
                    match algo {
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                            eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _e| {
                                let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                            })?
                        }
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let mut eng = StreamEngine::new(LegacyAdapterAlgo2, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                            eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _e| {
                                let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                            })?
                        }
                        MatchingAlgorithm::Fuzzy => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let mut eng = StreamEngine::new(LegacyAdapterFuzzy, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                            eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _e| {
                                let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                            })?
                        }
                        MatchingAlgorithm::FuzzyNoMiddle => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let mut eng = StreamEngine::new(LegacyAdapterFuzzyNoMiddle, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                            eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _e| {
                                let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                            })?
                        }
                        MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                            let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                            eng.for_each(&job, rows.iter(), inner_rows.iter(), |a, b, score, _e| {
                                let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                            })?
                        }
                        MatchingAlgorithm::FuzzyBirthdate => { 0 }
                    }
                } else {
                    let mut wrote = 0usize;
                    if let Some(map) = by_date.as_ref() {
                        use std::collections::HashMap as Hm;
                        let mut out_map: Hm<chrono::NaiveDate, Vec<&crate::models::Person>> = Hm::new();
                        for p in rows.iter() { if let Some(d) = p.birthdate { out_map.entry(d).or_default().push(p); } }
                        match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(&job, outs.iter().copied(), inners.iter(), |a, b, score, _e| {
                                            let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                            let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                                        })?;
                                    }
                                }
                            }
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let mut eng = StreamEngine::new(LegacyAdapterAlgo2, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(&job, outs.iter().copied(), inners.iter(), |a, b, score, _e| {
                                            let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                            let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                                        })?;
                                    }
                                }
                            }
                            MatchingAlgorithm::Fuzzy => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let mut eng = StreamEngine::new(LegacyAdapterFuzzy, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(&job, outs.iter().copied(), inners.iter(), |a, b, score, _e| {
                                            let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                            let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                                        })?;
                                    }
                                }
                            }
                            MatchingAlgorithm::FuzzyNoMiddle => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let mut eng = StreamEngine::new(LegacyAdapterFuzzyNoMiddle, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(&job, outs.iter().copied(), inners.iter(), |a, b, score, _e| {
                                            let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                            let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                                        })?;
                                    }
                                }
                            }
                            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let part_b = FnPartitioner::<crate::models::Person, _>(move |p| { if last_initial { let n = normalize_person(p); n.last_name.as_deref().and_then(|s| s.chars().next()).unwrap_or('\0').to_ascii_uppercase().to_string() } else { crate::matching::key_for_engine(key_algo, &normalize_person(p)).unwrap_or_default() } }, std::marker::PhantomData);
                                let mut eng = StreamEngine::new(LegacyAdapterAlgo1, part_a, part_b, FileCheckpointer::new(ck_path.clone()));
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(&job, outs.iter().copied(), inners.iter(), |a, b, score, _e| {
                                            let pair = if inner_is_t2 { crate::matching::to_pair_public(a, b, algo) } else { crate::matching::to_pair_public(b, a, algo) };
                                            let mut pair = pair; pair.confidence = (score as f32) / 100.0; on_match(&pair)
                                        })?;
                                    }
                                }
                            }
                            MatchingAlgorithm::FuzzyBirthdate => { }
                        }
                    }
                    wrote
                };
                total_written += wrote;
                let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                let eta = if frac > 0.0 { (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64 } else { 0 };
                let memx = crate::metrics::memory_stats_mb();
                on_progress(ProgressUpdate { processed, total: total_outer as usize, percent: frac*100.0, eta_secs: eta, mem_used_mb: memx.used_mb, mem_avail_mb: memx.avail_mb, stage: "streaming", batch_size_current: Some(batch), gpu_total_mb: 0, gpu_free_mb: 0, gpu_active: false });
                if let Some(pth) = cfg.checkpoint_path.as_ref() {
                    let _ = crate::util::checkpoint::save_checkpoint(pth, &crate::util::checkpoint::StreamCheckpoint { db: String::new(), table_inner: inner_table.to_string(), table_outer: outer_table.to_string(), algorithm: format!("{:?}", algo), batch_size: batch, next_offset: offset, total_outer, partition_idx: pi as i32, partition_name: p1.name.clone(), updated_utc: chrono::Utc::now().to_rfc3339() });
                }
                tokio::task::yield_now().await;
            }
        }
        if let Some(pth) = cfg.checkpoint_path.as_ref() { crate::util::checkpoint::remove_checkpoint(pth); }
        Ok(total_written)
    }
}

