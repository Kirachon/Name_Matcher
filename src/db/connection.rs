use crate::config::DatabaseConfig;
use anyhow::Result;
use sqlx::mysql::MySqlPoolOptions;
use sqlx::MySqlPool;
use std::time::Duration;

pub async fn make_pool(cfg: &DatabaseConfig) -> Result<MySqlPool> {
    make_pool_with_size(cfg, None).await
}

pub async fn make_pool_with_size(cfg: &DatabaseConfig, max: Option<u32>) -> Result<MySqlPool> {
    let url = cfg.to_url();
    let max_conn: u32 = if let Some(m) = max {
        m
    } else if let Ok(s) = std::env::var("NAME_MATCHER_POOL_SIZE") {
        s.parse().unwrap_or(0)
    } else {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) as u32;
        // OPTIMIZATION B1: Increase max connections from 32 to 64 for better streaming performance
        std::cmp::min(64, cores.saturating_mul(2))
    };
    let max_conn = if max_conn == 0 { 16 } else { max_conn };
    let min_conn: u32 = std::env::var("NAME_MATCHER_POOL_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    // OPTIMIZATION B1: Reduce acquire timeout from 30s to 5s for fail-fast behavior
    let acquire_ms: u64 = std::env::var("NAME_MATCHER_ACQUIRE_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(5_000);
    let idle_ms: u64 = std::env::var("NAME_MATCHER_IDLE_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(60_000);
    let life_ms: u64 = std::env::var("NAME_MATCHER_LIFETIME_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(600_000);

    let pool = MySqlPoolOptions::new()
        .max_connections(max_conn)
        .min_connections(min_conn)
        .acquire_timeout(Duration::from_millis(acquire_ms))
        .idle_timeout(Some(Duration::from_millis(idle_ms)))
        .max_lifetime(Some(Duration::from_millis(life_ms)))
        .connect(&url)
        .await?;

    // T1-DB-02: Connection Pool Warm-up
    // Pre-establish up to max_connections to avoid cold-start latency on first batch
    if std::env::var("NAME_MATCHER_POOL_WARMUP").ok().map(|v| v != "0").unwrap_or(true) {
        let warm_n = max_conn;
        log::info!("[DB] Warming up connection pool: establishing {} connections", warm_n);
        for _ in 0..warm_n {
            if let Ok(conn) = pool.acquire().await { drop(conn); } else { break; }
        }
        log::info!("[DB] Pool warm-up complete");
    }
    Ok(pool)
}

