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
    let max_conn = max
        .or_else(|| std::env::var("NAME_MATCHER_POOL_SIZE").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(16);
    let min_conn: u32 = std::env::var("NAME_MATCHER_POOL_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let acquire_ms: u64 = std::env::var("NAME_MATCHER_ACQUIRE_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(30_000);
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
    Ok(pool)
}

