use crate::config::DatabaseConfig;
use anyhow::Result;
use sqlx::mysql::MySqlPoolOptions;
use sqlx::MySqlPool;

pub async fn make_pool(cfg: &DatabaseConfig) -> Result<MySqlPool> {
    make_pool_with_size(cfg, None).await
}

pub async fn make_pool_with_size(cfg: &DatabaseConfig, max: Option<u32>) -> Result<MySqlPool> {
    let url = cfg.to_url();
    let max_conn = max
        .or_else(|| std::env::var("NAME_MATCHER_POOL_SIZE").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(10);
    let pool = MySqlPoolOptions::new()
        .max_connections(max_conn)
        .connect(&url)
        .await?;
    Ok(pool)
}

