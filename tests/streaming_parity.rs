#![cfg(feature = "new_engine")]

use std::collections::BTreeSet;
use anyhow::Result;
use sqlx::MySqlPool;
use name_matcher::matching::{MatchingAlgorithm, StreamingConfig, ProgressUpdate, PartitioningConfig, stream_match_csv_partitioned, MatchPair};
use name_matcher::engine::db_pipeline::db_pipeline::stream_new_engine_partitioned;

fn pairs_set(v: &[MatchPair]) -> BTreeSet<(i64,i64)> {
    v.iter().map(|m| (m.person1.id, m.person2.id)).collect()
}

async fn setup_db() -> Result<MySqlPool> {
    let url = std::env::var("NM_TEST_MYSQL_URL").expect("set NM_TEST_MYSQL_URL=mysql://user:pass@host:3307/db");
    let pool = MySqlPool::connect(&url).await?;
    // Create two small tables with required columns
    sqlx::query("DROP TABLE IF EXISTS nm_test_a").execute(&pool).await.ok();
    sqlx::query("DROP TABLE IF EXISTS nm_test_b").execute(&pool).await.ok();
    sqlx::query("CREATE TABLE nm_test_a (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        uuid VARCHAR(64) NOT NULL,
        first_name VARCHAR(64),
        middle_name VARCHAR(64),
        last_name VARCHAR(64),
        birthdate DATE
    )").execute(&pool).await?;
    sqlx::query("CREATE TABLE nm_test_b (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        uuid VARCHAR(64) NOT NULL,
        first_name VARCHAR(64),
        middle_name VARCHAR(64),
        last_name VARCHAR(64),
        birthdate DATE
    )").execute(&pool).await?;
    // Insert minimal data
    // A: John P Doe 1990-01-01, Jane  Smith 1991-02-03
    sqlx::query("INSERT INTO nm_test_a (uuid, first_name, middle_name, last_name, birthdate) VALUES
        ('u1','John','P','Doe','1990-01-01'),
        ('u2','Jane',NULL,'Smith','1991-02-03'),
        ('u3','Ann',NULL,'Brown','2000-05-06')").execute(&pool).await?;
    // B: john  p. doe 1990-01-01 (matches A1/A2), jane smith 1991-02-03, ann  brown 2000-05-06
    sqlx::query("INSERT INTO nm_test_b (uuid, first_name, middle_name, last_name, birthdate) VALUES
        ('v1','john','p.','doe','1990-01-01'),
        ('v2','JANE',NULL,'SMITH','1991-02-03'),
        ('v3','ann',NULL,'brown','2000-05-06')").execute(&pool).await?;
    Ok(pool)
}

async fn teardown_db(pool: &MySqlPool) {
    let _ = sqlx::query("DROP TABLE IF EXISTS nm_test_a").execute(pool).await;
    let _ = sqlx::query("DROP TABLE IF EXISTS nm_test_b").execute(pool).await;
}

#[tokio::test]
#[ignore]
async fn streaming_parity_direct_algorithms() -> Result<()> {
    let pool = setup_db().await?;
    let t1 = "nm_test_a"; let t2 = "nm_test_b";
    let mut out_legacy_a1: Vec<MatchPair> = Vec::new();
    let mut out_new_a1: Vec<MatchPair> = Vec::new();
    let cfg = StreamingConfig::default();

    // Use partitioned legacy streaming for deterministic small test
    let pc = PartitioningConfig { enabled: true, strategy: "last_initial".into() };
    let count_legacy_a1 = stream_match_csv_partitioned(&pool, t1, t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        |pair| { out_legacy_a1.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;

    let count_new_a1 = stream_new_engine_partitioned(&pool, t1, t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        |pair| { out_new_a1.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;

    assert_eq!(count_legacy_a1, count_new_a1);
    assert_eq!(pairs_set(&out_legacy_a1), pairs_set(&out_new_a1));

    // Algorithm 2
    let mut out_legacy_a2: Vec<MatchPair> = Vec::new();
    let mut out_new_a2: Vec<MatchPair> = Vec::new();
    let count_legacy_a2 = stream_match_csv_partitioned(&pool, t1, t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
        |pair| { out_legacy_a2.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;
    let count_new_a2 = stream_new_engine_partitioned(&pool, t1, t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
        |pair| { out_new_a2.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;

    assert_eq!(count_legacy_a2, count_new_a2);
    assert_eq!(pairs_set(&out_legacy_a2), pairs_set(&out_new_a2));

    teardown_db(&pool).await;
    Ok(())
}



#[tokio::test]
#[ignore]
async fn streaming_parity_fuzzy_algorithms() -> Result<()> {
    let pool = setup_db().await?;
    let t1 = "nm_test_a"; let t2 = "nm_test_b";
    let cfg = StreamingConfig::default();
    let pc = PartitioningConfig { enabled: true, strategy: "last_initial".into() };

    // Fuzzy
    let mut out_legacy_fz: Vec<MatchPair> = Vec::new();
    let mut out_new_fz: Vec<MatchPair> = Vec::new();
    let c_legacy_fz = stream_match_csv_partitioned(&pool, t1, t2, MatchingAlgorithm::Fuzzy,
        |pair| { out_legacy_fz.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;
    let c_new_fz = stream_new_engine_partitioned(&pool, t1, t2, MatchingAlgorithm::Fuzzy,
        |pair| { out_new_fz.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;
    assert_eq!(c_legacy_fz, c_new_fz);
    assert_eq!(pairs_set(&out_legacy_fz), pairs_set(&out_new_fz));

    // FuzzyNoMiddle
    let mut out_legacy_fnm: Vec<MatchPair> = Vec::new();
    let mut out_new_fnm: Vec<MatchPair> = Vec::new();
    let c_legacy_fnm = stream_match_csv_partitioned(&pool, t1, t2, MatchingAlgorithm::FuzzyNoMiddle,
        |pair| { out_legacy_fnm.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;
    let c_new_fnm = stream_new_engine_partitioned(&pool, t1, t2, MatchingAlgorithm::FuzzyNoMiddle,
        |pair| { out_new_fnm.push(pair.clone()); Ok(()) }, cfg.clone(), |_u: ProgressUpdate| {}, None, None, None, pc.clone()).await?;
    assert_eq!(c_legacy_fnm, c_new_fnm);
    assert_eq!(pairs_set(&out_legacy_fnm), pairs_set(&out_new_fnm));

    teardown_db(&pool).await;
    Ok(())
}

#[tokio::test]
#[ignore]
async fn streaming_parity_cross_db_direct() -> Result<()> {
    let url = std::env::var("NM_TEST_MYSQL_URL").expect("set NM_TEST_MYSQL_URL");
    let pool1 = MySqlPool::connect(&url).await?;
    let pool2 = MySqlPool::connect(&url).await?;
    // reuse setup on pool1
    let _ = setup_db().await?; // creates tables on pool bound by env's default DB
    let t1 = "nm_test_a"; let t2 = "nm_test_b";
    let cfg = StreamingConfig::default();

    let mut out_legacy: Vec<MatchPair> = Vec::new();
    let mut out_new: Vec<MatchPair> = Vec::new();
    let c_legacy = name_matcher::matching::stream_match_csv_dual(&pool1, &pool2, t1, t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        |p| { out_legacy.push(p.clone()); Ok(()) }, cfg.clone(), |_u| {}, None).await?;
    let c_new = name_matcher::engine::db_pipeline::db_pipeline::stream_new_engine_dual(&pool1, &pool2, t1, t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        |p| { out_new.push(p.clone()); Ok(()) }, cfg.clone(), |_u| {}, None).await?;
    assert_eq!(c_legacy, c_new);
    assert_eq!(pairs_set(&out_legacy), pairs_set(&out_new));

    teardown_db(&pool1).await;
    Ok(())
}

#[tokio::test]
#[ignore]
async fn streaming_perf_smoke() -> Result<()> {
    let pool = setup_db().await?;
    let t1 = "nm_test_a"; let t2 = "nm_test_b";
    let cfg = StreamingConfig::default();
    let pc = PartitioningConfig { enabled: true, strategy: "last_initial".into() };

    use std::time::Instant;
    let s1 = Instant::now();
    let mut c1 = 0usize; let _ = stream_match_csv_partitioned(&pool, t1, t2, MatchingAlgorithm::Fuzzy,
        |_p| { c1 += 1; Ok(()) }, cfg.clone(), |_u| {}, None, None, None, pc.clone()).await?;
    let d1 = s1.elapsed();
    let s2 = Instant::now();
    let mut c2 = 0usize; let _ = stream_new_engine_partitioned(&pool, t1, t2, MatchingAlgorithm::Fuzzy,
        |_p| { c2 += 1; Ok(()) }, cfg.clone(), |_u| {}, None, None, None, pc.clone()).await?;
    let d2 = s2.elapsed();

    // Allow up to 2.5x slower in CI for trait-based version (heuristic guard)
    assert!((d2.as_secs_f32() / d1.as_secs_f32()) < 2.5, "new engine too slow: {:?} vs {:?}", d2, d1);
    assert_eq!(c1, c2);

    teardown_db(&pool).await;
    Ok(())
}
