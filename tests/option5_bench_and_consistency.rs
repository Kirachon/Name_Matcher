use anyhow::Result;
use name_matcher::db::schema::get_person_rows;
use name_matcher::matching::{match_households_gpu_inmemory, MatchOptions, ComputeBackend, ProgressConfig, MatchingAlgorithm, apply_gpu_enhancements_for_algo};
use sqlx::mysql::MySqlPoolOptions;
use std::time::Instant;

fn parse_thr(s: &str) -> f32 {
    let s = s.trim();
    if let Some(p) = s.strip_suffix('%') { p.parse::<f32>().map(|v| (v/100.0)).unwrap_or(0.95) }
    else if s.contains('.') { s.parse::<f32>().map(|v| if v>1.0 { v/100.0 } else { v }).unwrap_or(0.95) }
    else { s.parse::<f32>().map(|v| (v/100.0)).unwrap_or(0.95) }
}

async fn load_tables() -> Result<(Vec<name_matcher::models::Person>, Vec<name_matcher::models::Person>)> {
    let host = std::env::var("DB_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("DB_PORT").ok().and_then(|s| s.parse::<u16>().ok()).unwrap_or(3307);
    let user = std::env::var("DB_USER").unwrap_or_else(|_| "root".into());
    let pass = std::env::var("DB_PASS").unwrap_or_else(|_| "root".into());
    let db   = std::env::var("DB_NAME").unwrap_or_else(|_| "duplicate_checker".into());
    let url = format!("mysql://{}:{}@{}:{}/{}", user, pass, host, port, db);
    let pool = MySqlPoolOptions::new().max_connections(10).connect(&url).await?;
    let t1 = get_person_rows(&pool, "sample_a").await?;
    let t2 = get_person_rows(&pool, "sample_b").await?;
    Ok((t1,t2))
}

#[tokio::test]
#[ignore]
async fn option5_bench_cpu_vs_gpu() -> Result<()> {
    let (t1, t2) = load_tables().await?;
    assert!(t1.len() > 0 && t2.len() > 0);
    let thr = std::env::var("NAME_MATCHER_HOUSEHOLD_THRESHOLD").unwrap_or_else(|_| "95".to_string());
    let thr = parse_thr(&thr);

    // CPU
    let opts_cpu = MatchOptions { backend: ComputeBackend::Cpu, gpu: None, progress: ProgressConfig::default() };
    let t0 = Instant::now();
    let rows_cpu = match_households_gpu_inmemory(&t1, &t2, opts_cpu, thr, |_u| {});
    let d_cpu = t0.elapsed();

    // GPU (requires build with --features gpu and a CUDA-capable device)
    #[cfg(feature = "gpu")]
    {
        apply_gpu_enhancements_for_algo(MatchingAlgorithm::HouseholdGpu, false, true, false, false);
        let opts_gpu = MatchOptions { backend: ComputeBackend::Gpu, gpu: Some(name_matcher::matching::GpuConfig { device_id: None, mem_budget_mb: 512 }), progress: ProgressConfig::default() };
        let t1i = Instant::now();
        let rows_gpu = match_households_gpu_inmemory(&t1, &t2, opts_gpu, thr, |_u| {});
        let d_gpu = t1i.elapsed();
        println!("[Option5 Bench] CPU: {:?} ({} rows) | GPU: {:?} ({} rows)", d_cpu, rows_cpu.len(), d_gpu, rows_gpu.len());
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("[Option5 Bench] CPU: {:?} ({} rows) | GPU: (feature disabled)", d_cpu, rows_cpu.len());
    }

    Ok(())
}

#[tokio::test]
#[ignore]
async fn option5_cpu_gpu_consistency() -> Result<()> {
    let (t1, t2) = load_tables().await?;
    assert!(t1.len() > 0 && t2.len() > 0);
    let thr = parse_thr(&std::env::var("NAME_MATCHER_HOUSEHOLD_THRESHOLD").unwrap_or_else(|_| "95".to_string()));

    // CPU baseline
    let opts_cpu = MatchOptions { backend: ComputeBackend::Cpu, gpu: None, progress: ProgressConfig::default() };
    let mut rows_cpu = match_households_gpu_inmemory(&t1, &t2, opts_cpu, thr, |_u| {});

    #[cfg(feature = "gpu")]
    {
        apply_gpu_enhancements_for_algo(MatchingAlgorithm::HouseholdGpu, false, true, false, false);
        let opts_gpu = MatchOptions { backend: ComputeBackend::Gpu, gpu: Some(name_matcher::matching::GpuConfig { device_id: None, mem_budget_mb: 512 }), progress: ProgressConfig::default() };
        let mut rows_gpu = match_households_gpu_inmemory(&t1, &t2, opts_gpu, thr, |_u| {});
        rows_cpu.sort_by(|a,b| a.uuid.cmp(&b.uuid).then_with(|| a.hh_id.cmp(&b.hh_id)));
        rows_gpu.sort_by(|a,b| a.uuid.cmp(&b.uuid).then_with(|| a.hh_id.cmp(&b.hh_id)));
        assert_eq!(rows_cpu.len(), rows_gpu.len(), "row count differs");
        for (a,b) in rows_cpu.iter().zip(rows_gpu.iter()) {
            assert_eq!(a.uuid, b.uuid);
            assert_eq!(a.hh_id, b.hh_id);
            assert!((a.match_percentage - b.match_percentage).abs() < 1e-3, "percentage differs");
        }
    }

    Ok(())
}

