/// Comprehensive benchmark harness for Name_Matcher performance validation
/// Measures end-to-end runtime, throughput, memory usage, and GPU utilization
/// Supports all 6 matching algorithms in both in-memory and streaming modes

use anyhow::{Result, Context};
use chrono::Utc;
use name_matcher::config::DatabaseConfig;
use name_matcher::db::{connection::make_pool, get_person_rows, get_person_count};
use name_matcher::matching::{
    MatchingAlgorithm, MatchOptions, MatchPair, ProgressConfig, ProgressUpdate,
    ComputeBackend, GpuConfig, StreamingConfig, match_all_with_opts, match_all_progress,
    stream_match_gpu_hash_join,
};
use name_matcher::metrics::memory_stats_mb;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args)?;
    
    log::info!("=== Name_Matcher Performance Benchmark ===");
    log::info!("Configuration: {:?}", config);
    
    // Connect to database
    let db_config = DatabaseConfig {
        host: config.host.clone(),
        port: config.port,
        username: config.username.clone(),
        password: config.password.clone(),
        database: config.database.clone(),
    };
    
    let pool = make_pool(&db_config).await
        .context("Failed to connect to database")?;
    
    // Verify tables exist
    let count_a = get_person_count(&pool, &config.table_a).await?;
    let count_b = get_person_count(&pool, &config.table_b).await?;
    log::info!("Table {} has {} records", config.table_a, count_a);
    log::info!("Table {} has {} records", config.table_b, count_b);
    
    // Run benchmarks
    let mut results = Vec::new();
    
    for algo in &config.algorithms {
        for mode in &config.modes {
            for use_gpu in &config.gpu_modes {
                let bench_config = BenchmarkConfig {
                    algorithm: *algo,
                    mode: mode.clone(),
                    use_gpu: *use_gpu,
                    table_a: config.table_a.clone(),
                    table_b: config.table_b.clone(),
                    runs: config.runs,
                };
                
                log::info!("Running benchmark: {:?}", bench_config);
                
                let result = run_benchmark(&pool, &bench_config).await?;
                results.push(result);
                
                // Brief pause between benchmarks
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
    
    // Save results
    save_results(&results, &config.output)?;
    
    // Print summary
    print_summary(&results);
    
    log::info!("=== Benchmark complete ===");
    Ok(())
}

#[derive(Debug, Clone)]
struct Config {
    host: String,
    port: u16,
    username: String,
    password: String,
    database: String,
    table_a: String,
    table_b: String,
    algorithms: Vec<MatchingAlgorithm>,
    modes: Vec<String>,
    gpu_modes: Vec<bool>,
    runs: usize,
    output: String,
}

fn parse_args(args: &[String]) -> Result<Config> {
    let host = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1".into());
    let port = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3307);
    let username = args.get(3).cloned().unwrap_or_else(|| "root".into());
    let password = args.get(4).cloned().unwrap_or_else(|| "root".into());
    let database = args.get(5).cloned().unwrap_or_else(|| "benchmark_nm".into());
    let table_a = args.get(6).cloned().unwrap_or_else(|| "clean_a".into());
    let table_b = args.get(7).cloned().unwrap_or_else(|| "clean_b".into());
    let algo_str = args.get(8).cloned().unwrap_or_else(|| "1,2,3,4".into());
    let mode_str = args.get(9).cloned().unwrap_or_else(|| "memory,streaming".into());
    let runs = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(3);
    let output = args.get(11).cloned().unwrap_or_else(|| "benchmark_results.json".into());
    
    // Parse algorithms
    let algorithms: Vec<MatchingAlgorithm> = algo_str.split(',')
        .filter_map(|s| match s.trim() {
            "1" => Some(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd),
            "2" => Some(MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd),
            "3" => Some(MatchingAlgorithm::Fuzzy),
            "4" => Some(MatchingAlgorithm::FuzzyNoMiddle),
            "5" => Some(MatchingAlgorithm::HouseholdGpu),
            "6" => Some(MatchingAlgorithm::HouseholdGpuOpt6),
            _ => None,
        })
        .collect();
    
    // Parse modes
    let modes: Vec<String> = mode_str.split(',')
        .map(|s| s.trim().to_string())
        .collect();
    
    // GPU modes: test both CPU and GPU if GPU feature enabled
    #[cfg(feature = "gpu")]
    let gpu_modes = vec![false, true];
    #[cfg(not(feature = "gpu"))]
    let gpu_modes = vec![false];
    
    Ok(Config {
        host, port, username, password, database,
        table_a, table_b, algorithms, modes, gpu_modes, runs, output,
    })
}

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    algorithm: MatchingAlgorithm,
    mode: String,
    use_gpu: bool,
    table_a: String,
    table_b: String,
    runs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    timestamp: String,
    algorithm: String,
    mode: String,
    use_gpu: bool,
    table_a: String,
    table_b: String,
    table_a_count: i64,
    table_b_count: i64,
    runs: Vec<RunResult>,
    mean_runtime_secs: f64,
    median_runtime_secs: f64,
    std_dev_runtime_secs: f64,
    mean_throughput_rps: f64,
    mean_peak_memory_mb: f64,
    mean_matches_found: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunResult {
    run_number: usize,
    runtime_secs: f64,
    throughput_rps: f64,
    peak_memory_mb: u64,
    avg_memory_mb: u64,
    matches_found: usize,
    progress_updates: usize,
}

async fn run_benchmark(pool: &sqlx::MySqlPool, config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    let count_a = get_person_count(pool, &config.table_a).await?;
    let count_b = get_person_count(pool, &config.table_b).await?;
    
    let mut runs = Vec::new();
    
    for run in 1..=config.runs {
        log::info!("  Run {}/{}", run, config.runs);
        
        let run_result = if config.mode == "memory" {
            run_in_memory_benchmark(pool, config).await?
        } else {
            run_streaming_benchmark(pool, config).await?
        };
        
        runs.push(run_result);
        
        // Force garbage collection between runs
        std::thread::sleep(Duration::from_millis(500));
    }
    
    // Calculate statistics
    let runtimes: Vec<f64> = runs.iter().map(|r| r.runtime_secs).collect();
    let mean_runtime = runtimes.iter().sum::<f64>() / runtimes.len() as f64;
    let mut sorted_runtimes = runtimes.clone();
    sorted_runtimes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_runtime = sorted_runtimes[sorted_runtimes.len() / 2];
    let variance = runtimes.iter().map(|r| (r - mean_runtime).powi(2)).sum::<f64>() / runtimes.len() as f64;
    let std_dev = variance.sqrt();
    
    let mean_throughput = runs.iter().map(|r| r.throughput_rps).sum::<f64>() / runs.len() as f64;
    let mean_peak_memory = runs.iter().map(|r| r.peak_memory_mb).sum::<u64>() / runs.len() as u64;
    let mean_matches = runs.iter().map(|r| r.matches_found).sum::<usize>() / runs.len();
    
    Ok(BenchmarkResult {
        timestamp: Utc::now().to_rfc3339(),
        algorithm: format!("{:?}", config.algorithm),
        mode: config.mode.clone(),
        use_gpu: config.use_gpu,
        table_a: config.table_a.clone(),
        table_b: config.table_b.clone(),
        table_a_count: count_a,
        table_b_count: count_b,
        runs,
        mean_runtime_secs: mean_runtime,
        median_runtime_secs: median_runtime,
        std_dev_runtime_secs: std_dev,
        mean_throughput_rps: mean_throughput,
        mean_peak_memory_mb: mean_peak_memory as f64,
        mean_matches_found: mean_matches,
    })
}

async fn run_in_memory_benchmark(pool: &sqlx::MySqlPool, config: &BenchmarkConfig) -> Result<RunResult> {
    // Load data
    let t1 = get_person_rows(pool, &config.table_a).await?;
    let t2 = get_person_rows(pool, &config.table_b).await?;
    
    let total_records = t1.len() + t2.len();
    
    // Track memory and progress
    let peak_memory = Arc::new(Mutex::new(0u64));
    let total_memory = Arc::new(Mutex::new(0u64));
    let update_count = Arc::new(Mutex::new(0usize));
    
    let peak_mem_clone = peak_memory.clone();
    let total_mem_clone = total_memory.clone();
    let update_count_clone = update_count.clone();
    
    let progress_callback = move |_u: ProgressUpdate| {
        let mem = memory_stats_mb();
        let used = mem.used_mb;
        
        let mut peak = peak_mem_clone.lock().unwrap();
        if used > *peak {
            *peak = used;
        }
        
        *total_mem_clone.lock().unwrap() += used;
        *update_count_clone.lock().unwrap() += 1;
    };
    
    // Run matching
    let start = Instant::now();
    
    let matches = if matches!(config.algorithm, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
        let opts = MatchOptions {
            backend: if config.use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu },
            gpu: Some(GpuConfig { device_id: None, mem_budget_mb: 512 }),
            progress: ProgressConfig::default(),
        };
        match_all_with_opts(&t1, &t2, config.algorithm, opts, progress_callback)
    } else {
        match_all_progress(&t1, &t2, config.algorithm, ProgressConfig::default(), progress_callback)
    };
    
    let runtime = start.elapsed();
    
    let peak_mem = *peak_memory.lock().unwrap();
    let updates = *update_count.lock().unwrap();
    let avg_mem = if updates > 0 { *total_memory.lock().unwrap() / updates as u64 } else { 0 };
    
    Ok(RunResult {
        run_number: 0,
        runtime_secs: runtime.as_secs_f64(),
        throughput_rps: total_records as f64 / runtime.as_secs_f64(),
        peak_memory_mb: peak_mem,
        avg_memory_mb: avg_mem,
        matches_found: matches.len(),
        progress_updates: updates,
    })
}

async fn run_streaming_benchmark(pool: &sqlx::MySqlPool, config: &BenchmarkConfig) -> Result<RunResult> {
    let count_a = get_person_count(pool, &config.table_a).await?;
    let count_b = get_person_count(pool, &config.table_b).await?;
    let total_records = (count_a + count_b) as usize;
    
    // Track metrics
    let peak_memory = Arc::new(Mutex::new(0u64));
    let total_memory = Arc::new(Mutex::new(0u64));
    let update_count = Arc::new(Mutex::new(0usize));
    let match_count = Arc::new(Mutex::new(0usize));
    
    let peak_mem_clone = peak_memory.clone();
    let total_mem_clone = total_memory.clone();
    let update_count_clone = update_count.clone();
    
    let progress_callback = move |_u: ProgressUpdate| {
        let mem = memory_stats_mb();
        let used = mem.used_mb;
        
        let mut peak = peak_mem_clone.lock().unwrap();
        if used > *peak {
            *peak = used;
        }
        
        *total_mem_clone.lock().unwrap() += used;
        *update_count_clone.lock().unwrap() += 1;
    };
    
    let match_count_clone = match_count.clone();
    let mut on_match = move |_pair: &MatchPair| -> Result<()> {
        *match_count_clone.lock().unwrap() += 1;
        Ok(())
    };
    
    // Configure streaming
    let mut cfg = StreamingConfig::default();
    cfg.batch_size = 50_000;
    cfg.use_gpu_hash_join = config.use_gpu;
    cfg.use_gpu_build_hash = config.use_gpu;
    cfg.use_gpu_probe_hash = config.use_gpu;
    
    // Run streaming match
    let start = Instant::now();
    
    let _written = stream_match_gpu_hash_join(
        pool,
        &config.table_a,
        &config.table_b,
        config.algorithm,
        &mut on_match,
        cfg,
        progress_callback,
        None,
    ).await?;
    
    let runtime = start.elapsed();
    
    let peak_mem = *peak_memory.lock().unwrap();
    let updates = *update_count.lock().unwrap();
    let avg_mem = if updates > 0 { *total_memory.lock().unwrap() / updates as u64 } else { 0 };
    let matches = *match_count.lock().unwrap();
    
    Ok(RunResult {
        run_number: 0,
        runtime_secs: runtime.as_secs_f64(),
        throughput_rps: total_records as f64 / runtime.as_secs_f64(),
        peak_memory_mb: peak_mem,
        avg_memory_mb: avg_mem,
        matches_found: matches,
        progress_updates: updates,
    })
}

fn save_results(results: &[BenchmarkResult], output_path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    let mut file = File::create(output_path)?;
    file.write_all(json.as_bytes())?;
    log::info!("Results saved to {}", output_path);
    Ok(())
}

fn print_summary(results: &[BenchmarkResult]) {
    println!("\n=== Benchmark Summary ===\n");
    println!("{:<20} {:<12} {:<8} {:<12} {:<12} {:<12} {:<10}",
        "Algorithm", "Mode", "GPU", "Runtime(s)", "Throughput", "Peak Mem", "Matches");
    println!("{}", "-".repeat(100));
    
    for r in results {
        println!("{:<20} {:<12} {:<8} {:<12.2} {:<12.0} {:<12.0} {:<10}",
            r.algorithm,
            r.mode,
            if r.use_gpu { "Yes" } else { "No" },
            r.mean_runtime_secs,
            r.mean_throughput_rps,
            r.mean_peak_memory_mb,
            r.mean_matches_found
        );
    }
    println!();
}

