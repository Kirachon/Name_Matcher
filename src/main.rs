use anyhow::{Context, Result, bail};
use env_logger::Env;
use log::{error, info, warn};
use std::env;

mod config;
mod db;
mod export;
mod matching;
mod models;
mod normalize;
mod metrics;
mod util;


use crate::config::DatabaseConfig;
use crate::db::{discover_table_columns, get_person_rows, get_person_count, make_pool};
use crate::export::csv_export::{export_to_csv, CsvStreamWriter};
use crate::export::xlsx_export::{export_to_xlsx, SummaryContext, XlsxStreamWriter};
use crate::matching::{match_all_progress, MatchingAlgorithm, ProgressConfig, ProgressUpdate, StreamingConfig, stream_match_csv_partitioned, PartitioningConfig, stream_match_csv_dual};
use crate::util::envfile::{load_dotenv_if_present, write_env_template, parse_env_file};

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    if let Err(e) = run().await {
        error!("{:#}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    load_dotenv_if_present()?;
    let env_map = parse_env_file().unwrap_or_default();
    let args: Vec<String> = env::args().collect();

    // Utility subcommand: generate .env.template
    if args.get(1).map(|s| s.as_str()) == Some("env-template") {
        let path = args.get(2).cloned().unwrap_or_else(|| ".env.template".to_string());
        write_env_template(&path)?;
        println!("Wrote {}. Copy to .env and edit values as needed.", path);
        return Ok(());
    }

    if args.len() < 10 && !(std::env::var("DB_HOST").is_ok() && std::env::var("DB_PORT").is_ok() && std::env::var("DB_USER").is_ok() && std::env::var("DB_PASSWORD").is_ok() && std::env::var("DB_NAME").is_ok()) {
        eprintln!("Usage: {} <host> <port> <user> <password> <database> <table1> <table2> <algo:1|2|3> <out_path> [format: csv|xlsx|both] [--gpu-hash-join] [--gpu-fuzzy-direct-hash] [--direct-fuzzy-normalization]", args.get(0).map(String::as_str).unwrap_or("name_matcher"));
        eprintln!("       {} env-template [path]   # generate a .env.template", args.get(0).map(String::as_str).unwrap_or("name_matcher"));
        eprintln!("Notes:");
        eprintln!("  --gpu-hash-join                  Enable GPU-accelerated hash join prefilter for Algorithms 1/2 (falls back to CPU if unavailable)");
        eprintln!("  NAME_MATCHER_GPU_HASH_JOIN=1     can also enable this feature");
        eprintln!("  --gpu-fuzzy-direct-hash          GPU hash pre-pass for Fuzzy's direct phase (candidate filter only; behavior preserved)");
        eprintln!("  NAME_MATCHER_GPU_FUZZY_DIRECT_HASH=1 to enable the above");
        eprintln!("  --direct-fuzzy-normalization     Apply Fuzzy-style normalization to Algorithms 1 & 2 before equality checks");
        eprintln!("  NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION=1 to enable the above");
        eprintln!("Examples:");
        eprintln!("  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.csv --gpu-hash-join", args.get(0).unwrap_or(&"name_matcher".to_string()));
        eprintln!("  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.xlsx xlsx", args.get(0).unwrap_or(&"name_matcher".to_string()));
        eprintln!("  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches both", args.get(0).unwrap_or(&"name_matcher".to_string()));
        std::process::exit(2);
    }

    // Prefer environment variables if provided, else fallback to CLI args
    let host = env_map.get("DB_HOST").cloned().or_else(|| std::env::var("DB_HOST").ok()).unwrap_or_else(|| args.get(1).cloned().unwrap_or_default());
    let port: u16 = env_map.get("DB_PORT").and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("DB_PORT").ok().and_then(|s| s.parse().ok()))
        .or_else(|| args.get(2).and_then(|s| s.parse().ok()))
        .context("Invalid port")?;
    let user = env_map.get("DB_USER").cloned().or_else(|| std::env::var("DB_USER").ok()).unwrap_or_else(|| args.get(3).cloned().unwrap_or_default());
    let pass = env_map.get("DB_PASSWORD").cloned().or_else(|| std::env::var("DB_PASSWORD").ok()).unwrap_or_else(|| args.get(4).cloned().unwrap_or_default());
    let dbname = env_map.get("DB_NAME").cloned().or_else(|| std::env::var("DB_NAME").ok()).unwrap_or_else(|| args.get(5).cloned().unwrap_or_default());

    let cfg = DatabaseConfig { host, port, username: user, password: pass, database: dbname };
    let table1 = args.get(6).cloned().or_else(|| env_map.get("TABLE1").cloned()).unwrap_or_else(|| std::env::var("TABLE1").unwrap_or_else(|_| "table1".into()));
    let table2 = args.get(7).cloned().or_else(|| env_map.get("TABLE2").cloned()).unwrap_or_else(|| std::env::var("TABLE2").unwrap_or_else(|_| "table2".into()));
    let algo_num: u8 = args.get(8).and_then(|s| s.parse().ok()).or_else(|| env_map.get("ALGO").and_then(|s| s.parse().ok())).or_else(|| std::env::var("ALGO").ok().and_then(|s| s.parse().ok())).unwrap_or(1);
    let out_path = args.get(9).cloned().or_else(|| env_map.get("OUT_PATH").cloned()).unwrap_or_else(|| std::env::var("OUT_PATH").unwrap_or_else(|_| "matches.csv".into()));
    let format = args.get(10).map(|s| s.to_ascii_lowercase()).unwrap_or_else(|| "csv".to_string());


    let algorithm = match algo_num {
        1 => MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        2 => MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
        3 => MatchingAlgorithm::Fuzzy,
        4 => MatchingAlgorithm::FuzzyNoMiddle,
        _ => {
            eprintln!("algo must be 1, 2, 3 or 4");
            std::process::exit(2);
        }
    };

    info!("Connecting to MySQL at {}:{} / db {}", cfg.host, cfg.port, cfg.database);
    let cfg2_opt = {
        let host2 = std::env::var("DB2_HOST").ok();
        if let Some(h2) = host2 {
            let port2 = std::env::var("DB2_PORT").ok().and_then(|s| s.parse::<u16>().ok()).unwrap_or(cfg.port);
            let user2 = std::env::var("DB2_USER").ok().unwrap_or_else(|| cfg.username.clone());
            let pass2 = std::env::var("DB2_PASS").ok().unwrap_or_else(|| cfg.password.clone());
            let db2 = std::env::var("DB2_DATABASE").ok().unwrap_or_else(|| cfg.database.clone());
            Some(DatabaseConfig { host: h2, port: port2, username: user2, password: pass2, database: db2 })
        } else { None }
    };
    let pool1 = make_pool(&cfg).await?;
    let (pool2_opt, db_label) = if let Some(cfg2) = &cfg2_opt {
        info!("Connecting to second MySQL at {}:{} / db {}", cfg2.host, cfg2.port, cfg2.database);
        let p2 = make_pool(cfg2).await?;
        (Some(p2), format!("{} | {}", cfg.database, cfg2.database))
    } else {
        (None, cfg.database.clone())
    };


    if matches!(algorithm, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) && format != "csv" {
        eprintln!("Fuzzy algorithm supports CSV format only. Use format=csv.");
        std::process::exit(2);
    }
    let db2_name = cfg2_opt.as_ref().map(|c| c.database.clone()).unwrap_or_else(|| cfg.database.clone());


    // Validate schemas
    let cols1 = discover_table_columns(&pool1, &cfg.database, &table1).await?;
    let cols2 = discover_table_columns(pool2_opt.as_ref().unwrap_or(&pool1), &db2_name, &table2).await?;
    cols1.validate_basic()?;
    if !(cols2.has_id && cols2.has_first_name && cols2.has_last_name && cols2.has_birthdate) {
        bail!("Table {} missing required columns: requires id, first_name, last_name, birthdate (uuid optional)", table2);
    }
    info!("{} columns: {:?}", table1, cols1);
    info!("{} columns: {:?}", table2, cols2);
    if std::env::var("CHECK_ONLY").is_ok() {
        info!("Schema OK; exiting due to CHECK_ONLY");
        return Ok(())
    }

    // Decide execution mode (streaming vs in-memory)
    let c1 = get_person_count(&pool1, &table1).await?;
    let c2 = get_person_count(pool2_opt.as_ref().unwrap_or(&pool1), &table2).await?;
    let streaming_env = std::env::var("NAME_MATCHER_STREAMING").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true") ).unwrap_or(false);
    let big_data_heuristic = (c1 + c2) > 200_000; // heuristic threshold
    let part_strategy = std::env::var("NAME_MATCHER_PARTITION").unwrap_or_else(|_| "last_initial".to_string());
    // Command-line flags can be placed after required args; scan whole argv
    let gpu_hash_flag = args.iter().any(|a| a == "--gpu-hash-join");
let gpu_fuzzy_direct_flag = args.iter().any(|a| a == "--gpu-fuzzy-direct-hash");
let direct_norm_flag = args.iter().any(|a| a == "--direct-fuzzy-normalization");

    let use_streaming = streaming_env || big_data_heuristic;

    use crate::metrics::memory_stats_mb;
    let cfgp = ProgressConfig { update_every: 1000, ..Default::default() };

    if use_streaming {
    let gpu_hash_env = std::env::var("NAME_MATCHER_GPU_HASH_JOIN").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    let gpu_fuzzy_direct_env = std::env::var("NAME_MATCHER_GPU_FUZZY_DIRECT_HASH").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    let direct_norm_env = std::env::var("NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);

        info!("Streaming mode enabled ({} + {} rows).", c1, c2);
        let mut s_cfg = StreamingConfig::default();
        s_cfg.checkpoint_path = Some(format!("{}.nmckpt", out_path));
        s_cfg.use_gpu_hash_join = if matches!(algorithm, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) { false } else { gpu_hash_env || gpu_hash_flag }; // legacy master switch
        // Granular GPU controls default to the legacy switch for backward compatibility
        s_cfg.use_gpu_build_hash = s_cfg.use_gpu_hash_join;
        s_cfg.use_gpu_probe_hash = s_cfg.use_gpu_hash_join;
        s_cfg.use_gpu_fuzzy_direct_hash = gpu_fuzzy_direct_env || gpu_fuzzy_direct_flag;
        s_cfg.direct_use_fuzzy_normalization = direct_norm_env || direct_norm_flag;

        // Apply normalization alignment globally for this run (affects in-memory too if used)
        crate::matching::set_direct_normalization_fuzzy(s_cfg.direct_use_fuzzy_normalization);

        if format == "csv" || format == "both" {
            info!("Streaming CSV export to {} using {:?}", out_path, algorithm);
            // Read fuzzy threshold from env (supports 0.95, 95, or 95%)
            let fuzzy_thr_env = std::env::var("NAME_MATCHER_FUZZY_THRESHOLD").unwrap_or_else(|_| "95".to_string());
            let fuzzy_min_conf = (|| {
                let s = fuzzy_thr_env.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<f32>().ok().map(|v| (v/100.0).clamp(0.6, 1.0))
                } else if s.contains('.') {
                    s.parse::<f32>().ok().map(|v| if v > 1.0 { (v/100.0).clamp(0.6, 1.0) } else { v.clamp(0.6, 1.0) })
                } else {
                    s.parse::<f32>().ok().map(|v| (v/100.0).clamp(0.6, 1.0))
                }
            })().unwrap_or(0.95);
            let mut writer = CsvStreamWriter::create(&out_path, algorithm, fuzzy_min_conf)?;
            let t_match = std::time::Instant::now();
            let flush_every = s_cfg.flush_every;
            let mut n_flushed = 0usize;
            if let Some(p2) = &pool2_opt {
                if s_cfg.use_gpu_hash_join { warn!("GPU hash-join requested but cross-database GPU path is not yet available; proceeding with CPU for dual-db"); }
                let count = stream_match_csv_dual(&pool1, p2, &table1, &table2, algorithm, |pair| {
                    writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(())
                }, s_cfg.clone(), |u: ProgressUpdate| {
                    info!("[dual] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                        u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                }, None).await?;
                writer.flush()?;
                info!("Wrote {} matches (streaming, dual-db) in {:?}", count, t_match.elapsed());
            } else {
                // If GPU hash-join is enabled and single DB, use the GPU-accelerated streaming path
                if s_cfg.use_gpu_hash_join {
                    info!("GPU hash-join requested; using single-DB accelerated path");
                    let count = crate::matching::stream_match_csv(&pool1, &table1, &table2, algorithm, |pair| {
                        writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(())
                    }, s_cfg.clone(), |u: ProgressUpdate| {
                        info!("[gpu] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                            u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                    }, None).await?;
                    writer.flush()?;
                    info!("Wrote {} matches (streaming, gpu-accelerated) in {:?}", count, t_match.elapsed());
                } else {
                    let pc = PartitioningConfig { enabled: true, strategy: part_strategy.clone() };
                    let count = stream_match_csv_partitioned(&pool1, &table1, &table2, algorithm, |pair| { writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(()) }, s_cfg.clone(), |u: ProgressUpdate| {
                        info!("[part] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                            u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                    }, None, None, None, pc).await?;
                    writer.flush()?;
                    info!("Wrote {} matches (streaming) in {:?}", count, t_match.elapsed());
                }
            }
        }
        if format == "xlsx" || format == "both" {
            // Streaming both algorithms into an XLSX workbook
            let xlsx_path = if out_path.to_ascii_lowercase().ends_with(".xlsx") { out_path.clone() } else { out_path.replace(".csv", ".xlsx") };
            let mem_start = memory_stats_mb().used_mb;
            let mut xw = XlsxStreamWriter::create(&xlsx_path)?;
            let t1 = std::time::Instant::now();
            let mut algo1_count = 0usize;
            if let Some(p2) = &pool2_opt {
                stream_match_csv_dual(&pool1, p2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |pair| { algo1_count += 1; xw.append_algo1(pair) }, s_cfg.clone(), |_u| {}, None).await?;
            } else {
                let pc = PartitioningConfig { enabled: true, strategy: part_strategy.clone() };
                stream_match_csv_partitioned(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |pair| { algo1_count += 1; xw.append_algo1(pair) }, s_cfg.clone(), |_u| {}, None, None, None, pc.clone()).await?;
            }
            let took_a1 = t1.elapsed();
            let t2 = std::time::Instant::now();
            let mut algo2_count = 0usize;
            if let Some(p2) = &pool2_opt {
                stream_match_csv_dual(&pool1, p2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |pair| { algo2_count += 1; xw.append_algo2(pair) }, s_cfg.clone(), |_u| {}, None).await?;
            } else {
                let pc = PartitioningConfig { enabled: true, strategy: part_strategy.clone() };
                stream_match_csv_partitioned(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |pair| { algo2_count += 1; xw.append_algo2(pair) }, s_cfg.clone(), |_u| {}, None, None, None, pc).await?;
            }
            let took_a2 = t2.elapsed();
            let mem_end = memory_stats_mb().used_mb;
            let summary = SummaryContext {
                db_name: db_label.clone(),
                table1: table1.clone(),
                table2: table2.clone(),
                total_table1: c1 as usize,
                total_table2: c2 as usize,
                matches_algo1: algo1_count,
                matches_algo2: algo2_count,
                matches_fuzzy: 0,
                overlap_count: 0, // not tracked in streaming mode to save memory
                unique_algo1: algo1_count,
                unique_algo2: algo2_count,
                fetch_time: std::time::Duration::from_secs(0),
                match1_time: took_a1,
                match2_time: took_a2,
                export_time: std::time::Duration::from_secs(0),
                mem_used_start_mb: mem_start,
                mem_used_end_mb: mem_end,
                started_utc: chrono::Utc::now(),
                ended_utc: chrono::Utc::now(),

                duration_secs: 0.0,
                algo_used: "Both (1,2)".into(),
                gpu_used: false,
                gpu_total_mb: 0,
                gpu_free_mb_end: 0,
            };
            xw.finalize(&summary)?;
            info!("XLSX written (streaming) to {} | a1={} a2={}", xlsx_path, algo1_count, algo2_count);
            // Also write standalone summary CSV/XLSX next to streaming results
            let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
            let base_dir = std::path::Path::new(&xlsx_path).parent().unwrap_or(std::path::Path::new("."));
            let sum_csv = base_dir.join(format!("summary_report_{}.csv", ts));
            let sum_xlsx = base_dir.join(format!("summary_report_{}.xlsx", ts));
            let _ = crate::export::csv_export::export_summary_csv(sum_csv.to_string_lossy().as_ref(), &summary);
            let _ = crate::export::xlsx_export::export_summary_xlsx(sum_xlsx.to_string_lossy().as_ref(), &summary);

        }
    } else {
        // Honor normalization alignment option in in-memory mode as well
        let direct_norm_env = std::env::var("NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
        crate::matching::set_direct_normalization_fuzzy(direct_norm_env || direct_norm_flag);
        // Apply GPU fuzzy direct pre-pass toggle in in-memory mode too (affects algo 3/4 via match_all_with_opts)
        let gpu_fuzzy_direct_env = std::env::var("NAME_MATCHER_GPU_FUZZY_DIRECT_HASH").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
        crate::matching::set_gpu_fuzzy_direct_prep(gpu_fuzzy_direct_env || gpu_fuzzy_direct_flag);

        // In-memory fallback (previous behavior)
        info!("Fetching rows from {} and {}", table1, table2);
        let t_fetch = std::time::Instant::now();
        let people1 = get_person_rows(&pool1, &table1).await?;

        let people2 = get_person_rows(pool2_opt.as_ref().unwrap_or(&pool1), &table2).await?;
        let took_fetch = t_fetch.elapsed();
        if took_fetch.as_secs() >= 30 { info!("Fetching took {:?}", took_fetch); }

        // Always compute the requested algorithm (for CSV/back-compat)
        info!("Matching using {:?}", algorithm);
        let start = std::time::Instant::now();
        let matches_requested = match_all_progress(&people1, &people2, algorithm, cfgp, |u: ProgressUpdate| {
            info!("Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
        });
        let took_requested = start.elapsed();
        if took_requested.as_secs() >= 30 { info!("Matching stage took {:?}", took_requested); }

        // CSV export if requested or both
        if format == "csv" || format == "both" {
            info!("Exporting {} match rows to {}", matches_requested.len(), out_path);
            let t_export = std::time::Instant::now();
            let fuzzy_thr_env = std::env::var("NAME_MATCHER_FUZZY_THRESHOLD").unwrap_or_else(|_| "95".to_string());
            let fuzzy_min_conf = (|| {
                let s = fuzzy_thr_env.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<f32>().ok().map(|v| (v/100.0).clamp(0.6, 1.0))
                } else if s.contains('.') {
                    s.parse::<f32>().ok().map(|v| if v > 1.0 { (v/100.0).clamp(0.6, 1.0) } else { v.clamp(0.6, 1.0) })
                } else {
                    s.parse::<f32>().ok().map(|v| (v/100.0).clamp(0.6, 1.0))
                }
            })().unwrap_or(0.95);
            export_to_csv(&matches_requested, &out_path, algorithm, fuzzy_min_conf)?;
            let took_export = t_export.elapsed();
            if took_export.as_secs() >= 30 { info!("Export took {:?}", took_export); }
        }

        // XLSX export implies computing both algorithms and writing 3 sheets
        if format == "xlsx" || format == "both" {
            info!("Preparing XLSX export (both algorithms) ...");
            let mem_start = memory_stats_mb().used_mb;

            // Algo 1
            let t1 = std::time::Instant::now();
            let a1 = match_all_progress(&people1, &people2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, cfgp, |_u| {});
            let took_a1 = t1.elapsed();

            // Algo 2
            let t2 = std::time::Instant::now();
            let a2 = match_all_progress(&people1, &people2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, cfgp, |_u| {});
            let took_a2 = t2.elapsed();

            // Overlap/unique counts by (id1,id2)
            use std::collections::HashSet;
            let set1: HashSet<(i64,i64)> = a1.iter().map(|m| (m.person1.id, m.person2.id)).collect();
            let set2: HashSet<(i64,i64)> = a2.iter().map(|m| (m.person1.id, m.person2.id)).collect();
            let overlap = set1.intersection(&set2).count();
            let unique1 = set1.len().saturating_sub(overlap);
            let unique2 = set2.len().saturating_sub(overlap);

            let t_export = std::time::Instant::now();
            let mem_end = memory_stats_mb().used_mb;
            let summary = SummaryContext {
                db_name: db_label.clone(),
                table1: table1.clone(),
                table2: table2.clone(),
                total_table1: people1.len(),
                total_table2: people2.len(),
                matches_algo1: a1.len(),
                matches_algo2: a2.len(),
                matches_fuzzy: 0,
                overlap_count: overlap,
                unique_algo1: unique1,
                unique_algo2: unique2,
                fetch_time: took_fetch,
                match1_time: took_a1,
                match2_time: took_a2,
                export_time: std::time::Duration::from_secs(0),
                mem_used_start_mb: mem_start,
                mem_used_end_mb: mem_end,
                started_utc: chrono::Utc::now(),
                ended_utc: chrono::Utc::now(),
                duration_secs: (took_fetch + took_a1 + took_a2).as_secs_f64(),
                algo_used: "Both (1,2)".into(),
                gpu_used: false,
                gpu_total_mb: 0,
                gpu_free_mb_end: 0,
            };

            // Decide xlsx path
            let xlsx_path = if out_path.to_ascii_lowercase().ends_with(".xlsx") {
                out_path.clone()
            } else if out_path.to_ascii_lowercase().ends_with(".csv") {
                out_path.replace(".csv", ".xlsx")
            } else { out_path.clone() + ".xlsx" };

            export_to_xlsx(&a1, &a2, &xlsx_path, &summary)?;
            let took_export = t_export.elapsed();
            if took_export.as_secs() >= 30 { info!("XLSX export took {:?}", took_export); }
            info!("XLSX written to {}", xlsx_path);
            // Also write standalone summary CSV/XLSX next to results
            let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
            let base_dir = std::path::Path::new(&xlsx_path).parent().unwrap_or(std::path::Path::new("."));
            let sum_csv = base_dir.join(format!("summary_report_{}.csv", ts));
            let sum_xlsx = base_dir.join(format!("summary_report_{}.xlsx", ts));
            let _ = crate::export::csv_export::export_summary_csv(sum_csv.to_string_lossy().as_ref(), &summary);
            let _ = crate::export::xlsx_export::export_summary_xlsx(sum_xlsx.to_string_lossy().as_ref(), &summary);

        }
    }

    info!("Done.");
    Ok(())
}
