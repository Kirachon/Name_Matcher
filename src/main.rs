use anyhow::{Context, Result};
use env_logger::Env;
use log::{error, info};
use std::env;

mod config;
mod db;
mod export;
mod matching;
mod models;
mod normalize;
mod metrics;

use crate::config::DatabaseConfig;
use crate::db::{discover_table_columns, get_person_rows, get_person_count, make_pool};
use crate::export::csv_export::{export_to_csv, CsvStreamWriter};
use crate::export::xlsx_export::{export_to_xlsx, SummaryContext, XlsxStreamWriter};
use crate::matching::{match_all_progress, MatchingAlgorithm, ProgressConfig, ProgressUpdate, stream_match_csv, StreamingConfig};

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    if let Err(e) = run().await {
        error!("{:#}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 10 {
        eprintln!("Usage: {} <host> <port> <user> <password> <database> <table1> <table2> <algo:1|2|3> <out_path> [format: csv|xlsx|both]", args.get(0).map(String::as_str).unwrap_or("name_matcher"));
        eprintln!("Examples:");
        eprintln!("  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.csv", args[0]);
        eprintln!("  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.xlsx xlsx", args[0]);
        eprintln!("  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches both", args[0]);
        std::process::exit(2);
    }

    let cfg = DatabaseConfig {
        host: args[1].clone(),
        port: args[2].parse().context("Invalid port")?,
        username: args[3].clone(),
        password: args[4].clone(),
        database: args[5].clone(),
    };
    let table1 = args[6].clone();
    let table2 = args[7].clone();
    let algo_num: u8 = args[8].parse().context("algo must be 1, 2 or 3")?;
    let out_path = args[9].clone();
    let format = args.get(10).map(|s| s.to_ascii_lowercase()).unwrap_or_else(|| "csv".to_string());


    let algorithm = match algo_num {
        1 => MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        2 => MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
        3 => MatchingAlgorithm::Fuzzy,
        _ => {
            eprintln!("algo must be 1, 2 or 3");
            std::process::exit(2);
        }
    };

    info!("Connecting to MySQL at {}:{} / db {}", cfg.host, cfg.port, cfg.database);
    let pool = make_pool(&cfg).await?;


    if matches!(algorithm, MatchingAlgorithm::Fuzzy) && format != "csv" {
        eprintln!("Fuzzy algorithm supports CSV format only. Use format=csv.");
        std::process::exit(2);
    }

    // Validate schemas
    let cols1 = discover_table_columns(&pool, &cfg.database, &table1).await?;
    let cols2 = discover_table_columns(&pool, &cfg.database, &table2).await?;
    cols1.validate_basic()?;
    cols2.validate_basic()?;
    info!("{} columns: {:?}", table1, cols1);
    info!("{} columns: {:?}", table2, cols2);
    if std::env::var("CHECK_ONLY").is_ok() {
        info!("Schema OK; exiting due to CHECK_ONLY");
        return Ok(())
    }

    // Decide execution mode (streaming vs in-memory)
    let c1 = get_person_count(&pool, &table1).await?;
    let c2 = get_person_count(&pool, &table2).await?;
    let streaming_env = std::env::var("NAME_MATCHER_STREAMING").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true") ).unwrap_or(false);
    let big_data_heuristic = (c1 + c2) > 200_000; // heuristic threshold
    let use_streaming = if matches!(algorithm, MatchingAlgorithm::Fuzzy) { false } else { streaming_env || big_data_heuristic };

    use crate::metrics::memory_stats_mb;
    let cfgp = ProgressConfig { update_every: 1000, ..Default::default() };

    if use_streaming {
        info!("Streaming mode enabled ({} + {} rows).", c1, c2);
        let s_cfg = StreamingConfig::default();
        if format == "csv" || format == "both" {
            info!("Streaming CSV export to {} using {:?}", out_path, algorithm);
            let mut writer = CsvStreamWriter::create(&out_path, algorithm)?;
            let t_match = std::time::Instant::now();
            let count = stream_match_csv(&pool, &table1, &table2, algorithm, |pair| { writer.write(pair) }, s_cfg, |u: ProgressUpdate| {
                info!("Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                    u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
            }, None).await?;
            writer.flush()?;
            info!("Wrote {} matches (streaming) in {:?}", count, t_match.elapsed());
        }
        if format == "xlsx" || format == "both" {
            // Streaming both algorithms into an XLSX workbook
            let xlsx_path = if out_path.to_ascii_lowercase().ends_with(".xlsx") { out_path.clone() } else { out_path.replace(".csv", ".xlsx") };
            let mem_start = memory_stats_mb().used_mb;
            let mut xw = XlsxStreamWriter::create(&xlsx_path)?;
            let t1 = std::time::Instant::now();
            let mut algo1_count = 0usize;
            stream_match_csv(&pool, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |pair| { algo1_count += 1; xw.append_algo1(pair) }, s_cfg, |_u| {}, None).await?;
            let took_a1 = t1.elapsed();
            let t2 = std::time::Instant::now();
            let mut algo2_count = 0usize;
            stream_match_csv(&pool, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |pair| { algo2_count += 1; xw.append_algo2(pair) }, s_cfg, |_u| {}, None).await?;
            let took_a2 = t2.elapsed();
            let mem_end = memory_stats_mb().used_mb;
            let summary = SummaryContext {
                db_name: cfg.database.clone(),
                table1: table1.clone(),
                table2: table2.clone(),
                total_table1: c1 as usize,
                total_table2: c2 as usize,
                matches_algo1: algo1_count,
                matches_algo2: algo2_count,
                overlap_count: 0, // not tracked in streaming mode to save memory
                unique_algo1: algo1_count,
                unique_algo2: algo2_count,
                fetch_time: std::time::Duration::from_secs(0),
                match1_time: took_a1,
                match2_time: took_a2,
                export_time: std::time::Duration::from_secs(0),
                mem_used_start_mb: mem_start,
                mem_used_end_mb: mem_end,
                timestamp: chrono::Utc::now(),
            };
            xw.finalize(&summary)?;
            info!("XLSX written (streaming) to {} | a1={} a2={}", xlsx_path, algo1_count, algo2_count);
        }
    } else {
        // In-memory fallback (previous behavior)
        info!("Fetching rows from {} and {}", table1, table2);
        let t_fetch = std::time::Instant::now();
        let people1 = get_person_rows(&pool, &table1).await?;
        let people2 = get_person_rows(&pool, &table2).await?;
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
            export_to_csv(&matches_requested, &out_path, algorithm)?;
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
                db_name: cfg.database.clone(),
                table1: table1.clone(),
                table2: table2.clone(),
                total_table1: people1.len(),
                total_table2: people2.len(),
                matches_algo1: a1.len(),
                matches_algo2: a2.len(),
                overlap_count: overlap,
                unique_algo1: unique1,
                unique_algo2: unique2,
                fetch_time: took_fetch,
                match1_time: took_a1,
                match2_time: took_a2,
                export_time: std::time::Duration::from_secs(0),
                mem_used_start_mb: mem_start,
                mem_used_end_mb: mem_end,
                timestamp: chrono::Utc::now(),
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
        }
    }

    info!("Done.");
    Ok(())
}
