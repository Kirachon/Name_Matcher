use anyhow::Result;
use name_matcher::db::schema::{get_person_rows};
use name_matcher::export::csv_export::HouseholdCsvWriter;
use name_matcher::export::xlsx_export::export_households_xlsx;
use name_matcher::matching::{match_households_gpu_inmemory, MatchOptions, ComputeBackend, ProgressConfig};
use sqlx::mysql::MySqlPoolOptions;

fn out_path(name: &str) -> String { format!("./tmp/{}.xlsx", name) }
fn out_csv(name: &str) -> String { format!("./tmp/{}.csv", name) }

#[tokio::test]
#[ignore]
async fn option5_households_end_to_end() -> Result<()> {
    // Connect to local Docker MySQL (container maps 3307->3306)
    let host = std::env::var("DB_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("DB_PORT").ok().and_then(|s| s.parse::<u16>().ok()).unwrap_or(3307);
    let user = std::env::var("DB_USER").unwrap_or_else(|_| "root".into());
    let pass = std::env::var("DB_PASS").unwrap_or_else(|_| "root".into());
    let db   = std::env::var("DB_NAME").unwrap_or_else(|_| "duplicate_checker".into());

    let url = format!("mysql://{}:{}@{}:{}/{}", user, pass, host, port, db);
    let pool = MySqlPoolOptions::new().max_connections(10).connect(&url).await?;

    let t1 = get_person_rows(&pool, "sample_a").await?;
    let t2 = get_person_rows(&pool, "sample_b").await?;
    assert!(t1.len() >= 4000 && t2.len() >= 4000, "expected seeded dataset present");

    let opts = MatchOptions { backend: ComputeBackend::Cpu, gpu: None, progress: ProgressConfig::default() };

    for &thr in &[0.60_f32, 0.80_f32, 0.95_f32] {
        let rows = match_households_gpu_inmemory(&t1, &t2, opts, thr, |_u| {});
        // All rows must satisfy >50% by construction of the aggregator
        assert!(rows.iter().all(|r| r.match_percentage > 50.0), "all exported rows must be >50% matched");

        // XLSX export
        let xlsx_path = out_path(&format!("option5_households_{:.0}", thr*100.0));
        export_households_xlsx(&xlsx_path, &rows)?;

        // CSV export
        let csv_path = out_csv(&format!("option5_households_{:.0}", thr*100.0));
        let mut w = HouseholdCsvWriter::create(&csv_path)?;
        for r in &rows { w.write(r)?; }
        w.flush()?;

        // At least one >50% household should be present for 60% and 80% thresholds given injected H_GT50
        if thr <= 0.80 { assert!(rows.len() > 0, "expected some >50% households for thr {}", thr); }
    }

    Ok(())
}

