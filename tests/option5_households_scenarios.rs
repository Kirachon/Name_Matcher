use anyhow::Result;
use name_matcher::db::schema::get_person_rows;
use name_matcher::matching::{match_households_gpu_inmemory, MatchOptions, ComputeBackend, ProgressConfig};
use sqlx::mysql::MySqlPoolOptions;

fn has_pair(rows: &[(String, i64)], u: &str, h: i64) -> bool {
    rows.iter().any(|(uu, hh)| uu == u && *hh == h)
}

#[tokio::test]
async fn option5_households_multiple_scenarios() -> Result<()> {
    // Connect to local Docker MySQL (container maps 3307->3306)
    let host = std::env::var("DB_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("DB_PORT").ok().and_then(|s| s.parse::<u16>().ok()).unwrap_or(3307);
    let user = std::env::var("DB_USER").unwrap_or_else(|_| "root".into());
    let pass = std::env::var("DB_PASS").unwrap_or_else(|_| "root".into());
    let db   = std::env::var("DB_NAME").unwrap_or_else(|_| "duplicate_checker".into());
    let url = format!("mysql://{}:{}@{}:{}/{}", user, pass, host, port, db);
    let pool = MySqlPoolOptions::new().max_connections(10).connect(&url).await?;

    // Create scenario tables
    sqlx::query("DROP TABLE IF EXISTS t1_uuid_hh_scen").execute(&pool).await?;
    sqlx::query("DROP TABLE IF EXISTS t2_hhid_hh_scen").execute(&pool).await?;

    sqlx::query(
        r#"
        CREATE TABLE t1_uuid_hh_scen (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            uuid VARCHAR(64) NOT NULL,
            first_name VARCHAR(64) NULL,
            middle_name VARCHAR(64) NULL,
            last_name VARCHAR(64) NULL,
            birthdate DATE NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        "#
    ).execute(&pool).await?;

    sqlx::query(
        r#"
        CREATE TABLE t2_hhid_hh_scen (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            hh_id BIGINT NOT NULL,
            first_name VARCHAR(64) NULL,
            middle_name VARCHAR(64) NULL,
            last_name VARCHAR(64) NULL,
            birthdate DATE NOT NULL,
            INDEX idx_hhid (hh_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        "#
    ).execute(&pool).await?;

    // Seed scenarios
    // AA-UUID: 3 members -> matches hh 210 on 2/3 (66.7%), plus another stray in hh 214 (ignored at 60%+)
    // BB-UUID: 2 members -> matches hh 211 on 2/2 (100%)
    // CC-UUID: 4 members -> matches hh 212 on 3/4 (75%)
    // DD-UUID: 1 member -> no match
    let ins_t1 = r#"INSERT INTO t1_uuid_hh_scen (uuid, first_name, middle_name, last_name, birthdate) VALUES
        ('AA-UUID','John',NULL,'Doe','1990-01-01'),
        ('AA-UUID','Jane',NULL,'Doe','1992-02-02'),
        ('AA-UUID','Jimmy',NULL,'Doe','2010-03-03'),
        ('BB-UUID','Alice',NULL,'Smith','1985-05-05'),
        ('BB-UUID','Bob',NULL,'Smith','1987-06-06'),
        ('CC-UUID','Carl',NULL,'Ng','1970-07-07'),
        ('CC-UUID','Cathy',NULL,'Ng','1972-08-08'),
        ('CC-UUID','Chris',NULL,'Ng','1999-09-09'),
        ('CC-UUID','Cindy',NULL,'Ng','2001-10-10'),
        ('DD-UUID','Dan',NULL,'Solo','1991-11-11')
    "#;
    sqlx::query(ins_t1).execute(&pool).await?;

    let ins_t2 = r#"INSERT INTO t2_hhid_hh_scen (hh_id, first_name, middle_name, last_name, birthdate) VALUES
        (210,'John',NULL,'Doe','1990-01-01'),
        (210,'Jane',NULL,'Doe','1992-02-02'),
        (211,'Alice',NULL,'Smith','1985-05-05'),
        (211,'Bob',NULL,'Smith','1987-06-06'),
        (212,'Carl',NULL,'Ng','1970-07-07'),
        (212,'Cathy',NULL,'Ng','1972-08-08'),
        (212,'Chris',NULL,'Ng','1999-09-09'),
        (214,'Jimmy',NULL,'Doe','2010-03-03')
    "#;
    sqlx::query(ins_t2).execute(&pool).await?;

    // Load and run Option 5
    let t1 = get_person_rows(&pool, "t1_uuid_hh_scen").await?;
    let t2 = get_person_rows(&pool, "t2_hhid_hh_scen").await?;
    assert_eq!(t1.len(), 10);
    assert_eq!(t2.len(), 8);

    let opts = MatchOptions { backend: ComputeBackend::Cpu, gpu: None, progress: ProgressConfig::default() };

    // Threshold 0.60: expect AA-UUID->210 (66.7%), BB-UUID->211 (100%), CC-UUID->212 (75%). DD-UUID has none.
    let rows60 = match_households_gpu_inmemory(&t1, &t2, opts.clone(), 0.60, |_u| {});
    let pairs60: Vec<(String, i64)> = rows60.iter().map(|r| (r.uuid.clone(), r.hh_id)).collect();
    assert!(has_pair(&pairs60, "AA-UUID", 210), "expected AA-UUID->210 at 60%: got {:?}", pairs60);
    assert!(has_pair(&pairs60, "BB-UUID", 211), "expected BB-UUID->211 at 60%: got {:?}", pairs60);
    assert!(has_pair(&pairs60, "CC-UUID", 212), "expected CC-UUID->212 at 60%: got {:?}", pairs60);

    // Threshold 0.80: expect only 100% household (BB-UUID->211)
    let rows80 = match_households_gpu_inmemory(&t1, &t2, opts, 0.80, |_u| {});
    let pairs80: Vec<(String, i64)> = rows80.iter().map(|r| (r.uuid.clone(), r.hh_id)).collect();
    assert!(has_pair(&pairs80, "BB-UUID", 211), "expected only BB-UUID->211 at 80%: got {:?}", pairs80);

    Ok(())
}

