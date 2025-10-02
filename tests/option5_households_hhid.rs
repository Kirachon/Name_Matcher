use anyhow::Result;
use name_matcher::db::schema::get_person_rows;
use name_matcher::matching::{match_households_gpu_inmemory, MatchOptions, ComputeBackend, ProgressConfig};
use sqlx::mysql::MySqlPoolOptions;

fn has_pair(rows: &[(String, i64)], u: &str, h: i64) -> bool {
    rows.iter().any(|(uu, hh)| uu == u && *hh == h)
}

#[tokio::test]
async fn option5_households_uses_hhid_for_table2() -> Result<()> {
    // Connect to local Docker MySQL (container maps 3307->3306)
    let host = std::env::var("DB_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("DB_PORT").ok().and_then(|s| s.parse::<u16>().ok()).unwrap_or(3307);
    let user = std::env::var("DB_USER").unwrap_or_else(|_| "root".into());
    let pass = std::env::var("DB_PASS").unwrap_or_else(|_| "root".into());
    let db   = std::env::var("DB_NAME").unwrap_or_else(|_| "duplicate_checker".into());
    let url = format!("mysql://{}:{}@{}:{}/{}", user, pass, host, port, db);
    let pool = MySqlPoolOptions::new().max_connections(10).connect(&url).await?;

    // Create test tables
    sqlx::query("DROP TABLE IF EXISTS t1_uuid_hh").execute(&pool).await?;
    sqlx::query("DROP TABLE IF EXISTS t2_hhid_hh").execute(&pool).await?;

    sqlx::query(
        r#"
        CREATE TABLE t1_uuid_hh (
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
        CREATE TABLE t2_hhid_hh (
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

    // Seed data: Two T1 households (A-UUID, B-UUID). T2 has hh_id 100,101,102
    // A-UUID: 3 members -> T2 hh 100 has John/Jane (2/3 => 66.7%), hh 102 has Jimmy (1/3 => 33.3%)
    // B-UUID: 2 members -> T2 hh 101 has both (2/2 => 100%)
    let ins_t1 = r#"INSERT INTO t1_uuid_hh (uuid, first_name, middle_name, last_name, birthdate) VALUES
        ('A-UUID','John',NULL,'Doe','1990-01-01'),
        ('A-UUID','Jane',NULL,'Doe','1992-02-02'),
        ('A-UUID','Jimmy',NULL,'Doe','2010-03-03'),
        ('B-UUID','Alice',NULL,'Smith','1985-05-05'),
        ('B-UUID','Bob',NULL,'Smith','1987-06-06')
    "#;
    sqlx::query(ins_t1).execute(&pool).await?;

    let ins_t2 = r#"INSERT INTO t2_hhid_hh (hh_id, first_name, middle_name, last_name, birthdate) VALUES
        (100,'John',NULL,'Doe','1990-01-01'),
        (100,'Jane',NULL,'Doe','1992-02-02'),
        (101,'Alice',NULL,'Smith','1985-05-05'),
        (101,'Bob',NULL,'Smith','1987-06-06'),
        (102,'Jimmy',NULL,'Doe','2010-03-03')
    "#;
    sqlx::query(ins_t2).execute(&pool).await?;

    // Load and run Option 5
    let t1 = get_person_rows(&pool, "t1_uuid_hh").await?;
    let t2 = get_person_rows(&pool, "t2_hhid_hh").await?;
    assert_eq!(t1.len(), 5);
    assert_eq!(t2.len(), 5);

    let opts = MatchOptions { backend: ComputeBackend::Cpu, gpu: None, progress: ProgressConfig::default() };
    let rows = match_households_gpu_inmemory(&t1, &t2, opts, 0.95, |_u| {});

    // Expect exactly two aggregated rows using hh_id 100 and 101 (not person id)
    let pairs: Vec<(String, i64)> = rows.iter().map(|r| (r.uuid.clone(), r.hh_id)).collect();
    assert_eq!(rows.len(), 2, "expected two household matches >50%: got {:?}", pairs);
    assert!(has_pair(&pairs, "A-UUID", 100), "missing (A-UUID, hh_id=100). got: {:?}", pairs);
    assert!(has_pair(&pairs, "B-UUID", 101), "missing (B-UUID, hh_id=101). got: {:?}", pairs);

    Ok(())
}

