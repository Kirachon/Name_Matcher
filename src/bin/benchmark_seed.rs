/// Enhanced benchmark data generator for Name_Matcher performance validation
/// Generates realistic million-record datasets with:
/// - Representative name distributions (common/rare names, Unicode, diacritics)
/// - Clean and dirty datasets for fuzzy matching validation
/// - Configurable duplicate rates and error patterns
/// - Reproducible seeding for consistent benchmarks

use anyhow::{Result, Context};
use chrono::NaiveDate;
use sqlx::{mysql::MySqlPoolOptions, MySql, Pool};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    // Args: host port user pass db dataset_size [seed]
    let args: Vec<String> = std::env::args().collect();
    let host = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1".into());
    let port = args.get(2).and_then(|s| s.parse::<u16>().ok()).unwrap_or(3307);
    let user = args.get(3).cloned().unwrap_or_else(|| "root".into());
    let pass = args.get(4).cloned().unwrap_or_else(|| "root".into());
    let db   = args.get(5).cloned().unwrap_or_else(|| "benchmark_nm".into());
    let size_str = args.get(6).cloned().unwrap_or_else(|| "1M".into());
    let seed = args.get(7).and_then(|s| s.parse::<u64>().ok()).unwrap_or(42);
    
    let dataset_size = parse_size(&size_str)?;
    
    log::info!("=== Name_Matcher Benchmark Data Generator ===");
    log::info!("Target: MySQL {}:{} / database: {}", host, port, db);
    log::info!("Dataset size: {} records ({} per table)", dataset_size, size_str);
    log::info!("Random seed: {}", seed);
    
    // Connect to server and create database
    let url_server = format!("mysql://{user}:{pass}@{host}:{port}/mysql");
    let pool_server = MySqlPoolOptions::new().max_connections(5).connect(&url_server).await
        .context("Failed to connect to MySQL server")?;
    
    sqlx::query(&format!("CREATE DATABASE IF NOT EXISTS `{}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci", db))
        .execute(&pool_server).await?;
    log::info!("Database '{}' ready", db);
    
    // Connect to target database
    let url_db = format!("mysql://{user}:{pass}@{host}:{port}/{db}");
    let pool = MySqlPoolOptions::new()
        .max_connections(20)
        .connect(&url_db).await
        .context("Failed to connect to target database")?;
    
    // Create benchmark tables
    create_benchmark_tables(&pool).await?;
    
    // Generate datasets
    log::info!("Generating clean dataset (table: clean_a, clean_b)...");
    generate_clean_dataset(&pool, "clean_a", "clean_b", dataset_size, seed).await?;
    
    log::info!("Generating dirty dataset (table: dirty_a, dirty_b)...");
    generate_dirty_dataset(&pool, "dirty_a", "dirty_b", dataset_size, seed + 1000).await?;
    
    // Generate statistics
    print_dataset_stats(&pool, "clean_a", "clean_b").await?;
    print_dataset_stats(&pool, "dirty_a", "dirty_b").await?;
    
    log::info!("=== Benchmark data generation complete ===");
    Ok(())
}

fn parse_size(s: &str) -> Result<usize> {
    let s = s.to_uppercase();
    if let Some(num_str) = s.strip_suffix('M') {
        Ok(num_str.parse::<usize>()? * 1_000_000)
    } else if let Some(num_str) = s.strip_suffix('K') {
        Ok(num_str.parse::<usize>()? * 1_000)
    } else {
        Ok(s.parse::<usize>()?)
    }
}

async fn create_benchmark_tables(pool: &Pool<MySql>) -> Result<()> {
    log::info!("Creating benchmark tables...");
    
    let table_ddl = |name: &str| format!(
        "CREATE TABLE IF NOT EXISTS `{name}` (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            uuid VARCHAR(36) NULL,
            first_name VARCHAR(100) NOT NULL,
            middle_name VARCHAR(100) NULL,
            last_name VARCHAR(100) NOT NULL,
            birthdate DATE NOT NULL,
            hh_id BIGINT NULL,
            INDEX idx_name_bd (last_name, first_name, birthdate),
            INDEX idx_bd (birthdate),
            INDEX idx_uuid (uuid),
            INDEX idx_hh_id (hh_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"
    );
    
    for table in &["clean_a", "clean_b", "dirty_a", "dirty_b"] {
        sqlx::query(&table_ddl(table)).execute(pool).await?;
        sqlx::query(&format!("TRUNCATE TABLE `{}`", table)).execute(pool).await.ok();
    }
    
    log::info!("Tables created: clean_a, clean_b, dirty_a, dirty_b");
    Ok(())
}

async fn generate_clean_dataset(pool: &Pool<MySql>, table_a: &str, table_b: &str, size: usize, seed: u64) -> Result<()> {
    let mut rng = Lcg::new(seed);
    let names = NameGenerator::new();
    
    // Generate table A with exact duplicates (20% duplicate rate)
    let duplicate_rate = 0.20;
    let unique_count = (size as f64 * (1.0 - duplicate_rate)) as usize;
    let duplicate_count = size - unique_count;
    
    log::info!("  Table {}: {} unique + {} duplicates = {} total", table_a, unique_count, duplicate_count, size);
    
    let mut base_records: Vec<PersonRecord> = Vec::with_capacity(unique_count);
    for _ in 0..unique_count {
        base_records.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: names.random_first(&mut rng),
            middle_name: if rng.next() % 3 == 0 { Some(names.random_middle(&mut rng)) } else { None },
            last_name: names.random_last(&mut rng),
            birthdate: random_birthdate(&mut rng),
            hh_id: None,
        });
    }
    
    // Insert base records
    insert_records_batched(pool, table_a, &base_records).await?;
    
    // Insert duplicates (exact copies with different UUIDs)
    let mut duplicates: Vec<PersonRecord> = Vec::with_capacity(duplicate_count);
    for _ in 0..duplicate_count {
        let base = &base_records[rng.next() as usize % base_records.len()];
        duplicates.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: base.first_name.clone(),
            middle_name: base.middle_name.clone(),
            last_name: base.last_name.clone(),
            birthdate: base.birthdate,
            hh_id: None,
        });
    }
    insert_records_batched(pool, table_a, &duplicates).await?;
    
    // Generate table B (subset of A with 50% overlap)
    let overlap_rate = 0.50;
    let overlap_count = (size as f64 * overlap_rate) as usize;
    let unique_b_count = size - overlap_count;
    
    log::info!("  Table {}: {} from A + {} unique = {} total", table_b, overlap_count, unique_b_count, size);
    
    let mut table_b_records: Vec<PersonRecord> = Vec::with_capacity(size);
    
    // Add overlapping records from A
    for _ in 0..overlap_count {
        let base = &base_records[rng.next() as usize % base_records.len()];
        table_b_records.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: base.first_name.clone(),
            middle_name: base.middle_name.clone(),
            last_name: base.last_name.clone(),
            birthdate: base.birthdate,
            hh_id: None,
        });
    }
    
    // Add unique records to B
    for _ in 0..unique_b_count {
        table_b_records.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: names.random_first(&mut rng),
            middle_name: if rng.next() % 3 == 0 { Some(names.random_middle(&mut rng)) } else { None },
            last_name: names.random_last(&mut rng),
            birthdate: random_birthdate(&mut rng),
            hh_id: None,
        });
    }
    
    insert_records_batched(pool, table_b, &table_b_records).await?;
    
    Ok(())
}

async fn generate_dirty_dataset(pool: &Pool<MySql>, table_a: &str, table_b: &str, size: usize, seed: u64) -> Result<()> {
    let mut rng = Lcg::new(seed);
    let names = NameGenerator::new();
    
    // Generate table A with fuzzy duplicates (30% duplicate rate with errors)
    let duplicate_rate = 0.30;
    let unique_count = (size as f64 * (1.0 - duplicate_rate)) as usize;
    let duplicate_count = size - unique_count;
    
    log::info!("  Table {}: {} unique + {} fuzzy duplicates = {} total", table_a, unique_count, duplicate_count, size);
    
    let mut base_records: Vec<PersonRecord> = Vec::with_capacity(unique_count);
    for _ in 0..unique_count {
        base_records.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: names.random_first(&mut rng),
            middle_name: if rng.next() % 3 == 0 { Some(names.random_middle(&mut rng)) } else { None },
            last_name: names.random_last(&mut rng),
            birthdate: random_birthdate(&mut rng),
            hh_id: None,
        });
    }
    
    insert_records_batched(pool, table_a, &base_records).await?;
    
    // Insert fuzzy duplicates with realistic errors
    let mut duplicates: Vec<PersonRecord> = Vec::with_capacity(duplicate_count);
    for _ in 0..duplicate_count {
        let base = &base_records[rng.next() as usize % base_records.len()];
        let error_type = rng.next() % 5;
        
        let (first, last) = match error_type {
            0 => (add_typo(&base.first_name, &mut rng), base.last_name.clone()), // Typo in first name
            1 => (base.first_name.clone(), add_typo(&base.last_name, &mut rng)), // Typo in last name
            2 => (add_typo(&base.first_name, &mut rng), add_typo(&base.last_name, &mut rng)), // Typos in both
            3 => (truncate_name(&base.first_name, &mut rng), base.last_name.clone()), // Truncation
            _ => (base.first_name.clone(), base.last_name.clone()), // Exact match
        };
        
        duplicates.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: first,
            middle_name: if rng.next() % 2 == 0 { base.middle_name.clone() } else { None }, // Sometimes drop middle
            last_name: last,
            birthdate: base.birthdate, // Keep birthdate exact for matching
            hh_id: None,
        });
    }
    insert_records_batched(pool, table_a, &duplicates).await?;
    
    // Generate table B with similar fuzzy overlap
    let overlap_rate = 0.40;
    let overlap_count = (size as f64 * overlap_rate) as usize;
    let unique_b_count = size - overlap_count;
    
    log::info!("  Table {}: {} fuzzy from A + {} unique = {} total", table_b, overlap_count, unique_b_count, size);
    
    let mut table_b_records: Vec<PersonRecord> = Vec::with_capacity(size);
    
    // Add fuzzy overlapping records
    for _ in 0..overlap_count {
        let base = &base_records[rng.next() as usize % base_records.len()];
        let error_type = rng.next() % 4;
        
        let (first, last) = match error_type {
            0 => (add_typo(&base.first_name, &mut rng), base.last_name.clone()),
            1 => (base.first_name.clone(), add_typo(&base.last_name, &mut rng)),
            _ => (base.first_name.clone(), base.last_name.clone()),
        };
        
        table_b_records.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: first,
            middle_name: base.middle_name.clone(),
            last_name: last,
            birthdate: base.birthdate,
            hh_id: None,
        });
    }
    
    // Add unique records
    for _ in 0..unique_b_count {
        table_b_records.push(PersonRecord {
            uuid: Uuid::new_v4().to_string(),
            first_name: names.random_first(&mut rng),
            middle_name: if rng.next() % 3 == 0 { Some(names.random_middle(&mut rng)) } else { None },
            last_name: names.random_last(&mut rng),
            birthdate: random_birthdate(&mut rng),
            hh_id: None,
        });
    }
    
    insert_records_batched(pool, table_b, &table_b_records).await?;

    Ok(())
}

async fn insert_records_batched(pool: &Pool<MySql>, table: &str, records: &[PersonRecord]) -> Result<()> {
    let batch_size = 1000;
    let total = records.len();
    let mut inserted = 0;

    for chunk in records.chunks(batch_size) {
        let mut q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
        q.push("`").push(table).push("` (uuid, first_name, middle_name, last_name, birthdate, hh_id) VALUES ");

        for (i, rec) in chunk.iter().enumerate() {
            if i > 0 { q.push(", "); }
            q.push("(")
                .push_bind(Some(rec.uuid.clone()))
                .push(", ")
                .push_bind(&rec.first_name)
                .push(", ")
                .push_bind(rec.middle_name.as_ref())
                .push(", ")
                .push_bind(&rec.last_name)
                .push(", ")
                .push_bind(rec.birthdate)
                .push(", ")
                .push_bind(rec.hh_id)
                .push(")");
        }

        q.build().execute(pool).await?;
        inserted += chunk.len();

        if inserted % 50000 == 0 || inserted == total {
            log::info!("    Inserted {}/{} records into {}", inserted, total, table);
        }
    }

    Ok(())
}

async fn print_dataset_stats(pool: &Pool<MySql>, table_a: &str, table_b: &str) -> Result<()> {
    let count_a: (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM `{}`", table_a))
        .fetch_one(pool).await?;
    let count_b: (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM `{}`", table_b))
        .fetch_one(pool).await?;

    log::info!("Dataset stats: {} = {} rows, {} = {} rows", table_a, count_a.0, table_b, count_b.0);
    Ok(())
}

// Helper functions for data generation

#[derive(Clone)]
struct PersonRecord {
    uuid: String,
    first_name: String,
    middle_name: Option<String>,
    last_name: String,
    birthdate: NaiveDate,
    hh_id: Option<i64>,
}

fn random_birthdate(rng: &mut Lcg) -> NaiveDate {
    let year = 1950 + (rng.next() % 61) as i32;
    let month = 1 + (rng.next() % 12) as u32;
    let mut day_max = match month { 1|3|5|7|8|10|12 => 31, 4|6|9|11 => 30, _ => 28 };
    if month == 2 && (year % 4 == 0) { day_max = 29; }
    let day = 1 + (rng.next() % day_max as u64) as u32;
    NaiveDate::from_ymd_opt(year, month, day).unwrap_or_else(|| NaiveDate::from_ymd_opt(2000,1,1).unwrap())
}

fn add_typo(name: &str, rng: &mut Lcg) -> String {
    if name.is_empty() { return name.to_string(); }

    let chars: Vec<char> = name.chars().collect();
    let typo_type = rng.next() % 4;

    match typo_type {
        0 => { // Substitution
            let pos = (rng.next() as usize) % chars.len();
            let mut result = chars.clone();
            result[pos] = ((b'a' + (rng.next() % 26) as u8) as char).to_ascii_lowercase();
            result.iter().collect()
        }
        1 => { // Deletion
            if chars.len() > 1 {
                let pos = (rng.next() as usize) % chars.len();
                let mut result = chars.clone();
                result.remove(pos);
                result.iter().collect()
            } else {
                name.to_string()
            }
        }
        2 => { // Insertion
            let pos = (rng.next() as usize) % (chars.len() + 1);
            let mut result = chars.clone();
            let new_char = ((b'a' + (rng.next() % 26) as u8) as char).to_ascii_lowercase();
            result.insert(pos, new_char);
            result.iter().collect()
        }
        _ => { // Transposition
            if chars.len() > 1 {
                let pos = (rng.next() as usize) % (chars.len() - 1);
                let mut result = chars.clone();
                result.swap(pos, pos + 1);
                result.iter().collect()
            } else {
                name.to_string()
            }
        }
    }
}

fn truncate_name(name: &str, rng: &mut Lcg) -> String {
    if name.len() <= 2 { return name.to_string(); }
    let keep_len = 2 + (rng.next() as usize % (name.len() - 2));
    name.chars().take(keep_len).collect()
}

// Simple LCG for deterministic pseudo-random generation
struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }
}

// Name generator with realistic distributions including Unicode
struct NameGenerator {
    first_names: Vec<&'static str>,
    last_names: Vec<&'static str>,
    middle_names: Vec<&'static str>,
}

impl NameGenerator {
    fn new() -> Self {
        Self {
            first_names: vec![
                // Common English names
                "James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda","William","Elizabeth",
                "David","Barbara","Richard","Susan","Joseph","Jessica","Thomas","Sarah","Charles","Karen",
                "Christopher","Nancy","Daniel","Lisa","Matthew","Betty","Anthony","Margaret","Mark","Sandra",
                "Donald","Ashley","Steven","Kimberly","Paul","Emily","Andrew","Donna","Joshua","Michelle",
                // Names with diacritics
                "José","María","François","André","René","Zoë","Chloé","Anaïs","Björn","Søren",
                "Müller","Günther","Jürgen","Łukasz","Michał","Ángel","Sofía","Nicolás","Andrés",
                // Asian names
                "Wei","Ming","Li","Chen","Wang","Zhang","Yuki","Hiroshi","Kenji","Sakura",
                "Raj","Priya","Amit","Sanjay","Deepak","Mohammed","Ahmed","Fatima","Ali",
            ],
            last_names: vec![
                // Common surnames
                "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
                "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
                "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
                "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
                // Surnames with diacritics
                "García","Rodríguez","Martínez","Hernández","López","González","Pérez","Sánchez","Ramírez",
                "Müller","Schmidt","Schneider","Fischer","Weber","Meyer","Wagner","Becker","Schulz",
                "O'Brien","O'Connor","O'Neill","Ó Súilleabháin","Mc Donald","Mc Carthy",
            ],
            middle_names: vec![
                "Lee","Ann","Marie","Lynn","Ray","Mae","Jo","Jay","Kim","Sue",
                "A","B","C","D","E","F","G","H","J","K","L","M","N","P","R","S","T","W",
            ],
        }
    }

    fn random_first(&self, rng: &mut Lcg) -> String {
        self.first_names[(rng.next() as usize) % self.first_names.len()].to_string()
    }

    fn random_last(&self, rng: &mut Lcg) -> String {
        self.last_names[(rng.next() as usize) % self.last_names.len()].to_string()
    }

    fn random_middle(&self, rng: &mut Lcg) -> String {
        self.middle_names[(rng.next() as usize) % self.middle_names.len()].to_string()
    }
}

