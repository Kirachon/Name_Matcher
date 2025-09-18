use anyhow::{bail, Context, Result};
use sqlx::{MySql, MySqlPool, Row};

use crate::models::{Person, TableColumns};

fn validate_ident(name: &str) -> Result<()> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' ) {
        bail!("Invalid identifier: {}", name);
    }
    Ok(())
}

pub async fn discover_table_columns(pool: &MySqlPool, database: &str, table: &str) -> Result<TableColumns> {
    validate_ident(database)?;
    validate_ident(table)?;

    let rows = sqlx::query(
        r#"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"#
    )
    .bind(database)
    .bind(table)
    .fetch_all(pool)
    .await
    .with_context(|| format!("Failed to query columns for {}.{}", database, table))?;

    let mut cols = TableColumns {
        has_id: false,
        has_uuid: false,
        has_first_name: false,
        has_middle_name: false,
        has_last_name: false,
        has_birthdate: false,
    };
    for r in rows {
        let name: String = r.try_get("COLUMN_NAME")?;
        match name.as_str() {
            "id" => cols.has_id = true,
            "uuid" => cols.has_uuid = true,
            "first_name" => cols.has_first_name = true,
            "middle_name" => cols.has_middle_name = true,
            "last_name" => cols.has_last_name = true,
            "birthdate" => cols.has_birthdate = true,
            _ => {}
        }
    }
    Ok(cols)
}

pub async fn get_person_rows(pool: &MySqlPool, table: &str) -> Result<Vec<Person>> {
    validate_ident(table)?;
    // safe-ish table name injection after validation
    let sql = format!(
        "SELECT id, uuid, first_name, middle_name, last_name, DATE(birthdate) AS birthdate FROM `{}`",
        table
    );
    let rows: Vec<Person> = sqlx::query_as::<MySql, Person>(&sql)
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to fetch rows from {}", table))?;
    Ok(rows)
}



pub async fn get_person_count(pool: &MySqlPool, table: &str) -> Result<i64> {
    validate_ident(table)?;
    let sql = format!("SELECT COUNT(*) as cnt FROM `{}`", table);
    let row = sqlx::query(&sql).fetch_one(pool).await?;
    let cnt: i64 = row.try_get("cnt")?;
    Ok(cnt)
}

pub async fn fetch_person_rows_chunk(pool: &MySqlPool, table: &str, offset: i64, limit: i64) -> Result<Vec<Person>> {
    validate_ident(table)?;
    let sql = format!(
        "SELECT id, uuid, first_name, middle_name, last_name, DATE(birthdate) AS birthdate FROM `{}` ORDER BY id LIMIT ? OFFSET ?",
        table
    );
    let rows: Vec<Person> = sqlx::query_as::<MySql, Person>(&sql)
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to fetch chunk from {} (offset {}, limit {})", table, offset, limit))?;
    Ok(rows)
}
