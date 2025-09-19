use anyhow::{bail, Context, Result};
use sqlx::{MySql, MySqlPool, Row};

use crate::models::{Person, TableColumns, ColumnMapping};

#[derive(Debug, Clone)]
pub enum SqlBind { I64(i64), Str(String) }

fn build_select_list(mapping: Option<&ColumnMapping>) -> Result<String> {
    // Validate identifiers to prevent injection; when None, use defaults
    let m = mapping.cloned().unwrap_or_default();
    // Validate each column ident when provided
    fn val(id: &str) -> Result<()> { super::schema::validate_ident(id) }
    val(&m.id)?; val(&m.uuid)?; val(&m.first_name)?; val(&m.last_name)?; val(&m.birthdate)?;
    if let Some(ref mid) = m.middle_name { val(mid)?; }
    let mid_sql = if let Some(mid) = m.middle_name.as_ref() { format!("`{}` AS middle_name", mid) } else { "NULL AS middle_name".to_string() };
    Ok(format!(
        "`{id}` AS id, `{uuid}` AS uuid, `{first}` AS first_name, {mid}, `{last}` AS last_name, DATE(`{bd}`) AS birthdate",
        id = m.id, uuid = m.uuid, first = m.first_name, mid = mid_sql, last = m.last_name, bd = m.birthdate
    ))
}

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


// --- Flexible/mapped selection helpers and WHERE-aware fetchers ---
#[allow(dead_code)]
pub async fn get_person_rows_mapped(pool: &MySqlPool, table: &str, mapping: Option<&ColumnMapping>) -> Result<Vec<Person>> {
    super::schema::validate_ident(table)?;
    let select = build_select_list(mapping)?;
    let sql = format!("SELECT {select} FROM `{table}`", select=select, table=table);
    let rows: Vec<Person> = sqlx::query_as::<MySql, Person>(&sql)
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to fetch rows from {} (mapped)", table))?;
    Ok(rows)
}

pub async fn get_person_count_where(pool: &MySqlPool, table: &str, where_sql: &str, binds: &[SqlBind]) -> Result<i64> {
    super::schema::validate_ident(table)?;
    let sql = format!("SELECT COUNT(*) as cnt FROM `{}` WHERE {}", table, where_sql);
    let mut q = sqlx::query(&sql);
    for b in binds {
        q = match b { SqlBind::I64(v) => q.bind(*v), SqlBind::Str(s) => q.bind(s) };
    }
    let row = q.fetch_one(pool).await?;
    let cnt: i64 = row.try_get("cnt")?;
    Ok(cnt)
}

pub async fn get_person_rows_where(pool: &MySqlPool, table: &str, where_sql: &str, binds: &[SqlBind], mapping: Option<&ColumnMapping>) -> Result<Vec<Person>> {
    super::schema::validate_ident(table)?;
    let select = build_select_list(mapping)?;
    let sql = format!("SELECT {select} FROM `{table}` WHERE {where}", select=select, table=table, where=where_sql);
    let mut q = sqlx::query_as::<MySql, Person>(&sql);
    for b in binds { q = match b { SqlBind::I64(v) => q.bind(*v), SqlBind::Str(s) => q.bind(s) }; }
    let rows = q.fetch_all(pool).await.with_context(|| format!("Failed to fetch rows from {} with filter", table))?;
    Ok(rows)
}

pub async fn fetch_person_rows_chunk_where(pool: &MySqlPool, table: &str, offset: i64, limit: i64, where_sql: &str, binds: &[SqlBind], mapping: Option<&ColumnMapping>) -> Result<Vec<Person>> {
    super::schema::validate_ident(table)?;
    let select = build_select_list(mapping)?;
    let sql = format!("SELECT {select} FROM `{table}` WHERE {where} ORDER BY id LIMIT ? OFFSET ?", select=select, table=table, where=where_sql);
    let mut q = sqlx::query_as::<MySql, Person>(&sql);
    for b in binds { q = match b { SqlBind::I64(v) => q.bind(*v), SqlBind::Str(s) => q.bind(s) }; }
    let rows: Vec<Person> = q.bind(limit).bind(offset).fetch_all(pool).await
        .with_context(|| format!("Failed to fetch chunk from {} (offset {}, limit {}) with filter", table, offset, limit))?;
    Ok(rows)
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
