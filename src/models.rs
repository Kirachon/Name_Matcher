use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Person {
    pub id: i64,
    pub uuid: String,
    pub first_name: String,
    pub middle_name: Option<String>,
    pub last_name: String,
    pub birthdate: NaiveDate,
}

#[derive(Debug, Clone)]
pub struct NormalizedPerson {
    pub id: i64,
    pub uuid: String,
    pub first_name: String,
    pub middle_name: Option<String>,
    pub last_name: String,
    pub birthdate: NaiveDate,
}

#[derive(Debug, Clone)]
pub struct TableColumns {
    pub has_id: bool,
    pub has_uuid: bool,
    pub has_first_name: bool,
    pub has_middle_name: bool,
    pub has_last_name: bool,
    pub has_birthdate: bool,
}

impl TableColumns {
    pub fn validate_basic(&self) -> anyhow::Result<()> {
        use anyhow::bail;
        if !(self.has_id && self.has_uuid && self.has_first_name && self.has_last_name && self.has_birthdate) {
            bail!("Table missing required columns: requires id, uuid, first_name, last_name, birthdate (middle_name optional)");
        }
        Ok(())
    }
}

