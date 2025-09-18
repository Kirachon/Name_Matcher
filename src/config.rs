use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub username: String,
    pub password: String,
    pub host: String,
    pub port: u16,
    pub database: String,
}

impl DatabaseConfig {
    pub fn to_url(&self) -> String {
        format!(
            "mysql://{}:{}@{}:{}/{}",
            self.username,
            self.password,
            self.host,
            self.port,
            self.database
        )
    }
}

impl std::fmt::Debug for DatabaseConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatabaseConfig")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("database", &self.database)
            .finish()
    }
}

