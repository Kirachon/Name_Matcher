pub mod connection;
pub mod schema;

#[allow(unused_imports)]
pub use connection::{make_pool, make_pool_with_size};
pub use schema::{discover_table_columns, get_person_rows, get_person_count, fetch_person_rows_chunk};

