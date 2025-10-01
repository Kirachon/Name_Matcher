#![cfg(feature = "new_engine")]

use name_matcher::engine::{file_checkpointer::FileCheckpointer, Checkpointer};

#[test]
fn file_checkpointer_save_and_load_roundtrip() {
    let base = std::env::temp_dir().join(format!("nm_ckpt_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
    let mut ck = FileCheckpointer::new(&base);
    let job = "job1"; let part = "pA";
    assert!(ck.load(job, part).unwrap().is_none());
    ck.save(job, part, "token-123").unwrap();
    let got = ck.load(job, part).unwrap();
    assert_eq!(got.as_deref(), Some("token-123"));
}

#[test]
fn file_checkpointer_handles_missing() {
    let base = std::env::temp_dir().join("nm_ckpt_missing");
    let ck = FileCheckpointer::new(&base);
    assert!(ck.load("no_job", "no_part").unwrap().is_none());
}

