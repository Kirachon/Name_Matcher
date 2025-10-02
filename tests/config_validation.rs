use name_matcher::config::{AppConfig, DatabaseConfig, StreamingConfig, GpuConfig, MatchingConfig, ExportConfig};

#[test]
fn defaults_and_validation_ok() {
    let cfg = AppConfig {
        database: DatabaseConfig { username: "root".into(), password: "root".into(), host: "127.0.0.1".into(), port: 3307, database: "people".into() },
        streaming: StreamingConfig::default(),
        gpu: GpuConfig { enabled: false, use_hash_join: false, build_on_gpu: false, probe_on_gpu: false, vram_mb_budget: None },
        matching: MatchingConfig { algorithm: Some(3), min_score_export: Some(95.0) },
        export: ExportConfig { out_path: Some("./tmp/out.csv".into()), format: Some("csv".into()) },
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn validation_catches_issues() {
    let bad = AppConfig {
        database: DatabaseConfig { username: "".into(), password: "".into(), host: "".into(), port: 0, database: "".into() },
        ..Default::default()
    };
    let err = bad.validate().unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("missing required field") || msg.contains("out of range"));
}

