// SRS-II Name Matching Application GUI
// Creator/Author: Matthias Tangonan
// This file implements the desktop GUI using eframe/egui.

use eframe::egui::{self, ComboBox, Context, ProgressBar, TextEdit};
use eframe::{App, Frame, NativeOptions};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use std::thread;

use anyhow::Result;
use chrono::Utc;
use serde::{Serialize, Deserialize};
use std::fs;

use name_matcher::config::DatabaseConfig;
use name_matcher::db::make_pool_with_size;
use name_matcher::db::{get_person_rows, get_person_count};
use name_matcher::matching::{stream_match_csv, stream_match_csv_dual, match_all_progress, match_all_with_opts, match_households_gpu_inmemory, ProgressConfig, MatchingAlgorithm, ProgressUpdate, StreamingConfig, StreamControl, ComputeBackend, MatchOptions, GpuConfig};
use name_matcher::export::csv_export::{CsvStreamWriter, HouseholdCsvWriter};
use name_matcher::export::xlsx_export::{XlsxStreamWriter, SummaryContext, export_households_xlsx};

#[derive(Clone, Copy, PartialEq)]
enum ModeSel { Auto, Streaming, InMemory }

#[derive(Clone, Copy, PartialEq)]
enum FormatSel { Csv, Xlsx, Both }

#[derive(Debug)]
enum Msg {
    Progress(ProgressUpdate),
    Info(String),
    Tables(Vec<String>),
    Tables2(Vec<String>),
    Done { a1: usize, a2: usize, csv: usize, path: String },
    Error(String),
    ErrorRich { display: String, sqlstate: Option<String>, chain: String, operation: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ReportFormat { Text, Json }

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
enum ErrorCategory {
    DbConnection,
    TableValidation,
    SchemaValidation,
    DataFormat,
    ResourceConstraint,
    Configuration,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
struct DiagEvent {
    ts_utc: String,
    category: ErrorCategory,
    message: String,
    sqlstate: Option<String>,
    chain: Option<String>,
    operation: Option<String>,
    source_action: String,
    db1_host: String,
    db1_database: String,
    db2_host: Option<String>,
    db2_database: Option<String>,
    table1: Option<String>,
    table2: Option<String>,
    mem_avail_mb: u64,
    pool_size_cfg: u32,
    env_overrides: Vec<(String,String)>,
}

fn categorize_error(msg: &str) -> ErrorCategory {
    categorize_error_with(None, msg)
}

fn categorize_error_with(sqlstate: Option<&str>, msg: &str) -> ErrorCategory {
    if let Some(state) = sqlstate {
        match state {
            // Schema/table
            "42S02" => return ErrorCategory::TableValidation, // table doesn't exist
            "42S22" => return ErrorCategory::SchemaValidation, // column not found
            // Privilege / access
            "28000" => return ErrorCategory::DbConnection,     // access denied
            "42000" => {
                let m = msg.to_ascii_lowercase();
                if m.contains("permission") || m.contains("denied") { return ErrorCategory::TableValidation; }
                return ErrorCategory::Configuration; // syntax or access rule violation
            }
            // Connection errors
            "08001" | "08004" | "08S01" => return ErrorCategory::DbConnection,
            // Timeouts
            "HYT00" => return ErrorCategory::ResourceConstraint,
            // Data problems
            "22001" | "22003" | "22007" => return ErrorCategory::DataFormat, // truncation, out of range, invalid datetime
            // Integrity / FK
            "23000" => return ErrorCategory::DataFormat, // integrity constraint violation
            _ => {}
        }
    }
    let m = msg.to_ascii_lowercase();
    if m.contains("access denied") || m.contains("authentication") || m.contains("unknown database") || (m.contains("host") && m.contains("unreach")) || m.contains("timed out") || m.contains("connection") && m.contains("fail") {
        ErrorCategory::DbConnection
    } else if m.contains("doesn't exist") || m.contains("no such table") || (m.contains("table") && m.contains("permission")) {
        ErrorCategory::TableValidation
    } else if m.contains("unknown column") || (m.contains("column") && m.contains("type")) || (m.contains("index") && m.contains("missing")) {
        ErrorCategory::SchemaValidation
    } else if m.contains("incorrect date value") || m.contains("invalid date") || (m.contains("parse") && m.contains("date")) || (m.contains("null") && m.contains("required")) || m.contains("truncation") || m.contains("data too long") || m.contains("foreign key constraint fails") {
        ErrorCategory::DataFormat
    } else if m.contains("out of memory") || (m.contains("memory") && m.contains("insufficient")) || (m.contains("disk") && m.contains("space")) || m.contains("lock wait timeout") {
        ErrorCategory::ResourceConstraint
    } else if (m.contains("invalid") && m.contains("environment")) || (m.contains("env") && m.contains("invalid")) || m.contains("malformed") || m.contains("configuration") || m.contains("syntax error") {
        ErrorCategory::Configuration
    } else {
        ErrorCategory::Unknown
    }
}

fn extract_sqlstate_and_chain(e: &anyhow::Error) -> (Option<String>, String) {
    let sqlstate = e.downcast_ref::<sqlx::Error>().and_then(|se| match se {
        sqlx::Error::Database(db) => db.code().map(|c| c.to_string()),
        _ => None,
    });
    let chain = format!("{:?}", e);
    (sqlstate, chain)
}

struct GuiApp {
    host: String,
    port: String,
    user: String,
    pass: String,
    // Cross-Database (optional)
    enable_dual: bool,
    host2: String,
    port2: String,
    user2: String,
    pass2: String,
    db2: String,
    tables2: Vec<String>,

    db: String,
    tables: Vec<String>,
    table1_idx: usize,
    table2_idx: usize,
    algo: MatchingAlgorithm,
    path: String,
    fmt: FormatSel,
    mode: ModeSel,
    pool_size: String,
    batch_size: String,
    mem_thresh: String,
    use_gpu: bool,
    use_gpu_hash_join: bool,
    // Granular GPU hash-join controls
    use_gpu_build_hash: bool,
    use_gpu_probe_hash: bool,
    // New options
    use_gpu_fuzzy_direct_hash: bool,
    direct_norm_fuzzy: bool,
    gpu_mem_mb: String,           // fuzzy/in-memory GPU mem budget
    gpu_probe_mem_mb: String,     // advisory GPU mem for probe batches

    // Fuzzy threshold (percent 60..100) persisted across sessions
    fuzzy_threshold_pct: i32,

    // Runtime GPU indicators
    gpu_build_active_now: bool,
    gpu_probe_active_now: bool,

    // Storage and system hints
    ssd_storage: bool,

    // Recommendation system UI state
    save_recommendations: bool,
    aggressive_recommendations: bool,
    show_maxperf_dialog: bool,
    maxperf_summary: String,

    running: bool,
    progress: f32,
    eta_secs: u64,
    mem_used: u64,
    mem_avail: u64,
    processed: usize,
    total: usize,
    stage: String,
    batch_current: i64,
    rps: f32,
    last_tick: Option<std::time::Instant>,
    last_processed_prev: usize,
    // GPU status
    gpu_total_mb: u64,
    gpu_free_mb: u64,
    gpu_active: bool,

    a1_count: usize,
    a2_count: usize,
    csv_count: usize,
    status: String,

    // Diagnostics
    error_events: Vec<DiagEvent>,
    report_format: ReportFormat,
    last_action: String,
    // Advanced diagnostics
    schema_analysis_enabled: bool,
    log_buffer: Vec<String>,

    // CUDA diagnostics panel state
    cuda_diag_open: bool,
    cuda_diag_text: String,

    ctrl_cancel: Option<Arc<AtomicBool>>,
    ctrl_pause: Option<Arc<AtomicBool>>,

    tx: Option<Sender<Msg>>, rx: Receiver<Msg>,
}
impl GuiApp {
    fn read_fuzzy_threshold_pref() -> Option<i32> {
        let path = ".nm_fuzzy_threshold";
        match std::fs::read_to_string(path) {
            Ok(s) => {
                let s = s.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<i32>().ok().and_then(|v| if (60..=100).contains(&v) { Some(v) } else { None })
                } else {
                    s.parse::<i32>().ok().and_then(|v| if (60..=100).contains(&v) { Some(v) } else { None })
                }
            }
            Err(_) => None,
        }
    }
    fn save_fuzzy_threshold_pref(&self) {
        let _ = std::fs::write(".nm_fuzzy_threshold", format!("{}%", self.fuzzy_threshold_pct));
    }




    fn compute_cuda_diagnostics() -> String {
        let mut out = String::new();
        out.push_str("CUDA Diagnostics\n");
        #[cfg(feature = "gpu")]
        {
            use std::ffi::CStr;
            use cudarc::driver::sys as cu;
            unsafe {
                let mut init_ok = true;
                let r = cu::cuInit(0);
                if r != cu::CUresult::CUDA_SUCCESS {
                    out.push_str(&format!("cuInit failed: {:?}\n", r));
                    init_ok = false;
                }
                // Driver version
                let mut drv_ver: i32 = 0;
                let r = cu::cuDriverGetVersion(&mut drv_ver as *mut i32);
                if r == cu::CUresult::CUDA_SUCCESS {
                    out.push_str(&format!("Driver Version: {}\n", drv_ver));
                } else { out.push_str(&format!("Driver Version: <error {:?}>\n", r)); }
                // Device count
                let mut count: i32 = 0;
                let r = cu::cuDeviceGetCount(&mut count as *mut i32);
                if r == cu::CUresult::CUDA_SUCCESS {
                    out.push_str(&format!("Device Count: {}\n", count));
                } else { out.push_str(&format!("Device Count: <error {:?}>\n", r)); }
                for i in 0..count {
                    let mut dev: cu::CUdevice = 0;
                    let r = cu::cuDeviceGet(&mut dev as *mut _, i);
                    out.push_str(&format!("\nDevice {}:\n", i));
                    if r != cu::CUresult::CUDA_SUCCESS { out.push_str(&format!("  cuDeviceGet error {:?}\n", r)); continue; }
                    let mut name_buf = [0i8; 256];
                    let r = cu::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, dev);
                    if r == cu::CUresult::CUDA_SUCCESS {
                        let cstr = CStr::from_ptr(name_buf.as_ptr());
                        out.push_str(&format!("  Name: {}\n", cstr.to_string_lossy()));
                    } else { out.push_str(&format!("  Name: <error {:?}>\n", r)); }
                    // Compute capability
                    let mut major: i32 = 0; let mut minor: i32 = 0;
                    let r1 = cu::cuDeviceGetAttribute(&mut major as *mut i32, cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
                    let r2 = cu::cuDeviceGetAttribute(&mut minor as *mut i32, cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
                    if r1 == cu::CUresult::CUDA_SUCCESS && r2 == cu::CUresult::CUDA_SUCCESS {
                        out.push_str(&format!("  Compute Capability: {}.{}\n", major, minor));
                    } else { out.push_str("  Compute Capability: <error>\n"); }
                    // Total memory
                    let mut total_mem: usize = 0;
                    let r = cu::cuDeviceTotalMem_v2(&mut total_mem as *mut usize, dev);
                    if r == cu::CUresult::CUDA_SUCCESS {
                        out.push_str(&format!("  Total Memory: {} MB\n", total_mem / 1024 / 1024));
                    } else { out.push_str(&format!("  Total Memory: <error {:?}>\n", r)); }
                    // Try create context to get free/used
                    if init_ok {
                        if let Ok(ctx) = cudarc::driver::CudaContext::new(i as usize) {
                            let mut free: usize = 0; let mut total: usize = 0;
                            let _ = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
                            let used = total.saturating_sub(free);
                            out.push_str(&format!("  Free: {} MB | Used: {} MB\n", free/1024/1024, used/1024/1024));
                            drop(ctx);
                        } else {
                            out.push_str("  Context: unavailable (cannot create)\n");
                        }
                    }
                }
                out.push_str("\nTroubleshooting:\n - Ensure NVIDIA driver is installed and matches CUDA toolkit version.\n - Reboot after installing drivers.\n - If multiple GPUs, verify CUDA_VISIBLE_DEVICES.\n - On Windows, install latest Studio driver; on Linux, install kernel modules.\n - If this binary lacks GPU support, rebuild with `--features gpu`.\n");
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            out.push_str("This build was compiled without GPU support. Rebuild with `--features gpu`.\n");
        }
        out
    }
}



impl Default for GuiApp {
    fn default() -> Self {
        let (_tx, rx) = mpsc::channel();
        let thr = Self::read_fuzzy_threshold_pref().unwrap_or(95);
        Self {
            host: "127.0.0.1".into(),
            port: "3306".into(),
            user: "root".into(),
            pass: "".into(),
            // dual-db defaults
            enable_dual: false,
            host2: "127.0.0.1".into(),
            port2: "3306".into(),
            user2: "".into(),
            pass2: "".into(),
            db2: "".into(),
            tables2: vec![],
            // primary db
            db: "duplicate_checker".into(),
            tables: vec![],
            table1_idx: 0,
            // CUDA diag defaults
            cuda_diag_open: false,
            cuda_diag_text: String::new(),

            table2_idx: 0,
            algo: MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
            path: "matches.csv".into(),
            fmt: FormatSel::Csv,
            mode: ModeSel::Auto,
            pool_size: "16".into(),
            batch_size: "50000".into(),
            mem_thresh: "800".into(),
            use_gpu: false,
            use_gpu_hash_join: false,
            use_gpu_build_hash: true,
            use_gpu_probe_hash: true,
            // new options
            use_gpu_fuzzy_direct_hash: false,
            direct_norm_fuzzy: false,

            gpu_mem_mb: "512".into(),
            gpu_probe_mem_mb: "256".into(),
            fuzzy_threshold_pct: thr,
            gpu_build_active_now: false,
            gpu_probe_active_now: false,
            ssd_storage: false,
            // New recommendation UI state
            save_recommendations: false,
            aggressive_recommendations: false,
            show_maxperf_dialog: false,
            maxperf_summary: String::new(),
            running: false,
            progress: 0.0,
            eta_secs: 0,
            mem_used: 0,
            mem_avail: 0,
            processed: 0,
            total: 0,
            stage: String::from("idle"),
            batch_current: 0,
            rps: 0.0,
            last_tick: None,
            last_processed_prev: 0,

            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
            a1_count: 0,
            a2_count: 0,
            csv_count: 0,
            status: "Idle".into(),
            // Diagnostics
            error_events: Vec::new(),
            report_format: ReportFormat::Text,
            last_action: "idle".into(),
            schema_analysis_enabled: false,
            log_buffer: Vec::with_capacity(200),

            ctrl_cancel: None,
            ctrl_pause: None,
            tx: None, rx,
        }
    }
}

impl GuiApp {
    fn save_opt_profile(&mut self) {
        let content = format!(
            "pool_size={}\nbatch_size={}\nmem_thresh_mb={}\nuse_gpu={}\nuse_gpu_hash_join={}\nuse_gpu_build_hash={}\nuse_gpu_probe_hash={}\nuse_gpu_fuzzy_direct_hash={}\ndirect_norm_fuzzy={}\ngpu_mem_mb={}\ngpu_probe_mem_mb={}\nssd_storage={}\n",
            self.pool_size,
            self.batch_size,
            self.mem_thresh,
            self.use_gpu,
            self.use_gpu_hash_join,
            self.use_gpu_build_hash,
            self.use_gpu_probe_hash,
            self.use_gpu_fuzzy_direct_hash,
            self.direct_norm_fuzzy,
            self.gpu_mem_mb,
            self.gpu_probe_mem_mb,
            self.ssd_storage,
        );
        match std::fs::write(".nm_opt_profile", content) {
            Ok(_) => { self.status = "Optimization profile saved".into(); }
            Err(e) => { self.status = format!("Failed to save profile: {}", e); }
        }
    }

    fn load_opt_profile_and_apply(&mut self) {
        match std::fs::read_to_string(".nm_opt_profile") {
            Ok(s) => {
                for line in s.lines() {
                    let t = line.trim(); if t.is_empty() || t.starts_with('#') { continue; }
                    if let Some(eq) = t.find('=') {
                        let k = &t[..eq]; let v = t[eq+1..].trim();
                        match k {
                            "pool_size" => { self.pool_size = v.to_string(); }
                            "batch_size" => { self.batch_size = v.to_string(); }
                            "mem_thresh_mb" => { self.mem_thresh = v.to_string(); }
                            "use_gpu" => { self.use_gpu = v.eq_ignore_ascii_case("true") || v=="1"; }
                            "use_gpu_hash_join" => { self.use_gpu_hash_join = v.eq_ignore_ascii_case("true") || v=="1"; }
                            "use_gpu_build_hash" => { self.use_gpu_build_hash = v.eq_ignore_ascii_case("true") || v=="1"; }
                            "use_gpu_probe_hash" => { self.use_gpu_probe_hash = v.eq_ignore_ascii_case("true") || v=="1"; }
                            "use_gpu_fuzzy_direct_hash" => { self.use_gpu_fuzzy_direct_hash = v.eq_ignore_ascii_case("true") || v=="1"; }
                            "direct_norm_fuzzy" => { self.direct_norm_fuzzy = v.eq_ignore_ascii_case("true") || v=="1"; }
                            "gpu_mem_mb" => { self.gpu_mem_mb = v.to_string(); }
                            "gpu_probe_mem_mb" => { self.gpu_probe_mem_mb = v.to_string(); }
                            "ssd_storage" => { self.ssd_storage = v.eq_ignore_ascii_case("true") || v=="1"; }
                            _ => {}
                        }
                    }
                }
                self.status = "Optimization profile loaded".into();
            }
            Err(_) => { self.status = "No optimization profile found".into(); }
        }
    }

    fn ui_top(&mut self, ui: &mut egui::Ui) {
        ui.heading("🔎 SRS-II Name Matching Application");
        ui.separator();
        ui.label("Database Connection");
        ui.add_space(6.0);
        ui.label("Database 1 (Primary)");
        egui::Grid::new("db1_grid").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Host:");      ui.add(TextEdit::singleline(&mut self.host).hint_text("host")); ui.end_row();
            ui.label("Port:");      ui.add(TextEdit::singleline(&mut self.port).hint_text("port")); ui.end_row();
            ui.label("Username:");  ui.add(TextEdit::singleline(&mut self.user).hint_text("user")); ui.end_row();
            ui.label("Password:");  ui.add(TextEdit::singleline(&mut self.pass).hint_text("password").password(true)); ui.end_row();
            ui.label("Database:");  ui.add(TextEdit::singleline(&mut self.db).hint_text("database")); ui.end_row();
        });
        ui.add_space(6.0);
        ui.checkbox(&mut self.enable_dual, "Enable Cross-Database Matching")
            .on_hover_text("Match Table 1 from Database 1 with Table 2 from Database 2");
        if self.enable_dual {
            ui.add_space(6.0);
            ui.label("Database 2 (Secondary)");
            egui::Grid::new("db2_grid").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("Host:");      ui.add(TextEdit::singleline(&mut self.host2).hint_text("host")); ui.end_row();
                ui.label("Port:");      ui.add(TextEdit::singleline(&mut self.port2).hint_text("port")); ui.end_row();
                ui.label("Username:");  ui.add(TextEdit::singleline(&mut self.user2).hint_text("user")); ui.end_row();
                ui.label("Password:");  ui.add(TextEdit::singleline(&mut self.pass2).hint_text("password").password(true)); ui.end_row();
                ui.label("Database:");  ui.add(TextEdit::singleline(&mut self.db2).hint_text("database")); ui.end_row();
            });
        }

        // Action toolbar (compact grid)
        egui::Grid::new("action_toolbar").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
            ui.label("Database Actions:");
            ui.horizontal_wrapped(|ui| {
                if ui.button("Load Tables").on_hover_text("Query INFORMATION_SCHEMA to list tables").clicked() { self.load_tables(); }
                if ui.button("Test Connection").on_hover_text("Checks DB connectivity using a lightweight query").clicked() { self.test_connection(); }
                if ui.button("Estimate").on_hover_text("Estimate memory usage and choose a good mode").clicked() { self.estimate(); }
            });
            ui.end_row();
            ui.label("Configuration:");
            ui.horizontal_wrapped(|ui| {
                if ui.button("Generate .env Template").on_hover_text("Create a .env.template file with configurable keys").clicked() {
                    let dialog = rfd::FileDialog::new()
                        .add_filter("Template", &["template","env","txt"])
                        .set_file_name(".env.template");
                    if let Some(path) = dialog.save_file() {
                        match name_matcher::util::envfile::write_env_template(&path.display().to_string()) {
                            Ok(_) => { self.status = format!(".env template saved to {}", path.display()); },
                            Err(e) => { self.status = format!("Failed to save .env template: {}", e); }
                        }
                    }
                }
                if ui.button("Load .env File...").on_hover_text("Load variables from a .env file and prefill fields").clicked() {
                    if let Some(path) = rfd::FileDialog::new().add_filter("Env", &["env","txt","template"]).pick_file() {
                        match name_matcher::util::envfile::load_env_file_from(&path.display().to_string()) {
                            Ok(map) => {
                                if let Some(v) = map.get("DB_HOST") { self.host = v.clone(); }
                                if let Some(v) = map.get("DB_PORT") { self.port = v.clone(); }
                                if let Some(v) = map.get("DB_USER") { self.user = v.clone(); }
                                if let Some(v) = map.get("DB_PASSWORD") { self.pass = v.clone(); }
                                if let Some(v) = map.get("DB_NAME") { self.db = v.clone(); }
                                if let Some(v) = map.get("DB2_HOST") { self.host2 = v.clone(); self.enable_dual = true; }
                                if let Some(v) = map.get("DB2_PORT") { self.port2 = v.clone(); }
                                if let Some(v) = map.get("DB2_USER") { self.user2 = v.clone(); }
                                if let Some(v) = map.get("DB2_PASS") { self.pass2 = v.clone(); }
                                if let Some(v) = map.get("DB2_DATABASE") { self.db2 = v.clone(); self.enable_dual = true; }
                                self.status = format!("Loaded .env from {}", path.display());
                            }
                            Err(e) => { self.status = format!("Failed to load .env: {}", e); }
                        }
                    }
                }
            });
            ui.end_row();
        });

        ui.add_space(6.0);
        // Table selection and mode
        ui.horizontal_wrapped(|ui| {
            if self.enable_dual {
                // Dual DB: pick table1 from DB1 list, table2 from DB2 list
                if self.tables.is_empty() { ui.label("(Load DB1 tables)"); }
                if self.tables2.is_empty() { ui.label("(Load DB2 tables)"); }
                if !self.tables.is_empty() {
                    ComboBox::from_label("Table 1 (DB1)")
                        .selected_text(self.tables.get(self.table1_idx).cloned().unwrap_or_default())
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables.iter().enumerate() { ui.selectable_value(&mut self.table1_idx, i, t); }
                        });
                }
                if !self.tables2.is_empty() {
                    ComboBox::from_label("Table 2 (DB2)")
                        .selected_text(self.tables2.get(self.table2_idx).cloned().unwrap_or_default())
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables2.iter().enumerate() { ui.selectable_value(&mut self.table2_idx, i, t); }
                        });
                }
            } else {
                if !self.tables.is_empty() {
                    ComboBox::from_label("Table 1")
                        .selected_text(self.tables.get(self.table1_idx).cloned().unwrap_or_default())
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables.iter().enumerate() { ui.selectable_value(&mut self.table1_idx, i, t); }
                        });
                    ComboBox::from_label("Table 2")
                        .selected_text(self.tables.get(self.table2_idx).cloned().unwrap_or_default())
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables.iter().enumerate() { ui.selectable_value(&mut self.table2_idx, i, t); }
                        });
                } else {
                    ui.label("(Load tables to choose)");
                }

            }
            ComboBox::from_label("Mode")
                .selected_text(match self.mode { ModeSel::Auto=>"Auto", ModeSel::Streaming=>"Streaming", ModeSel::InMemory=>"In-memory" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.mode, ModeSel::Auto, "Auto").on_hover_text("Let the app choose based on data size");
                    ui.selectable_value(&mut self.mode, ModeSel::Streaming, "Streaming").on_hover_text("Index small table, stream large table");
                    ui.selectable_value(&mut self.mode, ModeSel::InMemory, "In-memory").on_hover_text("Load both tables into memory first");
                });
        });

        ui.add_space(8.0);
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, "Algorithm 1 (first+last+birthdate)");
            ui.radio_value(&mut self.algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, "Algorithm 2 (first+middle+last+birthdate)");
            ui.radio_value(&mut self.algo, MatchingAlgorithm::Fuzzy, "Fuzzy (Levenshtein/Jaro/Winkler; birthdate must match)")
                .on_hover_text("Ensemble of Levenshtein, Jaro, Jaro-Winkler; use the slider below to set the acceptance threshold.");
            ui.radio_value(&mut self.algo, MatchingAlgorithm::FuzzyNoMiddle, "Fuzzy (first+last only; birthdate must match)")
                .on_hover_text("Fuzzy ensemble excluding middle name. Uses first+last and requires exact birthdate equality.");
            ui.radio_value(&mut self.algo, MatchingAlgorithm::HouseholdGpu, "Household GPU (in-memory)")
                .on_hover_text("GPU-accelerated household matching: exact birthdate required, names fuzzy per slider; household kept if >50% members match.");
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.direct_norm_fuzzy, "Apply Fuzzy-style normalization to Algorithms 1 & 2")
                .on_hover_text("Lowercases and strips punctuation; treats hyphens as spaces. Aligns A1/A2 normalization with Fuzzy when checked.");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Fuzzy threshold");
            let fuzzy_enabled = matches!(self.algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu);
            ui.add_enabled(fuzzy_enabled, egui::Slider::new(&mut self.fuzzy_threshold_pct, 60..=100).suffix(" %"));
            ui.label(format!("{}%", self.fuzzy_threshold_pct));
            if !fuzzy_enabled { ui.label("(disabled for Algorithms 1 & 2)"); }
        });
        ui.add_space(6.0);
        ui.separator();

        ui.horizontal_wrapped(|ui| {
            ui.add(TextEdit::singleline(&mut self.path).hint_text("output file path"));
            if ui.button("Browse").clicked() {
                let mut dialog = rfd::FileDialog::new();
                match self.fmt {
                    FormatSel::Csv => { dialog = dialog.add_filter("CSV", &["csv"]); }
                    FormatSel::Xlsx => { dialog = dialog.add_filter("Excel", &["xlsx"]); }
                    FormatSel::Both => { dialog = dialog.add_filter("CSV", &["csv"]).add_filter("Excel", &["xlsx"]); }
                }
                if let Some(path) = dialog.set_file_name(&self.path).save_file() { self.path = path.display().to_string(); }
            }
            ComboBox::from_label("Format")
                .selected_text(match self.fmt { FormatSel::Csv=>"CSV", FormatSel::Xlsx=>"XLSX", FormatSel::Both=>"Both" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.fmt, FormatSel::Csv, "CSV");
                    ui.selectable_value(&mut self.fmt, FormatSel::Xlsx, "XLSX");
                    ui.selectable_value(&mut self.fmt, FormatSel::Both, "Both");
                });
        });
        ui.add_space(8.0);
        ui.separator();
        ui.group(|ui| {
            ui.heading("GPU Hash Join (Algorithms 1 & 2)");
            ui.checkbox(&mut self.use_gpu_hash_join, "Enable GPU Hash Join (A1/A2)")
                .on_hover_text("Accelerate deterministic joins via GPU pre-hash + CPU verify; requires CUDA build; falls back automatically.");
            if self.use_gpu_hash_join {
                egui::Grid::new("gpu_hash_join_grid").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
                    ui.label("GPU for Build Hash:"); ui.checkbox(&mut self.use_gpu_build_hash, ""); ui.end_row();
                    ui.label("GPU for Probe Hash:"); ui.checkbox(&mut self.use_gpu_probe_hash, ""); ui.end_row();
                ui.checkbox(&mut self.use_gpu_fuzzy_direct_hash, "GPU pre-pass for Fuzzy direct matching")
                    .on_hover_text("Use GPU hash filter to reduce Fuzzy candidate pairs before scoring. Exact birthdate + optional last-initial blocking; behavior preserved; CPU fallback.");

                    ui.label("Probe GPU Mem (MB):"); ui.add(TextEdit::singleline(&mut self.gpu_probe_mem_mb).desired_width(80.0)); ui.end_row();
                });
                if self.gpu_total_mb > 0 {
                    ui.label(format!("Status: Build {} | Probe {}",
                        if self.gpu_build_active_now { "active" } else { "idle" },
                        if self.gpu_probe_active_now { "active" } else { "idle" }
                    ));
                }
            }
        });
        ui.add_space(8.0);
        ui.separator();
        ui.collapsing("Advanced", |ui| {

            ui.add_space(4.0);
            ui.collapsing("Performance & Streaming", |ui| {
                let streaming_enabled = !matches!(self.mode, ModeSel::InMemory);
                egui::Grid::new("perf_stream_grid").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Pool size");
                    ui.add(TextEdit::singleline(&mut self.pool_size).desired_width(60.0)).on_hover_text("Max connections in SQL pool");
                    ui.end_row();

                    ui.label("Batch size");
                    let resp_batch = ui.add_enabled(streaming_enabled, TextEdit::singleline(&mut self.batch_size).desired_width(80.0));
                    if !streaming_enabled { resp_batch.on_hover_text("Batch size and memory threshold apply only in Streaming mode. In-Memory mode loads the entire dataset into RAM at once, making these settings irrelevant."); }
                    else { resp_batch.on_hover_text("Rows fetched per chunk in streaming mode"); }
                    ui.end_row();

                    ui.label("Mem thresh MB");
                    let resp_mem = ui.add_enabled(streaming_enabled, TextEdit::singleline(&mut self.mem_thresh).desired_width(80.0));
                    if !streaming_enabled { resp_mem.on_hover_text("Batch size and memory threshold apply only in Streaming mode. In-Memory mode loads the entire dataset into RAM at once, making these settings irrelevant."); }
                    else { resp_mem.on_hover_text("Soft minimum free memory before reducing batch size"); }
                    ui.end_row();

                    ui.label("Storage");
                    ui.checkbox(&mut self.ssd_storage, "SSD storage").on_hover_text("Optimize flush frequency for SSD (larger buffered writes)");
                    ui.end_row();
                });
            });

            ui.add_space(6.0);
            ui.collapsing("GPU Acceleration", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.use_gpu, "Use GPU (CUDA)").on_hover_text("Enable CUDA acceleration for Fuzzy (Algorithm 3). Falls back to CPU if unavailable.");
                    if !cfg!(feature = "gpu") { ui.label("Note: This build was compiled without GPU feature; enable with `cargo run --features gpu`."); }
                });
                egui::Grid::new("gpu_accel_grid").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("GPU Mem Budget (MB)"); ui.add(TextEdit::singleline(&mut self.gpu_mem_mb).desired_width(80.0)); ui.end_row();
                });
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Auto Optimize").on_hover_text("Detect hardware and set safe, high-performance parameters").clicked() {
                        let mem = name_matcher::metrics::memory_stats_mb();
                        let total = mem.total_mb; let avail = mem.avail_mb;
                        // Set streaming safety threshold: 15-20% of total RAM (min 1GB). Use 18% midpoint.
                        let mem_soft_min = ((total as f64 * 0.18) as u64).max(1024);
                        self.mem_thresh = mem_soft_min.to_string();
                        // Dynamic batch size: target ~75% of available RAM -> approximate rows by /4 heuristic
                        let target_batch_mem_mb = ((avail as f64) * 0.75) as u64;
                        let batch = ((target_batch_mem_mb as i64) / 4).clamp(10_000, 200_000);
                        self.batch_size = batch.to_string();
                        // Thread/Pool sizing: 2-4x cores scaled by total RAM
                        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) as u32;
                        let pool_suggestion = if total >= 32_768 { (cores.saturating_mul(4)).min(64) } else if total >= 16_384 { (cores.saturating_mul(3)).min(48) } else { (cores.saturating_mul(2)).min(32) };
                        self.pool_size = pool_suggestion.to_string();
                        // Prefer mode based on algorithm
                        self.mode = if matches!(self.algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu) { ModeSel::InMemory } else { ModeSel::Streaming };
                        // GPU VRAM budgeting based on currently available VRAM
                        #[cfg(feature = "gpu")]
                        {
                            use cudarc::driver::sys as cu;
                            unsafe {
                                let init_rc = cu::cuInit(0);
                                if init_rc == cu::CUresult::CUDA_SUCCESS {
                                    // cuMemGetInfo_v2 needs a current context; create one like in diagnostics
                                    if let Ok(ctx) = cudarc::driver::CudaContext::new(0) {
                                        let mut free_b: usize = 0; let mut total_b: usize = 0;
                                        let rc = cu::cuMemGetInfo_v2(&mut free_b as *mut usize, &mut total_b as *mut usize);
                                        drop(ctx);
                                        if rc == cu::CUresult::CUDA_SUCCESS && total_b > 0 {
                                            let free_mb = (free_b as u64) / (1024 * 1024) as u64;
                                            let total_mb = (total_b as u64) / (1024 * 1024) as u64;
                                            self.gpu_total_mb = total_mb; self.gpu_free_mb = free_mb;
                                            // Reserve 256-512MB; then take ~80% of the remainder
                                            let reserve_mb = if free_mb >= 4096 { 512 } else { 256 };
                                            let mut budget = ((free_mb.saturating_sub(reserve_mb)) as f64 * 0.80) as u64;
                                            budget = budget.clamp(512, total_mb.saturating_sub(128));
                                            self.gpu_mem_mb = budget.to_string();
                                            // Probe is 25-40% of build; use ~30%
                                            let probe = ((budget as f64) * 0.30) as u64;
                                            self.gpu_probe_mem_mb = probe.clamp(256, 4096).to_string();
                                            self.status = format!("Auto-Optimized | RAM tot {} MB, avail {} MB | Batch {} | SoftMin {} MB | Cores {} -> Pool {} | GPU free {} / tot {} MB | GPU budget {} MB, probe {} MB",
                                                total, avail, batch, mem_soft_min, cores, pool_suggestion, free_mb, total_mb, budget, probe);
                                        } else {
                                            self.status = format!("Auto-Optimized (CUDA mem info error {:?}) | RAM tot {} MB, avail {} MB | Batch {} | SoftMin {} MB | Cores {} -> Pool {}",
                                                rc, total, avail, batch, mem_soft_min, cores, pool_suggestion);
                                        }
                                    } else {
                                        self.status = format!("Auto-Optimized (CUDA context unavailable) | RAM tot {} MB, avail {} MB | Batch {} | SoftMin {} MB | Cores {} -> Pool {}",
                                            total, avail, batch, mem_soft_min, cores, pool_suggestion);
                                    }
                                } else {
                                    self.status = format!("Auto-Optimized (cuInit error {:?}) | RAM tot {} MB, avail {} MB | Batch {} | SoftMin {} MB | Cores {} -> Pool {}",
                                        init_rc, total, avail, batch, mem_soft_min, cores, pool_suggestion);
                                }
                            }
                        }
                        #[cfg(not(feature = "gpu"))]
                        {
                            self.status = format!("Auto-Optimized (CPU-only build) | RAM tot {} MB, avail {} MB | Batch {} | SoftMin {} MB | Cores {} -> Pool {}",
                                total, avail, batch, mem_soft_min, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8), self.pool_size);
                        }
                    }
                    if ui.button("Save Profile").on_hover_text("Save current optimization settings").clicked() { self.save_opt_profile(); }
                    if ui.button("Load Profile").on_hover_text("Load saved optimization settings").clicked() { self.load_opt_profile_and_apply(); }

                    ui.separator();
                    ui.checkbox(&mut self.save_recommendations, "Save Recommendations").on_hover_text("When enabled, exports a timestamped CSV of applied settings");
                    ui.checkbox(&mut self.aggressive_recommendations, "Aggressive").on_hover_text("Aggressive = higher throughput, lower safety margins; Conservative = safer defaults");

                    if ui.button("Max Performance Settings").on_hover_text("Analyze hardware and apply maximum safe performance settings").clicked() {
                        // Compute and apply inline to avoid nested methods inside impl scopes
                        let mem = name_matcher::metrics::memory_stats_mb();
                        let total = mem.total_mb; let avail = mem.avail_mb;
                        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) as u32;
                        let pool = if self.aggressive_recommendations { (cores.saturating_mul(6)).min(96) } else { (cores.saturating_mul(4)).min(64) };
                        self.pool_size = pool.to_string();
                        let target_frac = if self.aggressive_recommendations { 0.85 } else { 0.75 };
                        let target_batch_mem_mb = ((avail as f64) * target_frac) as u64;
                        let batch = ((target_batch_mem_mb as i64) / 4).clamp(10_000, 200_000);
                        self.batch_size = batch.to_string();
                        let soft_min = if self.aggressive_recommendations { ((total as f64 * 0.12) as u64).max(512) } else { ((total as f64 * 0.18) as u64).max(1024) };
                        self.mem_thresh = soft_min.to_string();
                        self.mode = if matches!(self.algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu) { ModeSel::InMemory } else { ModeSel::Streaming };
                        let mut gpu_line = String::from("GPU: not available (CPU-only build)");
                        #[cfg(feature = "gpu")]
                        {
                            use cudarc::driver::sys as cu;
                            unsafe {
                                let init_rc = cu::cuInit(0);
                                if init_rc == cu::CUresult::CUDA_SUCCESS {
                                    if let Ok(ctx) = cudarc::driver::CudaContext::new(0) {
                                        let mut free_b: usize = 0; let mut total_b: usize = 0;
                                        let rc = cu::cuMemGetInfo_v2(&mut free_b as *mut usize, &mut total_b as *mut usize);
                                        if rc == cu::CUresult::CUDA_SUCCESS && total_b > 0 {
                                            self.use_gpu = true;
                                            let free_mb = (free_b as u64) / (1024 * 1024) as u64;
                                            let total_mb = (total_b as u64) / (1024 * 1024) as u64;
                                            self.gpu_total_mb = total_mb; self.gpu_free_mb = free_mb;
                                            let reserve_mb = if free_mb >= 4096 { 512 } else { 256 };
                                            let mut budget = ((free_mb.saturating_sub(reserve_mb)) as f64 * if self.aggressive_recommendations { 0.90 } else { 0.80 }) as u64;
                                            budget = budget.clamp(512, total_mb.saturating_sub(128));
                                            self.gpu_mem_mb = budget.to_string();
                                            let probe = ((budget as f64) * if self.aggressive_recommendations { 0.35 } else { 0.30 }) as u64;
                                            self.gpu_probe_mem_mb = probe.clamp(256, 8192).to_string();
                                            gpu_line = format!("GPU: free {} / total {} MB | budget {} MB | probe {} MB", free_mb, total_mb, budget, probe);
                                        }
                                        drop(ctx);
                                    }
                                }
                            }
                        }
                        let mode_text = match self.mode { ModeSel::Auto=>"Auto", ModeSel::Streaming=>"Streaming", ModeSel::InMemory=>"In-Memory" };
                        self.maxperf_summary = format!(
                            "Applied Max Performance Settings\n- Cores: {}\n- RAM total: {} MB, avail: {} MB\n- Pool size: {}\n- Batch size (streaming): {}\n- Mem threshold: {} MB\n- Mode: {}\n- {}",
                            cores, total, avail, self.pool_size, self.batch_size, self.mem_thresh, mode_text, gpu_line
                        );
                        self.show_maxperf_dialog = true;
                        if self.save_recommendations {
                            let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
                            let filename = format!("name_matcher_recommended_settings_{}.csv", ts);
                            let mode_text = match self.mode { ModeSel::Auto=>"Auto", ModeSel::Streaming=>"Streaming", ModeSel::InMemory=>"InMemory" };
                            let mut csv = String::from("setting,value,explanation\n");
                            csv.push_str(&format!("pool_size,{},\"Max DB connections in pool (higher for more concurrency; capped)\"\n", self.pool_size));
                            csv.push_str(&format!("batch_size,{},\"Rows per streaming chunk; ignored in In-Memory mode\"\n", self.batch_size));
                            csv.push_str(&format!("mem_thresh_mb,{},\"Min free RAM before streaming reduces batch size\"\n", self.mem_thresh));
                            csv.push_str(&format!("use_gpu,{},\"Enable CUDA acceleration when available\"\n", if self.use_gpu {"true"} else {"false"}));
                            csv.push_str(&format!("use_gpu_hash_join,{},\"GPU hashing for A1/A2 (streaming)\"\n", if self.use_gpu_hash_join {"true"} else {"false"}));
                            csv.push_str(&format!("use_gpu_build_hash,{},\"Use GPU for build-side hashing (A1/A2)\"\n", if self.use_gpu_build_hash {"true"} else {"false"}));
                            csv.push_str(&format!("use_gpu_probe_hash,{},\"Use GPU for probe-side hashing (A1/A2)\"\n", if self.use_gpu_probe_hash {"true"} else {"false"}));
                            csv.push_str(&format!("gpu_mem_mb,{},\"GPU memory budget for kernels\"\n", self.gpu_mem_mb));
                            csv.push_str(&format!("gpu_probe_mem_mb,{},\"GPU memory target for probe hashing batches\"\n", self.gpu_probe_mem_mb));
                            csv.push_str(&format!("mode,{},\"Execution mode\"\n", mode_text));
                            match std::fs::write(&filename, csv) {
                                Ok(_) => { self.status = format!("Exported recommendations to {}", filename); }
                                Err(e) => { self.status = format!("Failed to export recommendations: {}", e); }
                            }
                        }
                    }

                    if ui.button("Import Settings").on_hover_text("Load settings from a previously exported CSV file").clicked() {
                        if let Some(path) = rfd::FileDialog::new().add_filter("CSV", &["csv"]).pick_file() {
                            match std::fs::read_to_string(&path) {
                                Ok(s) => {
                                    for (i, line) in s.lines().enumerate() {
                                        if i==0 { continue; } // skip header
                                        let mut parts = line.splitn(3, ',');
                                        let key = parts.next().unwrap_or("").trim().to_ascii_lowercase();
                                        let val = parts.next().unwrap_or("").trim().to_string();
                                        match key.as_str() {
                                            "pool_size" => self.pool_size = val,
                                            "batch_size" => self.batch_size = val,
                                            "mem_thresh_mb" => self.mem_thresh = val,
                                            "use_gpu" => self.use_gpu = val.eq_ignore_ascii_case("true") || val=="1",
                                            "use_gpu_hash_join" => self.use_gpu_hash_join = val.eq_ignore_ascii_case("true") || val=="1",
                                            "use_gpu_build_hash" => self.use_gpu_build_hash = val.eq_ignore_ascii_case("true") || val=="1",
                                            "use_gpu_probe_hash" => self.use_gpu_probe_hash = val.eq_ignore_ascii_case("true") || val=="1",
                                            "gpu_mem_mb" => self.gpu_mem_mb = val,
                                            "gpu_probe_mem_mb" => self.gpu_probe_mem_mb = val,
                                            "mode" => {
                                                self.mode = match val.as_str() { "Auto"|"auto"=>ModeSel::Auto, "Streaming"|"streaming"=>ModeSel::Streaming, _=>ModeSel::InMemory };
                                            }
                                            _ => {}
                                        }
                                    }
                                    self.status = format!("Imported settings from {}", path.to_string_lossy());
                                }
                                Err(e) => { self.status = format!("Failed to import: {}", e); }
                            }
                        } else {
                            self.status = "Import cancelled".into();
                        }
                    }

                    if self.gpu_total_mb > 0 { ui.label(format!("GPU: {} MB free / {} MB total | {}", self.gpu_free_mb, self.gpu_total_mb, if self.gpu_active { "active" } else { "idle" })); }
                });
            });

            ui.add_space(6.0);
            ui.collapsing("System Information", |ui| {
                let mem = name_matcher::metrics::memory_stats_mb();
                let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("CUDA Diagnostics").on_hover_text("Show CUDA devices, versions, and memory info").clicked() {
                        self.cuda_diag_text = Self::compute_cuda_diagnostics();
                        self.cuda_diag_open = true;
                    }
                    ui.label(format!("System: {} cores | free mem {} MB", cores, mem.avail_mb));
                });
            });
        });
        ui.separator();
        if self.running {
            ui.label(&self.status);
            ui.add(ProgressBar::new(self.progress/100.0).text(format!("{:.1}% | ETA {}s | Used {} MB | Avail {} MB", self.progress, self.eta_secs, self.mem_used, self.mem_avail)));
            ui.label(format!("Stage: {} | Records: {}/{} | Batch: {} | Throughput: {:.0} rec/s", self.stage, self.processed, self.total, self.batch_current, self.rps));
        } else {

            ui.label(&self.status);
        }
        ui.add_space(6.0);
        ui.separator();

        ui.horizontal_wrapped(|ui| {
            if !self.running {

                if ui.button("Start").on_hover_text("Run matching with selected mode and format").clicked() { self.start(); }
            } else {
                if let Some(p) = self.ctrl_pause.as_ref() {
                    let paused = p.load(Ordering::Relaxed);
                    if ui.button(if paused { "Resume" } else { "Pause" }).clicked() { p.store(!paused, Ordering::Relaxed); }
                }
                if let Some(c) = self.ctrl_cancel.as_ref() {
                    if ui.button("Cancel").clicked() { c.store(true, Ordering::Relaxed); }
                }
            }
            if ui.button("Reset").on_hover_text("Clear the form and state").clicked() { self.reset_state(); }
            if !self.error_events.is_empty() {
                ui.separator();
                ComboBox::from_label("Report Format")
                    .selected_text(match self.report_format { ReportFormat::Text=>"TXT", ReportFormat::Json=>"JSON" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.report_format, ReportFormat::Text, "TXT");
                        ui.selectable_value(&mut self.report_format, ReportFormat::Json, "JSON");
                    });
                ui.checkbox(&mut self.schema_analysis_enabled, "Include schema analysis in report")
                    .on_hover_text("Runs INFORMATION_SCHEMA queries to suggest missing columns/types/indexes; metadata-only, no row data");
                if ui.button("Export Error Report").on_hover_text("Export a sanitized diagnostic report for technical support").clicked() {
                    match self.export_error_report() {
                        Ok(p) => { self.status = format!("Error report saved to {}", p); }
                        Err(e) => { self.status = format!("Failed to export report: {}", e); }
                    }
                }
                ui.label(format!("Errors captured: {} (log tail {} entries)", self.error_events.len(), self.log_buffer.len()));
            }
        });

        ui.separator();
        ui.label(format!("Results: Algo1={} | Algo2={} | CSV={}  @ {}", self.a1_count, self.a2_count, self.csv_count, Utc::now()));
    }

    fn validate(&self) -> Result<()> {
        if self.host.trim().is_empty() { anyhow::bail!("Host is required"); }
        if self.port.parse::<u16>().is_err() { anyhow::bail!("Port must be a number"); }
        if self.user.trim().is_empty() { anyhow::bail!("Username is required"); }
        if self.db.trim().is_empty() { anyhow::bail!("Database is required"); }
        if self.path.trim().is_empty() { anyhow::bail!("Output path is required"); }
        if self.enable_dual {
            if self.host2.trim().is_empty() { anyhow::bail!("DB2 host is required"); }
            if self.port2.parse::<u16>().is_err() { anyhow::bail!("DB2 port must be a number"); }
            if self.user2.trim().is_empty() { anyhow::bail!("DB2 username is required"); }
            if self.db2.trim().is_empty() { anyhow::bail!("DB2 database is required"); }
            if self.tables.is_empty() { anyhow::bail!("Please load DB1 tables and select Table 1"); }
            if self.tables2.is_empty() { anyhow::bail!("Please load DB2 tables and select Table 2"); }
        } else {
            if self.tables.is_empty() { anyhow::bail!("Please load tables and select Table 1/2"); }
        }
        Ok(())
    }

    fn load_tables(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
        self.last_action = "Load Tables".into();
        let host = self.host.clone(); let port = self.port.clone(); let user = self.user.clone(); let pass = self.pass.clone(); let dbname = self.db.clone();
        let enable_dual = self.enable_dual;
        let host2 = self.host2.clone(); let port2 = self.port2.clone(); let user2 = self.user2.clone(); let pass2 = self.pass2.clone(); let dbname2 = self.db2.clone();
        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            if enable_dual {
                let res: Result<(Vec<String>, Vec<String>)> = rt.block_on(async move {
                    // DB1
                    let cfg1 = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname.clone() };
                    let pool1 = make_pool_with_size(&cfg1, Some(8)).await?;
                    let rows1 = sqlx::query_scalar::<_, String>("SELECT CAST(TABLE_NAME AS CHAR) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")
                        .bind(dbname).fetch_all(&pool1).await?;
                    // DB2
                    let cfg2 = DatabaseConfig { host: host2, port: port2.parse().unwrap_or(3306), username: user2, password: pass2, database: dbname2.clone() };
                    let pool2 = make_pool_with_size(&cfg2, Some(8)).await?;
                    let rows2 = sqlx::query_scalar::<_, String>("SELECT CAST(TABLE_NAME AS CHAR) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")
                        .bind(dbname2).fetch_all(&pool2).await?;
                    Ok((rows1, rows2))
                });
                match res {
                    Ok((t1, t2)) => { let _ = tx.send(Msg::Info(format!("Loaded DB1: {} tables | DB2: {} tables", t1.len(), t2.len()))); let _ = tx.send(Msg::Tables(t1)); let _ = tx.send(Msg::Tables2(t2)); },
                    Err(e) => { let (sqlstate, chain) = extract_sqlstate_and_chain(&e); let _ = tx.send(Msg::ErrorRich { display: format!("Failed to load tables: {}", e), sqlstate, chain, operation: Some("Load Tables (DB1+DB2)".into()) }); }
                }
            } else {
                let res: Result<Vec<String>> = rt.block_on(async move {
                    let cfg = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname.clone() };
                    let pool = make_pool_with_size(&cfg, Some(8)).await?;
                    let rows = sqlx::query_scalar::<_, String>("SELECT CAST(TABLE_NAME AS CHAR) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")
                        .bind(dbname).fetch_all(&pool).await?;
                    Ok(rows)
                });
                match res { Ok(tables) => { let _ = tx.send(Msg::Info(format!("Loaded {} tables", tables.len()))); let _ = tx.send(Msg::Tables(tables)); }, Err(e) => { let (sqlstate, chain) = extract_sqlstate_and_chain(&e); let _ = tx.send(Msg::ErrorRich { display: format!("Failed to load tables: {}", e), sqlstate, chain, operation: Some("Load Tables (DB1)".into()) }); } }
            }
        });
        self.rx = rx; // switch to listen on this job
        self.status = "Loading tables...".into();
    }

    fn start(&mut self) {
        if let Err(e) = self.validate() { self.status = format!("Error: {}", e); return; }
        self.running = true; self.progress = 0.0; self.status = "Running...".into(); self.a1_count = 0; self.a2_count = 0; self.csv_count = 0;
        // Persist current fuzzy threshold selection
        self.save_fuzzy_threshold_pref();

        self.gpu_build_active_now = false; self.gpu_probe_active_now = false;
        self.last_action = "Run Matching".into();
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone()); self.rx = rx;

        let cfg1 = DatabaseConfig { host: self.host.clone(), port: self.port.parse().unwrap_or(3306), username: self.user.clone(), password: self.pass.clone(), database: self.db.clone() };
        let enable_dual = self.enable_dual;
        let cfg2 = if enable_dual {
            Some(DatabaseConfig { host: self.host2.clone(), port: self.port2.parse().unwrap_or(3306), username: self.user2.clone(), password: self.pass2.clone(), database: self.db2.clone() })
        } else { None };
        let table1 = self.tables.get(self.table1_idx).cloned().unwrap_or_default();
        let table2 = if enable_dual {
            self.tables2.get(self.table2_idx).cloned().unwrap_or_default()
        } else {
            self.tables.get(self.table2_idx).cloned().unwrap_or_default()
        };
        let algo = self.algo; let path = self.path.clone(); let fmt = self.fmt; let mode = self.mode;
        let use_gpu = self.use_gpu; let gpu_mem = self.gpu_mem_mb.parse::<u64>().unwrap_or(512);
        let use_gpu_hash_join = self.use_gpu_hash_join;
        let use_gpu_build_hash = self.use_gpu_build_hash;
        let use_gpu_probe_hash = self.use_gpu_probe_hash;
        let gpu_probe_mem_mb_val = self.gpu_probe_mem_mb.parse::<u64>().unwrap_or(256);

        let use_gpu_fuzzy_direct_hash = self.use_gpu_fuzzy_direct_hash;
        let direct_norm_fuzzy = self.direct_norm_fuzzy;

        let pool_sz = self.pool_size.parse::<u32>().unwrap_or(16);
        let batch = self.batch_size.parse::<i64>().unwrap_or(50_000);

        let fuzzy_thr: f32 = (self.fuzzy_threshold_pct as f32) / 100.0;

        if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) && !matches!(fmt, FormatSel::Csv) {
            self.running = false;
            self.status = "Fuzzy algorithm supports CSV format only. Please select CSV.".into();
            return;
        }

        let mem_thr = self.mem_thresh.parse::<u64>().unwrap_or(800);

        // create control flags for pause/cancel
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let pause_flag = Arc::new(AtomicBool::new(false));
        self.ctrl_cancel = Some(cancel_flag.clone());
        self.ctrl_pause = Some(pause_flag.clone());

        let ssd_hint = self.ssd_storage;
        thread::spawn(move || {
            let tx_for_async = tx.clone();
            let ctrl = Some(StreamControl { cancel: cancel_flag.clone(), pause: pause_flag.clone() });


            let rt = tokio::runtime::Runtime::new().unwrap();
            let res: Result<(usize,usize,usize,String)> = rt.block_on(async move {
                let (pool1, pool2_opt) = if enable_dual {
                    let p1 = make_pool_with_size(&cfg1, Some(pool_sz)).await?;
                    let p2 = make_pool_with_size(cfg2.as_ref().unwrap(), Some(pool_sz)).await?;
                    (p1, Some(p2))

                } else {
                    (make_pool_with_size(&cfg1, Some(pool_sz)).await?, None)
                };
                let mut scfg = StreamingConfig { batch_size: batch, memory_soft_min_mb: mem_thr, ..Default::default() };
                // progressive saving/resume: write checkpoint next to output file
                let db_label = if enable_dual { format!("{} | {}", cfg1.database, cfg2.as_ref().unwrap().database) } else { cfg1.database.clone() };
                scfg.use_gpu_hash_join = use_gpu_hash_join;
                scfg.use_gpu_build_hash = use_gpu_build_hash;
                scfg.use_gpu_probe_hash = use_gpu_probe_hash;
                scfg.gpu_probe_batch_mb = gpu_probe_mem_mb_val;

                scfg.use_gpu_fuzzy_direct_hash = use_gpu_fuzzy_direct_hash;
                scfg.direct_use_fuzzy_normalization = direct_norm_fuzzy;
                // Apply global normalization alignment for in-memory comparators as well
                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);

                // Apply global GPU fuzzy pre-pass toggle (used by in-memory fuzzy)
                name_matcher::matching::set_gpu_fuzzy_direct_prep(use_gpu_fuzzy_direct_hash);

                scfg.checkpoint_path = Some(format!("{}.nmckpt", path));
                match fmt {

                    FormatSel::Csv => {
                // Tune flush frequency based on batch size to prevent spikes
                if ssd_hint {
                    scfg.flush_every = (batch as usize / 6).max(1000);
                } else {
                    scfg.flush_every = (batch as usize / 12).max(1000);
                }

                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                        if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Fuzzy uses in-memory mode (streaming disabled)".into())); }
                        if use_gpu && !cfg!(feature = "gpu") {
                            let _ = tx_for_async.send(Msg::Info("GPU requested but this binary lacks GPU support; will run on CPU. Rebuild with --features gpu".into()));
                        }
                        if use_streaming {
                            // GPU availability info (best-effort)
                            #[cfg(feature = "gpu")]
                            if use_gpu {
                                if let Ok(ctx) = cudarc::driver::CudaContext::new(0) {
                                    let mut free: usize = 0; let mut total: usize = 0;
                                    unsafe { let _ = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _ as *mut _, &mut total as *mut _ as *mut _); }
                                    let _ = tx_for_async.send(Msg::Info(format!("CUDA active | Free {} MB / Total {} MB", (free/1024/1024), (total/1024/1024))));
                                    drop(ctx);
                                } else {
                                    let _ = tx_for_async.send(Msg::Info("CUDA requested but unavailable; falling back to CPU".into()));
                                }
                            }

                            let mut w = CsvStreamWriter::create(&path, algo, fuzzy_thr)?;
                            let mut cnt = 0usize;
                            let mut kept = 0usize;
                            let flush_every = scfg.flush_every;
                            let txp = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, algo, |p| {
                                    cnt += 1;
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                        if p.confidence >= 0.95 { kept += 1; }
                                    }
                                    w.write(p)?;
                                    if cnt % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| { let _ = txp.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, algo, |p| {
                                    cnt += 1;
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                        if p.confidence >= 0.95 { kept += 1; }
                                    }
                                    w.write(p)?;
                                    if cnt % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| { let _ = txp.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            w.flush()?;
                            let csv_val = if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) { kept } else { cnt };
                            Ok((0,0,csv_val,path.clone()))
                        } else {
                            // In-memory path
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let txp = tx_for_async.clone();
                                // Also reflect GPU fuzzy pre-pass in in-memory path
                                name_matcher::matching::set_gpu_fuzzy_direct_prep(use_gpu_fuzzy_direct_hash);

                                // Apply normalization alignment globally for in-memory deterministics as well
                                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);

                            if matches!(algo, MatchingAlgorithm::HouseholdGpu) {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp };
                                let rows = match_households_gpu_inmemory(&t1, &t2, mo, fuzzy_thr, move |u| { let _ = txp.send(Msg::Progress(u)); });
                                let mut w = HouseholdCsvWriter::create(&path)?;
                                for r in &rows { w.write(r)?; }
                                w.flush()?;
                                Ok((0,0, rows.len(), path.clone()))
                            } else {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp };
                                let pairs = match_all_with_opts(&t1, &t2, algo, mo, move |u| { let _ = txp.send(Msg::Progress(u)); });
                                let mut w = CsvStreamWriter::create(&path, algo, fuzzy_thr)?;
                                for p in &pairs { w.write(p)?; }
                                w.flush()?;
                                let kept = if matches!(algo, MatchingAlgorithm::Fuzzy) { let thr = fuzzy_thr; pairs.iter().filter(|p| p.confidence >= thr).count() } else { pairs.len() };
                                Ok((0,0, kept, path.clone()))
                            }
                        }
                    }
                    FormatSel::Xlsx => {
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                            if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Selected algorithm uses in-memory mode (streaming disabled)".into())); }
                        if use_streaming {
                            let mut xw = XlsxStreamWriter::create(&path)?;
                            let mut a1 = 0usize; let mut a2 = 0usize;
                            let txp1 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| { let _ = txp1.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| { let _ = txp1.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            let txp2 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| { let _ = txp2.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| { let _ = txp2.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            // Fetch row counts for summary
                            let c1 = get_person_count(&pool1, &table1).await?;
                            let c2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_count(pool2, &table2).await? } else { get_person_count(&pool1, &table2).await? };

                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: c1 as usize, total_table2: c2 as usize,
                                matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                started_utc: Utc::now(), ended_utc: Utc::now(), duration_secs: 0.0,
                                algo_used: "Both (1,2)".into(), gpu_used: false, gpu_total_mb: 0, gpu_free_mb_end: 0,
                            })?;
                            Ok((a1,a2,0,path.clone()))
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                                // Apply normalization alignment for in-memory deterministic algorithms
                                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);

                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let mut xw = XlsxStreamWriter::create(&path)?;
                            let mut a1: usize = 0; let mut a2: usize = 0;
                            if matches!(algo, MatchingAlgorithm::HouseholdGpu) {
                                let txp = tx_for_async.clone();
let rows = match_households_gpu_inmemory(&t1, &t2, MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp }, fuzzy_thr, move |u| { let _ = txp.send(Msg::Progress(u)); });
                                export_households_xlsx(&path, &rows)?;
                            } else {
                                let txp1 = tx_for_async.clone();
                                // Also reflect GPU fuzzy pre-pass for in-memory runs
                                name_matcher::matching::set_gpu_fuzzy_direct_prep(use_gpu_fuzzy_direct_hash);

                                let pairs1 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, cfgp, move |u| { let _ = txp1.send(Msg::Progress(u)); });
                                for p in &pairs1 { xw.append_algo1(p)?; }
                                let txp2 = tx_for_async.clone();
                                let pairs2 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, cfgp, move |u| { let _ = txp2.send(Msg::Progress(u)); });
                                for p in &pairs2 { xw.append_algo2(p)?; }
                                a1 = pairs1.len(); a2 = pairs2.len();
                            }
                            // counts taken from pairs computed above (or 0 for household)
                            // GPU availability info (best-effort)
                            #[cfg(feature = "gpu")]
                            if use_gpu {
                                if let Ok(ctx) = cudarc::driver::CudaContext::new(0) {
                                    let mut free: usize = 0; let mut total: usize = 0;
                                    unsafe { let _ = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _ as *mut _, &mut total as *mut _ as *mut _); }
                                    let _ = tx_for_async.send(Msg::Info(format!("CUDA active | Free {} MB / Total {} MB", (free/1024/1024), (total/1024/1024))));
                                    drop(ctx);
                                } else {
                                    let _ = tx_for_async.send(Msg::Info("CUDA requested but unavailable; falling back to CPU".into()));
                                }
                            }

                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: t1.len(), total_table2: t2.len(),
                                matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                started_utc: Utc::now(), ended_utc: Utc::now(), duration_secs: 0.0,
                                algo_used: "Both (1,2)".into(), gpu_used: false, gpu_total_mb: 0, gpu_free_mb_end: 0,
                            })?;
                            Ok((a1,a2,0,path.clone()))
                        }
                    }
                    FormatSel::Both => {
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                            if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Fuzzy uses in-memory mode (streaming disabled)".into())); }
                        let mut csv_path = path.clone(); if !csv_path.to_ascii_lowercase().ends_with(".csv") { csv_path.push_str(".csv"); }
                        let mut csv_count = 0usize;
                        if use_streaming {
                            let mut w = CsvStreamWriter::create(&csv_path, algo, fuzzy_thr)?;
                            let flush_every = scfg.flush_every;
                            let txp3 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, algo, |p| {
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                        if p.confidence >= 0.95 { csv_count += 1; }
                                    } else { csv_count += 1; }
                                    w.write(p)?;
                                    if csv_count % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| { let _ = txp3.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, algo, |p| {
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                        if p.confidence >= 0.95 { csv_count += 1; }
                                    } else { csv_count += 1; }
                                    w.write(p)?;
                                    if csv_count % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| { let _ = txp3.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            w.flush()?;
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                                // Apply normalization alignment for in-memory A1/A2
                                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);

                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let txp = tx_for_async.clone();
                            let pairs = if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp };
                                match_all_with_opts(&t1, &t2, algo, mo, move |u| { let _ = txp.send(Msg::Progress(u)); })
                            } else {
                                match_all_progress(&t1, &t2, algo, cfgp, move |u| { let _ = txp.send(Msg::Progress(u)); })
                            };
                            if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                let thr = fuzzy_thr;
                                csv_count = pairs.iter().filter(|p| p.confidence >= thr).count();
                            } else { csv_count = pairs.len(); }
                            let mut w = CsvStreamWriter::create(&csv_path, algo, fuzzy_thr)?;
                            for p in &pairs { w.write(p)?; }
                            w.flush()?;
                        }
                        let xlsx_path = if path.to_ascii_lowercase().ends_with(".xlsx") { path.clone() } else { path.replace(".csv", ".xlsx") };
                        if use_streaming {
                            let mut xw = XlsxStreamWriter::create(&xlsx_path)?;
                            let mut a1 = 0usize; let mut a2 = 0usize;
                            let txp4 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| { let _ = txp4.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| { let _ = txp4.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            let txp5 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| { let _ = txp5.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| { let _ = txp5.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            // Fetch row counts for summary
                            let c1 = get_person_count(&pool1, &table1).await?;
                            let c2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_count(pool2, &table2).await? } else { get_person_count(&pool1, &table2).await? };

                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: c1 as usize, total_table2: c2 as usize,
                                matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                started_utc: Utc::now(), ended_utc: Utc::now(), duration_secs: 0.0,
                                algo_used: "Both (1,2)".into(), gpu_used: false, gpu_total_mb: 0, gpu_free_mb_end: 0,
                            })?;
                            Ok((a1,a2,csv_count,path.clone()))
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let mut xw = XlsxStreamWriter::create(&xlsx_path)?;
                            let txp1 = tx_for_async.clone();
                            let pairs1 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, cfgp, move |u| { let _ = txp1.send(Msg::Progress(u)); });
                            for p in &pairs1 { xw.append_algo1(p)?; }
                            let txp2 = tx_for_async.clone();
                            let pairs2 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, cfgp, move |u| { let _ = txp2.send(Msg::Progress(u)); });
                            for p in &pairs2 { xw.append_algo2(p)?; }
                            let a1 = pairs1.len(); let a2 = pairs2.len();
                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: t1.len(), total_table2: t2.len(),
                                matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                started_utc: Utc::now(), ended_utc: Utc::now(), duration_secs: 0.0,
                                algo_used: "Both (1,2)".into(), gpu_used: false, gpu_total_mb: 0, gpu_free_mb_end: 0,
                            })?;
                            Ok((a1,a2,csv_count,path.clone()))
                        }
                    }
                }
            });
            match res { Ok((a1,a2,csv,out_path)) => { let _ = tx.send(Msg::Done { a1, a2, csv, path: out_path }); }, Err(e) => { let (sqlstate, chain) = extract_sqlstate_and_chain(&e); let _ = tx.send(Msg::ErrorRich { display: format!("{}", e), sqlstate, chain, operation: Some("Run Matching".into()) }); } }
        });
    }

    fn reset_state(&mut self) {
        *self = GuiApp::default();
    }
    fn test_connection(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
        self.last_action = "Test Connection".into();
        let enable_dual = self.enable_dual;
        // DB1
        let host = self.host.clone(); let port = self.port.clone(); let user = self.user.clone(); let pass = self.pass.clone(); let dbname = self.db.clone();
        let t1 = self.tables.get(self.table1_idx).cloned();
        // DB2 (optional)
        let host2 = self.host2.clone(); let port2 = self.port2.clone(); let user2 = self.user2.clone(); let pass2 = self.pass2.clone(); let dbname2 = self.db2.clone();
        let t2 = if enable_dual { self.tables2.get(self.table2_idx).cloned() } else { self.tables.get(self.table2_idx).cloned() };
        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let res: Result<(Option<i64>, Option<i64>)> = rt.block_on(async move {
                if enable_dual {
                    // connect to both DBs
                    let cfg1 = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname };
                    let pool1 = make_pool_with_size(&cfg1, Some(4)).await?;
                    let _pong1: i32 = sqlx::query_scalar("SELECT 1").fetch_one(&pool1).await?;
                    let cfg2 = DatabaseConfig { host: host2, port: port2.parse().unwrap_or(3306), username: user2, password: pass2, database: dbname2 };
                    let pool2 = make_pool_with_size(&cfg2, Some(4)).await?;
                    let _pong2: i32 = sqlx::query_scalar("SELECT 1").fetch_one(&pool2).await?;
                    let c1 = if let Some(t) = t1.as_ref() { Some(get_person_count(&pool1, t).await?) } else { None };
                    let c2 = if let Some(t) = t2.as_ref() { Some(get_person_count(&pool2, t).await?) } else { None };
                    Ok((c1, c2))
                } else {
                    let cfg = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname };
                    let pool = make_pool_with_size(&cfg, Some(4)).await?;
                    let _pong: i32 = sqlx::query_scalar("SELECT 1").fetch_one(&pool).await?;
                    let c1 = if let Some(t) = t1.as_ref() { Some(get_person_count(&pool, t).await?) } else { None };
                    let c2 = if let Some(t) = t2.as_ref() { Some(get_person_count(&pool, t).await?) } else { None };
                    Ok((c1, c2))
                }
            });
            match res {
                Ok((c1,c2)) => { let _=tx.send(Msg::Info(format!("Connected. Row counts: t1={:?}, t2={:?}", c1, c2))); },
                Err(e) => { let (sqlstate, chain) = extract_sqlstate_and_chain(&e); let _=tx.send(Msg::ErrorRich { display: format!("Connection failed: {}", e), sqlstate, chain, operation: Some("Connect/Test".into()) }); }
            }
        });
        self.rx = rx; self.status = "Testing connection...".into();
    }

    fn estimate(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
        self.last_action = "Estimate".into();
        let host = self.host.clone(); let port = self.port.clone(); let user = self.user.clone(); let pass = self.pass.clone(); let dbname = self.db.clone();
        let t1 = self.tables.get(self.table1_idx).cloned();
        let t2 = self.tables.get(self.table2_idx).cloned();
        let mem_thr_mb = self.mem_thresh.parse::<u64>().unwrap_or(800);
        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let res: Result<String> = rt.block_on(async move {
                let cfg = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname };
                let pool = make_pool_with_size(&cfg, Some(4)).await?;
                let (c1,c2) = match (t1.as_ref(), t2.as_ref()) {
                    (Some(a), Some(b)) => (get_person_count(&pool, a).await?, get_person_count(&pool, b).await?),
                    _ => (0,0)
                };
                let small = c1.min(c2) as u64;
                // very rough estimate: ~96 bytes per index row; batch overhead ~ 64 bytes/row
                let index_bytes = small.saturating_mul(96);
                let index_mb = (index_bytes as f64 / (1024.0*1024.0)).ceil() as u64;
                let suggestion = if index_mb > mem_thr_mb || small > 200_000 { "Streaming" } else { "In-memory" };
                Ok(format!("Estimated index ~{} MB ({} vs {}). Suggested mode: {} (threshold {} MB)", index_mb, c1, c2, suggestion, mem_thr_mb))
            });
            match res { Ok(s) => { let _=tx.send(Msg::Info(s)); }, Err(e) => { let (sqlstate, chain) = extract_sqlstate_and_chain(&e); let _=tx.send(Msg::ErrorRich { display: format!("Estimate failed: {}", e), sqlstate, chain, operation: Some("Estimate Resources".into()) }); } }
        });
        self.rx = rx; self.status = "Estimating...".into();
    }


    fn poll_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Progress(u) => {
                    self.progress = u.percent; self.eta_secs = u.eta_secs; self.mem_used = u.mem_used_mb; self.mem_avail = u.mem_avail_mb;
                    self.gpu_total_mb = u.gpu_total_mb; self.gpu_free_mb = u.gpu_free_mb; self.gpu_active = u.gpu_active;
                    self.processed = u.processed; self.total = u.total; self.batch_current = u.batch_size_current.unwrap_or(0); self.stage = u.stage.to_string();
                    // Runtime monitoring: warn if free memory dips below threshold; streaming will reduce batch size automatically
                    let mem_thr = self.mem_thresh.parse::<u64>().unwrap_or(800);
                    if self.mem_avail < mem_thr {
                        self.status = format!("Warning: low free memory ({} MB < {} MB). Auto-throttling batches.", self.mem_avail, mem_thr);
                    }
                    // Separate GPU build/probe active indicators by stage hint
                    match u.stage {
                        "gpu_hash" => { self.gpu_build_active_now = true; }
                        "gpu_hash_done" => { self.gpu_build_active_now = true; }
                        "gpu_probe_hash" => { self.gpu_probe_active_now = true; }
                        "gpu_probe_hash_done" => { self.gpu_probe_active_now = true; }
                        _ => {}
                    }
                    let now = std::time::Instant::now();
                    if let Some(t0) = self.last_tick {
                        let dt = now.duration_since(t0).as_secs_f32().max(1e-3);
                        self.rps = ((self.processed.saturating_sub(self.last_processed_prev)) as f32 / dt).max(0.0);
                    } else {
                        self.rps = 0.0;
                    }
                    self.last_tick = Some(now);
                    self.last_processed_prev = self.processed;
                    self.status = format!("{} | {:.1}% | {} / {} recs | {:.0} rec/s", self.stage, u.percent, self.processed, self.total, self.rps);

                    // log tail
                    let line = format!("{} PROGRESS stage={} percent={:.1} processed={}/{} rps={:.0}", chrono::Utc::now().to_rfc3339(), self.stage, u.percent, self.processed, self.total, self.rps);
                    self.log_buffer.push(line);
                    if self.log_buffer.len()>200 { let drop = self.log_buffer.len()-200; self.log_buffer.drain(0..drop); }
                }
                Msg::Info(s) => {
                    self.status = s.clone();
                    let line = format!("{} INFO {}", chrono::Utc::now().to_rfc3339(), s);
                    self.log_buffer.push(line);
                    if self.log_buffer.len()>200 { let drop = self.log_buffer.len()-200; self.log_buffer.drain(0..drop); }
                }
                Msg::Tables(v) => { self.tables = v; self.table1_idx = 0; if self.table1_idx >= self.tables.len() { self.table1_idx = 0; } self.status = format!("Loaded {} tables (DB1)", self.tables.len()); }
                Msg::Tables2(v2) => { self.tables2 = v2; self.table2_idx = 0; if self.table2_idx >= self.tables2.len() { self.table2_idx = 0; } self.status = format!("Loaded {} tables (DB2)", self.tables2.len()); }
                Msg::Done { a1, a2, csv, path } => {
                    self.running = false; self.a1_count = a1; self.a2_count = a2; self.csv_count = csv; self.progress = 100.0;
                    if self.use_gpu && matches!(self.algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) && self.gpu_total_mb == 0 {
                        self.status = format!("Done (CPU fallback — no GPU activity detected). Output: {}", path);
                    } else {
                        // Write standalone summary CSV/XLSX alongside output
                        let db_label = if self.enable_dual { format!("{} | {}", self.db, self.db2) } else { self.db.clone() };
                        let t1 = self.tables.get(self.table1_idx).cloned().unwrap_or_default();
                        let t2 = if self.enable_dual { self.tables2.get(self.table2_idx).cloned().unwrap_or_default() } else { self.tables.get(self.table2_idx).cloned().unwrap_or_default() };
                        let matches_fuzzy = if matches!(self.algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) { csv } else { 0 };
                        // Derive accurate table totals for summary using lightweight COUNT(*)
                        let (sum_c1, sum_c2) = {
                            let host = self.host.clone(); let port = self.port.clone(); let user = self.user.clone(); let pass = self.pass.clone(); let dbname = self.db.clone();
                            let host2 = self.host2.clone(); let port2 = self.port2.clone(); let user2 = self.user2.clone(); let pass2 = self.pass2.clone(); let dbname2 = self.db2.clone();
                            let t1n = self.tables.get(self.table1_idx).cloned().unwrap_or_default();
                            let t2n = if self.enable_dual { self.tables2.get(self.table2_idx).cloned().unwrap_or_default() } else { self.tables.get(self.table2_idx).cloned().unwrap_or_default() };
                            let enable_dual = self.enable_dual;
                            let rt = tokio::runtime::Runtime::new().unwrap();
                            rt.block_on(async move {
                                use name_matcher::db::connection::make_pool;
                                use name_matcher::config::DatabaseConfig;
                                use name_matcher::db::get_person_count;
                                if enable_dual {
                                    let cfg1 = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname };
                                    let cfg2 = DatabaseConfig { host: host2, port: port2.parse().unwrap_or(3306), username: user2, password: pass2, database: dbname2 };
                                    if let (Ok(p1), Ok(p2)) = (make_pool(&cfg1).await, make_pool(&cfg2).await) {
                                        let c1 = get_person_count(&p1, &t1n).await.unwrap_or(0);
                                        let c2 = get_person_count(&p2, &t2n).await.unwrap_or(0);
                                        (c1, c2)
                                    } else { (0,0) }
                                } else {
                                    let cfg = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname };
                                    if let Ok(p) = make_pool(&cfg).await {
                                        let c1 = get_person_count(&p, &t1n).await.unwrap_or(0);
                                        let c2 = get_person_count(&p, &t2n).await.unwrap_or(0);
                                        (c1, c2)
                                    } else { (0,0) }
                                }
                            })
                        };

                        let summary = SummaryContext {
                            db_name: db_label,
                            table1: t1,
                            table2: t2,
                            total_table1: sum_c1 as usize,
                            total_table2: sum_c2 as usize,
                            matches_algo1: a1,
                            matches_algo2: a2,
                            matches_fuzzy,
                            overlap_count: 0,
                            unique_algo1: a1,
                            unique_algo2: a2,
                            fetch_time: std::time::Duration::from_secs(0),
                            match1_time: std::time::Duration::from_secs(0),
                            match2_time: std::time::Duration::from_secs(0),
                            export_time: std::time::Duration::from_secs(0),
                            mem_used_start_mb: 0,
                            mem_used_end_mb: 0,
                            started_utc: Utc::now(),
                            ended_utc: Utc::now(),
                            duration_secs: 0.0,
                            algo_used: format!("{:?}", self.algo),
                            gpu_used: self.gpu_active,
                            gpu_total_mb: self.gpu_total_mb,
                            gpu_free_mb_end: self.gpu_free_mb,
                        };
                        let out_dir = std::path::Path::new(&path).parent().unwrap_or(std::path::Path::new("."));
                        let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
                        let sum_csv = out_dir.join(format!("summary_report_{}.csv", ts));
                        let sum_xlsx = out_dir.join(format!("summary_report_{}.xlsx", ts));
                        if let Err(e) = name_matcher::export::csv_export::export_summary_csv(sum_csv.to_string_lossy().as_ref(), &summary) {
                            self.status = format!("Finished, but failed to write summary CSV: {}", e);
                        }
                        if let Err(e) = name_matcher::export::xlsx_export::export_summary_xlsx(sum_xlsx.to_string_lossy().as_ref(), &summary) {
                            self.status = format!("Finished, but failed to write summary XLSX: {}", e);
                        }

                        self.status = format!("Done. Output: {}", path);
                    }
                }
                Msg::Error(e) => {
                    self.running = false;
                    self.status = format!("Error: {}", e);
                    self.record_error(e);
                }
                Msg::ErrorRich { display, sqlstate, chain, operation } => {
                    self.running = false;
                    self.status = format!("Error: {}", display);
                    self.record_error_with_details(display, sqlstate, Some(chain), operation);
                }
            }
        }
    }

    fn record_error(&mut self, message: String) {
        let sanitize = |s: &str| -> String {
            if let Some(pos) = s.find("mysql://") {
                if let Some(at) = s[pos..].find('@') { let mut out = s.to_string(); out.replace_range(pos..pos+at, "mysql://[REDACTED]"); return out; }
            }
            s.to_string()
        };
        let mstats = name_matcher::metrics::memory_stats_mb();
        let pool_sz = self.pool_size.parse::<u32>().unwrap_or(0);
        let envs = ["NAME_MATCHER_POOL_SIZE","NAME_MATCHER_POOL_MIN","NAME_MATCHER_STREAMING","NAME_MATCHER_PARTITION"];
        let mut env_overrides = Vec::new();
        for k in envs { if let Ok(v) = std::env::var(k) { env_overrides.push((k.to_string(), v)); } }
        let evt = DiagEvent {
            ts_utc: Utc::now().to_rfc3339(),
            category: categorize_error(&message),
            message: sanitize(&message),
            sqlstate: None,
            chain: None,
            operation: None,
            source_action: self.last_action.clone(),
            db1_host: self.host.clone(),
            db1_database: self.db.clone(),
            db2_host: if self.enable_dual { Some(self.host2.clone()) } else { None },
            db2_database: if self.enable_dual { Some(self.db2.clone()) } else { None },
            table1: self.tables.get(self.table1_idx).cloned(),
            table2: if self.enable_dual { self.tables2.get(self.table2_idx).cloned() } else { self.tables.get(self.table2_idx).cloned() },
            mem_avail_mb: mstats.avail_mb,
            pool_size_cfg: pool_sz,
            env_overrides,
        };
        self.error_events.push(evt);
    }

    fn export_error_report(&mut self) -> Result<String> {
        let ts = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let default_name = match self.report_format { ReportFormat::Text => format!("error_report_{}.txt", ts), ReportFormat::Json => format!("error_report_{}.json", ts) };
        let mut dialog = rfd::FileDialog::new().set_file_name(&default_name);
        match self.report_format { ReportFormat::Text => { dialog = dialog.add_filter("Text", &["txt"]); }, ReportFormat::Json => { dialog = dialog.add_filter("JSON", &["json"]); } }
        let path = dialog.save_file().map(|p| p.display().to_string()).unwrap_or(default_name);

        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
        let mem = name_matcher::metrics::memory_stats_mb();
        let ver = env!("CARGO_PKG_VERSION");
        let os = std::env::consts::OS; let arch = std::env::consts::ARCH;
        // Collect selected environment variables with sanitization
        let mut env_selected: Vec<(String,String)> = Vec::new();
        for (k,v) in std::env::vars() {
            if k.starts_with("SQLX_") || k=="RUST_LOG" || k=="RUST_BACKTRACE" {
                let vv = if v.contains("mysql://") { v.replacen("mysql://", "mysql://[REDACTED]@", 1) } else { v.clone() };
                env_selected.push((k, vv));
            }
        }

        let suggestions_for = |cat: ErrorCategory| -> Vec<&'static str> {
            match cat {
                ErrorCategory::DbConnection => vec![
                    "Verify host/port reachability (ping, firewall)",
                    "Check username/password and privileges",
                    "Confirm database name exists and user has access",
                ],
                ErrorCategory::TableValidation => vec![
                    "Ensure selected tables exist and user has SELECT permission",
                    "Click 'Load Tables' to refresh the list",
                ],
                ErrorCategory::SchemaValidation => vec![
                    "Add required columns and indexes (see README: Required Indexes)",
                    "Verify column types match expected formats",
                ],
                ErrorCategory::DataFormat => vec![
                    "Normalize/cleanse date formats (YYYY-MM-DD)",
                    "Fill required fields or exclude nulls where needed",
                ],
                ErrorCategory::ResourceConstraint => vec![
                    "Use Streaming mode and reduce batch size",
                    "Close other apps to free RAM; ensure sufficient disk space",

                ],
                ErrorCategory::Configuration => vec![
                    "Check environment variables (NAME_MATCHER_*)",
                    "Re-enter GUI settings and retry",
                ],
                ErrorCategory::Unknown => vec!["Review logs and contact support with this report"],
            }
        };

        match self.report_format {
            ReportFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!("SRS-II Name Matching - Diagnostic Report\nVersion: {}\nTimestamp: {}\n\n", ver, Utc::now().to_rfc3339()));
                out.push_str(&format!("System: os={} arch={} cores={} | mem_avail={} MB\n", os, arch, cores, mem.avail_mb));
                let mode_str = match self.mode { ModeSel::Auto => "Auto", ModeSel::Streaming => "Streaming", ModeSel::InMemory => "InMemory" };
                let fmt_str = match self.fmt { FormatSel::Csv => "CSV", FormatSel::Xlsx => "XLSX", FormatSel::Both => "Both" };
                let algo_str = match self.algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => "Algo1", MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => "Algo2", MatchingAlgorithm::Fuzzy => "Fuzzy", MatchingAlgorithm::FuzzyNoMiddle => "FuzzyNoMiddle", MatchingAlgorithm::HouseholdGpu => "Household" };
                out.push_str(&format!("Config: db1_host={} db1_db={} enable_dual={} pool_size={} ssd_storage={} mode={} algo={} fmt={}\n\n",
                    self.host, self.db, self.enable_dual, self.pool_size, self.ssd_storage, mode_str, algo_str, fmt_str));
                for (i, evt) in self.error_events.iter().enumerate() {
                    out.push_str(&format!("[{}] {} | Category: {:?}\nAction: {}\nDB1: {}.{}\nDB2: {}.{}\nTables: {:?} vs {:?}\nMem avail: {} MB | Pool: {}\nMessage: {}\n",
                        i+1, evt.ts_utc, evt.category, evt.source_action,
                        evt.db1_host, evt.db1_database,
                        evt.db2_host.clone().unwrap_or("-".into()), evt.db2_database.clone().unwrap_or("-".into()),
                        evt.table1, evt.table2, evt.mem_avail_mb, evt.pool_size_cfg, evt.message));
                    if let Some(ref ss) = evt.sqlstate { out.push_str(&format!("SQLSTATE: {}\n", ss)); }
                    if let Some(ref op) = evt.operation { out.push_str(&format!("Operation: {}\n", op)); }
                    if let Some(ref ch) = evt.chain { out.push_str("Chain:\n"); out.push_str(ch); out.push_str("\n"); }
                    if !evt.env_overrides.is_empty() { out.push_str("Env overrides:\n"); for (k,v) in &evt.env_overrides { out.push_str(&format!("  {}={}\n", k, v)); } }
                    // Specific remediation hints from message
                    let mut specific: Vec<String> = Vec::new();
                    if evt.message.contains("Unknown column '") {
                        if let Some(s) = evt.message.find("Unknown column '") {
                            let rest = &evt.message[s + "Unknown column '".len()..]; if let Some(e) = rest.find('\'') {
                                let col = &rest[..e]; let tbl = evt.table1.clone().or(evt.table2.clone()).unwrap_or("<table>".into());
                                specific.push(format!("ALTER TABLE {} ADD COLUMN {} <TYPE>" , tbl, col));
                            }
                        }
                    }
                    if evt.message.to_ascii_lowercase().contains("doesn't exist") && evt.message.to_ascii_lowercase().contains("table") {
                        let tbl = evt.table1.clone().or(evt.table2.clone()).unwrap_or("<table>".into());
                        specific.push(format!("CREATE TABLE {} (...) or select an existing table", tbl));
                    }
                    if evt.message.contains("Incorrect date value") || evt.message.contains("invalid date") {
                        specific.push("Normalize date format to YYYY-MM-DD and ensure column type is DATE".into());
                    }
                    if !specific.is_empty() { out.push_str("Specific remediation:\n"); for s in &specific { out.push_str(&format!("  - {}\n", s)); } }
                    out.push_str("Remediation:\n"); for s in suggestions_for(evt.category) { out.push_str(&format!("  - {}\n", s)); }
                    out.push_str("\n");
                    // Optional schema analysis
                    if self.schema_analysis_enabled {
                        let t1 = self.tables.get(self.table1_idx).cloned();
                        let t2 = if self.enable_dual { self.tables2.get(self.table2_idx).cloned() } else { None };
                        let host = self.host.clone(); let port = self.port.clone(); let user = self.user.clone(); let pass = self.pass.clone(); let dbname = self.db.clone();
                        let host2 = self.host2.clone(); let port2 = self.port2.clone(); let user2 = self.user2.clone(); let pass2 = self.pass2.clone(); let dbname2 = self.db2.clone();
                        let mut summary = String::new();
                        let mut index_suggestions: Vec<String> = Vec::new();
                        let mut grant_suggestions: Vec<String> = Vec::new();
                        let mut charset_notes: Vec<String> = Vec::new();
                        // derive hints from last error
                        if let Some(last) = self.error_events.last() {
                            let mm = last.message.to_ascii_lowercase() + "\n" + &last.chain.clone().unwrap_or_default().to_ascii_lowercase();
                            if mm.contains("command denied") || mm.contains("access denied") { grant_suggestions.push(format!("GRANT SELECT ON `{}`.* TO '{}'@'%';", self.db, self.user)); }
                            if mm.contains("incorrect string value") || mm.contains("collation") { charset_notes.push("Consider aligning character set/collation: ALTER TABLE `<db>.<table>` CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;".into()); }
                            if mm.contains("foreign key constraint fails") { charset_notes.push("Verify parent rows exist and consider indexing FK columns in child table to speed checks.".into()); }
                        }
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        let res: anyhow::Result<()> = rt.block_on(async {
                            let cfg1 = DatabaseConfig { host: host.clone(), port: port.parse().unwrap_or(3306), username: user.clone(), password: pass.clone(), database: dbname.clone() };
                            let pool1 = make_pool_with_size(&cfg1, Some(4)).await?;
                            if let Some(table) = t1.as_ref() {
                                summary.push_str(&format!("[DB1:{}] Table `{}`\n", dbname, table));
                                // Columns
                                let cols = sqlx::query(
                                    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION"
                                ).bind(&dbname).bind(table).fetch_all(&pool1).await?;
                                let mut actual: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                                for row in cols {
                                    use sqlx::Row;
                                    let cname: String = row.get::<String,_>("COLUMN_NAME");
                                    let dtype: String = row.get::<String,_>("DATA_TYPE");
                                    let is_null: String = row.get::<String,_>("IS_NULLABLE");
                                    let clen: Option<i64> = row.try_get::<i64,_>("CHARACTER_MAXIMUM_LENGTH").ok();
                                    actual.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null == "YES", clen));
                                }
                                let expected = [
                                    ("id","bigint", false, None),
                                    ("uuid","varchar", false, Some(64)),
                                    ("first_name","varchar", false, Some(255)),
                                    ("middle_name","varchar", true, Some(255)),
                                    ("last_name","varchar", false, Some(255)),
                                    ("birthdate","date", false, None),
                                ];
                                for (name, ty, nullable, len) in expected {
                                    match actual.get(name) {
                                        None => summary.push_str(&format!("  - Missing column `{}` (expected {}{})\n", name, ty, len.map(|n| format!("({})", n)).unwrap_or_default())),
                                        Some((aty, anull, alen)) => {
                                            // type check (contains to allow varchar vs varchar)
                                            if !aty.contains(ty) { summary.push_str(&format!("  - Type mismatch `{}` actual {} vs expected {}\n", name, aty, ty)); }
                                            if *anull && !nullable { summary.push_str(&format!("  - Nullability mismatch `{}` is NULL but expected NOT NULL\n", name)); }
                                            if let (Some(exp), Some(act)) = (len, *alen) { if act < exp as i64 { summary.push_str(&format!("  - Length `{}` actual {} < expected {}\n", name, act, exp)); } }
                                        }
                                    }
                                }
                                // Index check on id
                                let idx = sqlx::query(
                                    "SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
                                ).bind(&dbname).bind(table).fetch_all(&pool1).await?;
                                let has_id_idx = idx.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" });
                                if !has_id_idx { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname, table, table)); }
                            }
                            if let (true, Some(table2)) = (self.enable_dual, t2.as_ref()) {
                                let cfg2 = DatabaseConfig { host: host2.clone(), port: port2.parse().unwrap_or(3306), username: user2.clone(), password: pass2.clone(), database: dbname2.clone() };
                                let pool2 = make_pool_with_size(&cfg2, Some(4)).await?;
                                summary.push_str(&format!("[DB2:{}] Table `{}`\n", dbname2, table2));
                                let cols2 = sqlx::query(
                                    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION"
                                ).bind(&dbname2).bind(table2).fetch_all(&pool2).await?;
                                let mut actual2: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                                for row in cols2 {
                                    use sqlx::Row;
                                    let cname: String = row.get::<String,_>("COLUMN_NAME");
                                    let dtype: String = row.get::<String,_>("DATA_TYPE");
                                    let is_null: String = row.get::<String,_>("IS_NULLABLE");
                                    let clen: Option<i64> = row.try_get::<i64,_>("CHARACTER_MAXIMUM_LENGTH").ok();
                                    actual2.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null=="YES", clen));
                                }
                                for name in ["id","uuid","first_name","last_name","birthdate"] {
                                    if !actual2.contains_key(name) { summary.push_str(&format!("  - Missing column `{}`\n", name)); }
                                }
                                let idx2 = sqlx::query(
                                    "SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
                                ).bind(&dbname2).bind(table2).fetch_all(&pool2).await?;
                                let has_id_idx2 = idx2.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" });
                                if !has_id_idx2 { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname2, table2, table2)); }
                            }
                            anyhow::Ok(())
                        });
                        let _ = res; // ignore analysis errors silently in report generation
                        out.push_str("Schema Analysis (metadata-only):\n");
                        if summary.is_empty() { out.push_str("  OK: No obvious schema issues detected for selected tables.\n"); } else { out.push_str(&summary); }
                        if !index_suggestions.is_empty() { out.push_str("Index Suggestions:\n"); for s in &index_suggestions { out.push_str(&format!("  {}\n", s)); } }
                        if !grant_suggestions.is_empty() { out.push_str("Privilege Suggestions:\n"); for s in &grant_suggestions { out.push_str(&format!("  {}\n", s)); } }
                        if !charset_notes.is_empty() { out.push_str("Charset/Collation Notes:\n"); for s in &charset_notes { out.push_str(&format!("  {}\n", s)); } }
                    }
                    // Env (selected)
                    if !env_selected.is_empty() { out.push_str("Env selected (sanitized):\n"); for (k,v) in &env_selected { out.push_str(&format!("  {}={}\n", k, v)); } }
                    // Log tail
                    if !self.log_buffer.is_empty() { out.push_str("Log tail (most recent first):\n"); for line in self.log_buffer.iter().rev().take(200) { out.push_str(&format!("  {}\n", line)); } }
                }
                fs::write(&path, out)?;
                Ok(path)
            }
            ReportFormat::Json => {
                let escape = |s: &str| s.replace('"', "\\\"");
                let mut out = String::new();
                out.push_str("{\n");
                out.push_str(&format!("  \"app\": \"SRS-II Name Matching Application\",\n"));
                out.push_str(&format!("  \"version\": \"{}\",\n", ver));
                out.push_str(&format!("  \"timestamp\": \"{}\",\n", Utc::now().to_rfc3339()));
                out.push_str(&format!("  \"system\": {{ \"os\": \"{}\", \"arch\": \"{}\", \"cores\": {}, \"mem_avail_mb\": {} }},\n", os, arch, cores, mem.avail_mb));
                out.push_str("  \"config\": {\n");
                out.push_str(&format!("    \"db1_host\": \"{}\",\n", escape(&self.host)));
                out.push_str(&format!("    \"db1_database\": \"{}\",\n", escape(&self.db)));
                out.push_str(&format!("    \"enable_dual\": {},\n", if self.enable_dual {"true"} else {"false"}));
                out.push_str(&format!("    \"pool_size_cfg\": \"{}\",\n", escape(&self.pool_size)));
                out.push_str(&format!("    \"ssd_storage\": {},\n", if self.ssd_storage {"true"} else {"false"}));
                let mode_str = match self.mode { ModeSel::Auto => "Auto", ModeSel::Streaming => "Streaming", ModeSel::InMemory => "InMemory" };
                let fmt_str = match self.fmt { FormatSel::Csv => "CSV", FormatSel::Xlsx => "XLSX", FormatSel::Both => "Both" };
                let algo_str = match self.algo { MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => "Algo1", MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => "Algo2", MatchingAlgorithm::Fuzzy => "Fuzzy", MatchingAlgorithm::FuzzyNoMiddle => "FuzzyNoMiddle", MatchingAlgorithm::HouseholdGpu => "Household" };
                out.push_str(&format!("    \"mode\": \"{}\",\n", mode_str));
                out.push_str(&format!("    \"algo\": \"{}\",\n", algo_str));
                out.push_str(&format!("    \"fmt\": \"{}\"\n", fmt_str));
                out.push_str("  },\n");
                // env_selected
                out.push_str("  \"env_selected\": [\n");
                for (i,(k,v)) in env_selected.iter().enumerate() { if i>0 { out.push_str(",\n"); } out.push_str(&format!("    {{ \"key\": \"{}\", \"value\": \"{}\" }}", escape(k), escape(v))); }
                out.push_str("\n  ],\n");
                // log_tail
                out.push_str("  \"log_tail\": [\n");
                for (i,line) in self.log_buffer.iter().rev().take(200).enumerate() { if i>0 { out.push_str(",\n"); } out.push_str(&format!("    \"{}\"", escape(line))); }
                out.push_str("\n  ],\n");
                // optional schema analysis
                if self.schema_analysis_enabled {
                    let t1 = self.tables.get(self.table1_idx).cloned();
                    let t2 = if self.enable_dual { self.tables2.get(self.table2_idx).cloned() } else { None };
                    let host = self.host.clone(); let port = self.port.clone(); let user = self.user.clone(); let pass = self.pass.clone(); let dbname = self.db.clone();
                    let host2 = self.host2.clone(); let port2 = self.port2.clone(); let user2 = self.user2.clone(); let pass2 = self.pass2.clone(); let dbname2 = self.db2.clone();
                    let mut summary = String::new();
                    let mut index_suggestions: Vec<String> = Vec::new();
                    let mut grant_suggestions: Vec<String> = Vec::new();
                    let mut charset_notes: Vec<String> = Vec::new();
                    if let Some(last) = self.error_events.last() {
                        let mm = last.message.to_ascii_lowercase() + "\n" + &last.chain.clone().unwrap_or_default().to_ascii_lowercase();
                        if mm.contains("command denied") || mm.contains("access denied") { grant_suggestions.push(format!("GRANT SELECT ON `{}`.* TO '{}'@'%';", self.db, self.user)); }
                        if mm.contains("incorrect string value") || mm.contains("collation") { charset_notes.push("Consider aligning character set/collation: ALTER TABLE `<db>.<table>` CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;".into()); }
                        if mm.contains("foreign key constraint fails") { charset_notes.push("Verify parent rows exist and consider indexing FK columns in child table to speed checks.".into()); }
                    }
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let _ = rt.block_on(async {
                        let cfg1 = DatabaseConfig { host: host.clone(), port: port.parse().unwrap_or(3306), username: user.clone(), password: pass.clone(), database: dbname.clone() };
                        let pool1 = make_pool_with_size(&cfg1, Some(4)).await?;
                        if let Some(table) = t1.as_ref() {
                            summary.push_str(&format!("[DB1:{}] Table `{}`\n", dbname, table));
                            let cols = sqlx::query("SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION").bind(&dbname).bind(table).fetch_all(&pool1).await?;
                            let mut actual: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                            for row in cols { use sqlx::Row; let cname: String = row.get("COLUMN_NAME"); let dtype: String = row.get("DATA_TYPE"); let is_null: String = row.get("IS_NULLABLE"); let clen: Option<i64> = row.try_get("CHARACTER_MAXIMUM_LENGTH").ok(); actual.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null=="YES", clen)); }
                            let expected = [("id","bigint", false, None),("uuid","varchar", false, Some(64)),("first_name","varchar", false, Some(255)),("middle_name","varchar", true, Some(255)),("last_name","varchar", false, Some(255)),("birthdate","date", false, None)];
                            for (name, ty, nullable, len) in expected { match actual.get(name) { None => summary.push_str(&format!("  - Missing column `{}` (expected {}{})\n", name, ty, len.map(|n| format!("({})", n)).unwrap_or_default())), Some((aty, anull, alen)) => { if !aty.contains(ty) { summary.push_str(&format!("  - Type mismatch `{}` actual {} vs expected {}\n", name, aty, ty)); } if *anull && !nullable { summary.push_str(&format!("  - Nullability mismatch `{}` is NULL but expected NOT NULL\n", name)); } if let (Some(exp), Some(act)) = (len, *alen) { if act < exp as i64 { summary.push_str(&format!("  - Length `{}` actual {} < expected {}\n", name, act, exp)); } } } } }
                            let idx = sqlx::query("SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?").bind(&dbname).bind(table).fetch_all(&pool1).await?; let has_id_idx = idx.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" }); if !has_id_idx { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname, table, table)); }
                        }
                        if let (true, Some(table2)) = (self.enable_dual, t2.as_ref()) {
                            let cfg2 = DatabaseConfig { host: host2.clone(), port: port2.parse().unwrap_or(3306), username: user2.clone(), password: pass2.clone(), database: dbname2.clone() };
                            let pool2 = make_pool_with_size(&cfg2, Some(4)).await?;
                            summary.push_str(&format!("[DB2:{}] Table `{}`\n", dbname2, table2));
                            let cols2 = sqlx::query("SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION").bind(&dbname2).bind(table2).fetch_all(&pool2).await?;
                            let mut actual2: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                            for row in cols2 { use sqlx::Row; let cname: String = row.get("COLUMN_NAME"); let dtype: String = row.get("DATA_TYPE"); let is_null: String = row.get("IS_NULLABLE"); let clen: Option<i64> = row.try_get("CHARACTER_MAXIMUM_LENGTH").ok(); actual2.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null=="YES", clen)); }
                            for name in ["id","uuid","first_name","last_name","birthdate"] { if !actual2.contains_key(name) { summary.push_str(&format!("  - Missing column `{}`\n", name)); } }
                            let idx2 = sqlx::query("SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?").bind(&dbname2).bind(table2).fetch_all(&pool2).await?; let has_id_idx2 = idx2.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" }); if !has_id_idx2 { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname2, table2, table2)); }
                        }
                        anyhow::Ok(())
                    });
                    out.push_str("  \"schema_analysis\": {\n");
                    out.push_str(&format!("    \"summary\": \"{}\",\n", escape(&summary)));
                    out.push_str("    \"index_suggestions\": [\n"); for (i,s) in index_suggestions.iter().enumerate() { if i>0 { out.push_str(",\n"); } out.push_str(&format!("      \"{}\"", escape(s))); } out.push_str("\n    ],\n");
                    out.push_str("    \"grant_suggestions\": [\n"); for (i,s) in grant_suggestions.iter().enumerate() { if i>0 { out.push_str(",\n"); } out.push_str(&format!("      \"{}\"", escape(s))); } out.push_str("\n    ],\n");
                    out.push_str("    \"charset_notes\": [\n"); for (i,s) in charset_notes.iter().enumerate() { if i>0 { out.push_str(",\n"); } out.push_str(&format!("      \"{}\"", escape(s))); } out.push_str("\n    ]\n");
                    out.push_str("  },\n");
                }
                out.push_str("  \"events\": [\n");
                for (i, e) in self.error_events.iter().enumerate() {
                    if i>0 { out.push_str(",\n"); }
                    out.push_str("    {\n");
                    out.push_str(&format!("      \"ts_utc\": \"{}\",\n", escape(&e.ts_utc)));
                    out.push_str(&format!("      \"category\": \"{:?}\",\n", e.category));
                    out.push_str(&format!("      \"message\": \"{}\",\n", escape(&e.message)));
                    out.push_str(&format!("      \"sqlstate\": \"{}\",\n", escape(&e.sqlstate.clone().unwrap_or_default())));
                    out.push_str(&format!("      \"operation\": \"{}\",\n", escape(&e.operation.clone().unwrap_or_default())));
                    out.push_str(&format!("      \"chain\": \"{}\",\n", escape(&e.chain.clone().unwrap_or_default())));
                    out.push_str(&format!("      \"source_action\": \"{}\",\n", escape(&e.source_action)));
                    out.push_str(&format!("      \"db1_host\": \"{}\",\n", escape(&e.db1_host)));
                    out.push_str(&format!("      \"db1_database\": \"{}\",\n", escape(&e.db1_database)));
                    out.push_str(&format!("      \"db2_host\": \"{}\",\n", escape(&e.db2_host.clone().unwrap_or_default())));

                    out.push_str(&format!("      \"db2_database\": \"{}\",\n", escape(&e.db2_database.clone().unwrap_or_default())));
                    out.push_str(&format!("      \"table1\": \"{}\",\n", escape(&e.table1.clone().unwrap_or_default())));
                    out.push_str(&format!("      \"table2\": \"{}\",\n", escape(&e.table2.clone().unwrap_or_default())));
                    out.push_str(&format!("      \"mem_avail_mb\": {},\n", e.mem_avail_mb));
                    out.push_str(&format!("      \"pool_size_cfg\": {}\n", e.pool_size_cfg));
                    out.push_str("    }");
                }
                out.push_str("\n  ],\n");
                out.push_str("  \"suggestions_by_event\": [\n");
                for (i, e) in self.error_events.iter().enumerate() {
                    if i>0 { out.push_str(",\n"); }
                    out.push_str("    [\n");
                    let sugg = suggestions_for(e.category);
                    for (j, s) in sugg.iter().enumerate() {
                        if j>0 { out.push_str(",\n"); }
                        out.push_str(&format!("      \"{}\"", escape(s)));
                    }
                    out.push_str("\n    ]");
                }
                out.push_str("\n  ]\n");
                out.push_str("}\n");
                fs::write(&path, out)?;
                Ok(path)
            }
        }
    }

    fn record_error_with_details(&mut self, message: String, sqlstate: Option<String>, chain: Option<String>, operation: Option<String>) {
        let sanitize = |s: &str| -> String {
            if let Some(pos) = s.find("mysql://") {
                if let Some(at) = s[pos..].find('@') { let mut out = s.to_string(); out.replace_range(pos..pos+at, "mysql://[REDACTED]"); return out; }
            }
            s.to_string()
        };
        let mstats = name_matcher::metrics::memory_stats_mb();
        let pool_sz = self.pool_size.parse::<u32>().unwrap_or(0);
        let envs = ["NAME_MATCHER_POOL_SIZE","NAME_MATCHER_POOL_MIN","NAME_MATCHER_STREAMING","NAME_MATCHER_PARTITION"];
        let mut env_overrides = Vec::new();
        for k in envs { if let Ok(v) = std::env::var(k) { env_overrides.push((k.to_string(), v)); } }
        let cat = categorize_error_with(sqlstate.as_deref(), &message);


        let evt = DiagEvent {
            ts_utc: Utc::now().to_rfc3339(),
            category: cat,
            message: sanitize(&message),
            sqlstate,
            chain,
            operation,
            source_action: self.last_action.clone(),
            db1_host: self.host.clone(),
            db1_database: self.db.clone(),
            db2_host: if self.enable_dual { Some(self.host2.clone()) } else { None },
            db2_database: if self.enable_dual { Some(self.db2.clone()) } else { None },
            table1: self.tables.get(self.table1_idx).cloned(),
            table2: if self.enable_dual { self.tables2.get(self.table2_idx).cloned() } else { self.tables.get(self.table2_idx).cloned() },
            mem_avail_mb: mstats.avail_mb,
            pool_size_cfg: pool_sz,
            env_overrides,
        };
        // Automatic parameter reduction on memory-related resource constraints
        if matches!(cat, ErrorCategory::ResourceConstraint) && message.to_ascii_lowercase().contains("memory") {
            if let Ok(cur_batch) = self.batch_size.parse::<i64>() { let new_batch = (cur_batch / 2).max(10_000); self.batch_size = new_batch.to_string(); }
            if let Ok(cur_thr) = self.mem_thresh.parse::<u64>() { let new_thr = (((cur_thr as f64) * 1.25) as u64).max(1024); self.mem_thresh = new_thr.to_string(); }
            if let Ok(gm) = self.gpu_mem_mb.parse::<u64>() { let gm_new = ((gm as f64) * 0.80).max(256.0) as u64; self.gpu_mem_mb = gm_new.to_string(); }
            if let Ok(pp) = self.gpu_probe_mem_mb.parse::<u64>() { let pp_new = ((pp as f64) * 0.75).max(256.0) as u64; self.gpu_probe_mem_mb = pp_new.to_string(); }
            self.status = "Resource constraint detected; auto-reduced batch and adjusted thresholds. Please retry.".into();
        }

        self.error_events.push(evt);
    }

}


impl App for GuiApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        self.poll_messages();
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                self.ui_top(ui);
            });
        });

        // CUDA Diagnostics Window
        egui::Window::new("CUDA Diagnostics")
            .open(&mut self.cuda_diag_open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Comprehensive CUDA system information:");
                ui.add(egui::TextEdit::multiline(&mut self.cuda_diag_text).desired_width(600.0).desired_rows(20));
            });
        // Applied Settings Summary Window
        egui::Window::new("Applied Settings")
            .open(&mut self.show_maxperf_dialog)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Recommended configuration was applied:");
                ui.add(egui::TextEdit::multiline(&mut self.maxperf_summary).desired_width(600.0).desired_rows(12));
            });

    }
}


fn main() -> eframe::Result<()> {
    let opts = NativeOptions::default();
    eframe::run_native(
        "SRS-II Name Matching Application",
        opts,
        Box::new(|_cc| Ok::<Box<dyn App>, Box<(dyn std::error::Error + Send + Sync + 'static)>>(Box::new(GuiApp::default()))),
    )
}

