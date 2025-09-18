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

use name_matcher::config::DatabaseConfig;
use name_matcher::db::connection::make_pool_with_size;
use name_matcher::db::{get_person_rows, get_person_count};
use name_matcher::matching::{stream_match_csv, stream_match_csv_dual, match_all_progress, match_all_with_opts, ProgressConfig, MatchingAlgorithm, ProgressUpdate, StreamingConfig, StreamControl, ComputeBackend, MatchOptions, GpuConfig};
use name_matcher::export::csv_export::CsvStreamWriter;
use name_matcher::export::xlsx_export::{XlsxStreamWriter, SummaryContext};

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
    gpu_mem_mb: String,

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
    // GPU status
    gpu_total_mb: u64,
    gpu_free_mb: u64,
    gpu_active: bool,

    a1_count: usize,
    a2_count: usize,
    csv_count: usize,
    status: String,

    ctrl_cancel: Option<Arc<AtomicBool>>,
    ctrl_pause: Option<Arc<AtomicBool>>,

    tx: Option<Sender<Msg>>, rx: Receiver<Msg>,
}

impl Default for GuiApp {
    fn default() -> Self {
        let (_tx, rx) = mpsc::channel();
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
            table2_idx: 0,
            algo: MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
            path: "matches.csv".into(),
            fmt: FormatSel::Csv,
            mode: ModeSel::Auto,
            pool_size: "16".into(),
            batch_size: "50000".into(),
            mem_thresh: "800".into(),
            use_gpu: false,
            gpu_mem_mb: "512".into(),
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

            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
            a1_count: 0,
            a2_count: 0,
            csv_count: 0,
            status: "Idle".into(),
            ctrl_cancel: None,
            ctrl_pause: None,
            tx: None, rx,
        }
    }
}

impl GuiApp {
    fn ui_top(&mut self, ui: &mut egui::Ui) {
        ui.heading("🔎 SRS-II Name Matching Application");
        ui.separator();
        ui.label("Database Connection");
        ui.horizontal(|ui| {
        ui.label("Database 1 (Primary)");

            ui.add(TextEdit::singleline(&mut self.host).hint_text("host"));
            ui.add(TextEdit::singleline(&mut self.port).hint_text("port"));
            ui.add(TextEdit::singleline(&mut self.user).hint_text("user"));
            ui.add(TextEdit::singleline(&mut self.pass).hint_text("password").password(true));
            ui.add(TextEdit::singleline(&mut self.db).hint_text("database"));
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.enable_dual, "Enable Cross-Database Matching")
                .on_hover_text("Match Table 1 from Database 1 with Table 2 from Database 2");
        });
        if self.enable_dual {
            ui.label("Database 2 (Secondary)");
            ui.horizontal(|ui| {
                ui.add(TextEdit::singleline(&mut self.host2).hint_text("host"));
                ui.add(TextEdit::singleline(&mut self.port2).hint_text("port"));
                ui.add(TextEdit::singleline(&mut self.user2).hint_text("user"));
                ui.add(TextEdit::singleline(&mut self.pass2).hint_text("password").password(true));
                ui.add(TextEdit::singleline(&mut self.db2).hint_text("database"));
            });
        }

        ui.horizontal(|ui| {
            if ui.button("Load Tables").on_hover_text("Query INFORMATION_SCHEMA to list tables").clicked() { self.load_tables(); }
                if ui.button("Test Connection").on_hover_text("Checks DB connectivity using a lightweight query").clicked() { self.test_connection(); }
                if ui.button("Estimate").on_hover_text("Estimate memory usage and choose a good mode").clicked() { self.estimate(); }
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
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, "Algorithm 1 (first+last+birthdate)");
            ui.radio_value(&mut self.algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, "Algorithm 2 (first+middle+last+birthdate)");
            ui.radio_value(&mut self.algo, MatchingAlgorithm::Fuzzy, "Fuzzy (Levenshtein/Jaro/Winkler; birthdate must match)")
                .on_hover_text("Ensemble of Levenshtein, Jaro, Jaro-Winkler; >=95% Auto-Match, >=85% Review");
        });
        ui.horizontal(|ui| {
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
        ui.collapsing("Advanced", |ui| {
            ui.horizontal(|ui| {
                ui.label("Pool size").on_hover_text("Max connections in SQL pool"); ui.add(TextEdit::singleline(&mut self.pool_size).desired_width(60.0));
                ui.label("Batch size").on_hover_text("Rows fetched per chunk in streaming mode"); ui.add(TextEdit::singleline(&mut self.batch_size).desired_width(80.0));
                ui.label("Mem thresh MB").on_hover_text("Soft minimum free memory before reducing batch size"); ui.add(TextEdit::singleline(&mut self.mem_thresh).desired_width(80.0));
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.use_gpu, "Use GPU (CUDA)").on_hover_text("Enable CUDA acceleration for Fuzzy (Algorithm 3). Falls back to CPU if unavailable.");
                ui.label("GPU Mem Budget (MB)"); ui.add(TextEdit::singleline(&mut self.gpu_mem_mb).desired_width(80.0));
                if ui.button("Auto Optimize").clicked() {
                    // simple heuristic: larger memory on desktop, smaller on laptop
                    self.gpu_mem_mb = if self.mem_avail > 8192 { "1024".into() } else { "512".into() };
            if self.gpu_total_mb > 0 {
                ui.label(format!("GPU: {} MB free / {} MB total | {}", self.gpu_free_mb, self.gpu_total_mb, if self.gpu_active { "active" } else { "idle" }));
            }

                }
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
        ui.horizontal(|ui| {
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
                    Err(e) => { let _ = tx.send(Msg::Error(format!("Failed to load tables: {}", e))); }
                }
            } else {
                let res: Result<Vec<String>> = rt.block_on(async move {
                    let cfg = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname.clone() };
                    let pool = make_pool_with_size(&cfg, Some(8)).await?;
                    let rows = sqlx::query_scalar::<_, String>("SELECT CAST(TABLE_NAME AS CHAR) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")
                        .bind(dbname).fetch_all(&pool).await?;
                    Ok(rows)
                });
                match res { Ok(tables) => { let _ = tx.send(Msg::Info(format!("Loaded {} tables", tables.len()))); let _ = tx.send(Msg::Tables(tables)); }, Err(e) => { let _ = tx.send(Msg::Error(format!("Failed to load tables: {}", e))); } }
            }
        });
        self.rx = rx; // switch to listen on this job
        self.status = "Loading tables...".into();
    }

    fn start(&mut self) {
        if let Err(e) = self.validate() { self.status = format!("Error: {}", e); return; }
        self.running = true; self.progress = 0.0; self.status = "Running...".into(); self.a1_count = 0; self.a2_count = 0; self.csv_count = 0;
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
        let pool_sz = self.pool_size.parse::<u32>().unwrap_or(16);
        let batch = self.batch_size.parse::<i64>().unwrap_or(50_000);

        if matches!(algo, MatchingAlgorithm::Fuzzy) && !matches!(fmt, FormatSel::Csv) {
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

                scfg.checkpoint_path = Some(format!("{}.nmckpt", path));
                match fmt {
                    FormatSel::Csv => {
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                        if matches!(algo, MatchingAlgorithm::Fuzzy) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Fuzzy uses in-memory mode (streaming disabled)".into())); }
                        if use_streaming {
                            // GPU availability info (best-effort)
                            #[cfg(feature = "gpu")]
                            if use_gpu {
                                if let Ok(dev) = cudarc::driver::CudaDevice::new(0) {
                                    let mut free: usize = 0; let mut total: usize = 0;
                                    unsafe { let _ = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _ as *mut _, &mut total as *mut _ as *mut _); }
                                    let _ = tx_for_async.send(Msg::Info(format!("CUDA active | Free {} MB / Total {} MB", (free/1024/1024), (total/1024/1024))));
                                    drop(dev);
                                } else {
                                    let _ = tx_for_async.send(Msg::Info("CUDA requested but unavailable; falling back to CPU".into()));
                                }
                            }

                            let mut w = CsvStreamWriter::create(&path, algo)?;
                            let mut cnt = 0usize;
                            let flush_every = scfg.flush_every;
                            let txp = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, algo, |p| { cnt+=1; w.write(p)?; if cnt % flush_every == 0 { w.flush_partial()?; } Ok(()) }, scfg.clone(), move |u| { let _ = txp.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, algo, |p| { cnt+=1; w.write(p)?; if cnt % flush_every == 0 { w.flush_partial()?; } Ok(()) }, scfg.clone(), move |u| { let _ = txp.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            w.flush()?;
                            Ok((0,0,cnt,path.clone()))
                        } else {
                            // In-memory path
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let txp = tx_for_async.clone();
                            let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp };
                            let pairs = match_all_with_opts(&t1, &t2, algo, mo, move |u| { let _ = txp.send(Msg::Progress(u)); });
                            let mut w = CsvStreamWriter::create(&path, algo)?;
                            for p in &pairs { w.write(p)?; }
                            w.flush()?;
                            Ok((0,0,pairs.len(), path.clone()))
                        }
                    }
                    FormatSel::Xlsx => {
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                            if matches!(algo, MatchingAlgorithm::Fuzzy) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Fuzzy uses in-memory mode (streaming disabled)".into())); }
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
                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: 0, total_table2: 0,
                                matches_algo1: a1, matches_algo2: a2, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0, timestamp: Utc::now()
                            })?;
                            Ok((a1,a2,0,path.clone()))
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let mut xw = XlsxStreamWriter::create(&path)?;
                            let txp1 = tx_for_async.clone();
                            let pairs1 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, cfgp, move |u| { let _ = txp1.send(Msg::Progress(u)); });
                            for p in &pairs1 { xw.append_algo1(p)?; }
                            let txp2 = tx_for_async.clone();
                            let pairs2 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, cfgp, move |u| { let _ = txp2.send(Msg::Progress(u)); });
                            for p in &pairs2 { xw.append_algo2(p)?; }
                            let a1 = pairs1.len(); let a2 = pairs2.len();
                            // GPU availability info (best-effort)
                            #[cfg(feature = "gpu")]
                            if use_gpu {
                                if let Ok(dev) = cudarc::driver::CudaDevice::new(0) {
                                    let mut free: usize = 0; let mut total: usize = 0;
                                    unsafe { let _ = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _ as *mut _, &mut total as *mut _ as *mut _); }
                                    let _ = tx_for_async.send(Msg::Info(format!("CUDA active | Free {} MB / Total {} MB", (free/1024/1024), (total/1024/1024))));
                                    drop(dev);
                                } else {
                                    let _ = tx_for_async.send(Msg::Info("CUDA requested but unavailable; falling back to CPU".into()));
                                }
                            }

                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: 0, total_table2: 0,
                                matches_algo1: a1, matches_algo2: a2, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0, timestamp: Utc::now()
                            })?;
                            Ok((a1,a2,0,path.clone()))
                        }
                    }
                    FormatSel::Both => {
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                            if matches!(algo, MatchingAlgorithm::Fuzzy) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Fuzzy uses in-memory mode (streaming disabled)".into())); }
                        let mut csv_path = path.clone(); if !csv_path.to_ascii_lowercase().ends_with(".csv") { csv_path.push_str(".csv"); }
                        let mut csv_count = 0usize;
                        if use_streaming {
                            let mut w = CsvStreamWriter::create(&csv_path, algo)?;
                            let flush_every = scfg.flush_every;
                            let txp3 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, algo, |p| { csv_count+=1; w.write(p)?; if csv_count % flush_every == 0 { w.flush_partial()?; } Ok(()) }, scfg.clone(), move |u| { let _ = txp3.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, algo, |p| { csv_count+=1; w.write(p)?; if csv_count % flush_every == 0 { w.flush_partial()?; } Ok(()) }, scfg.clone(), move |u| { let _ = txp3.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            w.flush()?;
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let txp = tx_for_async.clone();
                            let pairs = if matches!(algo, MatchingAlgorithm::Fuzzy) {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp };
                                match_all_with_opts(&t1, &t2, algo, mo, move |u| { let _ = txp.send(Msg::Progress(u)); })
                            } else {
                                match_all_progress(&t1, &t2, algo, cfgp, move |u| { let _ = txp.send(Msg::Progress(u)); })
                            };
                            csv_count = pairs.len();
                            let mut w = CsvStreamWriter::create(&csv_path, algo)?;
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
                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: 0, total_table2: 0,
                                matches_algo1: a1, matches_algo2: a2, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0, timestamp: Utc::now()
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
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: 0, total_table2: 0,
                                matches_algo1: a1, matches_algo2: a2, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0, timestamp: Utc::now()
                            })?;
                            Ok((a1,a2,csv_count,path.clone()))
                        }
                    }
                }
            });
            match res { Ok((a1,a2,csv,out_path)) => { let _ = tx.send(Msg::Done { a1, a2, csv, path: out_path }); }, Err(e) => { let _ = tx.send(Msg::Error(format!("{}", e))); } }
        });
    }

    fn reset_state(&mut self) {
        *self = GuiApp::default();
    }
    fn test_connection(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
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
                Err(e) => { let _=tx.send(Msg::Error(format!("Connection failed: {}", e))); }
            }
        });
        self.rx = rx; self.status = "Testing connection...".into();
    }

    fn estimate(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
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
            match res { Ok(s) => { let _=tx.send(Msg::Info(s)); }, Err(e) => { let _=tx.send(Msg::Error(format!("Estimate failed: {}", e))); } }
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
                    let now = std::time::Instant::now();
                    static mut LAST_T: Option<std::time::Instant> = None; static mut LAST_P: usize = 0;
                    unsafe {
                        match LAST_T {
                            Some(t0) => { let dt = now.duration_since(t0).as_secs_f32().max(1e-3); self.rps = ((self.processed.saturating_sub(LAST_P)) as f32 / dt).max(0.0); },
                            None => { self.rps = 0.0; }
                        }
                        LAST_T = Some(now); LAST_P = self.processed;
                    }
                    self.status = format!("{} | {:.1}% | {} / {} recs | {:.0} rec/s", self.stage, u.percent, self.processed, self.total, self.rps);
                }
                Msg::Info(s) => { self.status = s; }
                Msg::Tables(v) => { self.tables = v; self.table1_idx = 0; if self.table1_idx >= self.tables.len() { self.table1_idx = 0; } self.status = format!("Loaded {} tables (DB1)", self.tables.len()); }
                Msg::Tables2(v2) => { self.tables2 = v2; self.table2_idx = 0; if self.table2_idx >= self.tables2.len() { self.table2_idx = 0; } self.status = format!("Loaded {} tables (DB2)", self.tables2.len()); }
                Msg::Done { a1, a2, csv, path } => { self.running = false; self.a1_count = a1; self.a2_count = a2; self.csv_count = csv; self.status = format!("Done. Output: {}", path); self.progress = 100.0; }
                Msg::Error(e) => { self.running = false; self.status = format!("Error: {}", e); }
            }
        }
    }
}

impl App for GuiApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        self.poll_messages();
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                self.ui_top(ui);
            });
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

