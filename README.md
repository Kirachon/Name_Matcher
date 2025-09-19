# SRS-II Name Matching Application

Creator/Author: Matthias Tangonan

SRS-II Name Matching Application is a high‑performance Rust tool for matching person records across MySQL tables. It supports both single‑database and cross‑database workflows, offers exact and fuzzy matching, and exports results to CSV or Excel (XLSX). A desktop GUI (eframe/egui) and a CLI are provided.

---

## Table of Contents
- Overview and Purpose
- Features
- Matching Algorithms
- Installation and Setup
- Usage (CLI and GUI)
- Cross‑Database Configuration (Environment Variables)
- Export Formats
- Performance Considerations (Streaming vs In‑Memory)
- System Requirements
- Troubleshooting
- Attribution

---

## Overview and Purpose
SRS‑II helps reconcile and deduplicate person records between two tables that may live in the same MySQL database or in two different MySQL servers/databases. It focuses on correctness and robustness while offering fast processing for large datasets via streaming.

Typical use cases:
- Deduplicating records within a database table
- Matching two tables across different databases/servers (e.g., reference data vs. imported data)
- Producing reviewable outputs for human validation or downstream pipelines

---

## Features
- Single‑database matching (both tables from DB1)
- Cross‑database matching with dual MySQL pools (DB1 + DB2)
- Exact matching algorithms (deterministic key‑based options)
- Fuzzy matching (Levenshtein, Jaro, Jaro‑Winkler ensemble)
- Two execution modes: Streaming (low memory) and In‑Memory (fast on smaller data)
- CSV and XLSX export (streaming XLSX summary supported)
- CLI and desktop GUI
- Resilient Unicode handling for names

---

## Matching Algorithms
Currently supported algorithms (select at runtime):

1) Algorithm 1 (first+last+birthdate) — deterministic
2) Algorithm 2 (first+middle+last+birthdate) — deterministic
3) Fuzzy (Levenshtein/Jaro/Jaro‑Winkler ensemble; birthdate must match)

Notes:
- The fuzzy ensemble uses strong thresholds (e.g., >=95% auto‑match, >=85% review) and robust Unicode normalization for reliability.
- Fuzzy mode currently supports CSV export for full details; XLSX provides an aggregated/summary‑style streaming export.

---

## Installation and Setup

### Prerequisites
- Rust toolchain (2021+ edition; tested on stable)
- MySQL 5.7/8.0 or compatible server(s)
- For GUI: a desktop environment capable of running eframe/egui apps

### Build
```
cargo build
```

### Optional GPU Support
A GPU feature flag is scaffolded but off by default. If you plan to extend GPU support, enable the `gpu` feature after ensuring CUDA toolchain availability.

---

## Usage

### CLI
```
Usage: name_matcher <host> <port> <user> <password> <database> <table1> <table2> <algo:1|2|3> <out_path> [format: csv|xlsx|both]
```
Examples:
```
name_matcher 127.0.0.1 3306 root secret db people_a people_b 1 D:/out/matches.csv
name_matcher 127.0.0.1 3306 root secret db people_a people_b 2 D:/out/matches.xlsx xlsx
name_matcher 127.0.0.1 3306 root secret db people_a people_b 3 D:/out/matches both
```

To enable cross‑database mode for CLI, set the DB2_* environment variables (see below). When set, `table1` is read from DB1 and `table2` from DB2.

### GUI
Run the desktop GUI:
```
cargo run --bin gui
```
Key steps in the GUI:
- Fill Database 1 (Primary) connection fields
- Optionally check “Enable Cross‑Database Matching” to reveal Database 2 (Secondary)
- Click “Load Tables”; select Table 1 (DB1) and Table 2 (DB2) in dual‑DB mode (or both from DB1 in single‑DB mode)
- Choose algorithm and mode (Auto/Streaming/In‑Memory)
- Set output path and format (CSV / XLSX)
- Click Start

---

## Cross‑Database Configuration (Environment Variables)
Set these to enable dual‑database mode in both CLI and internal logic:
- DB2_HOST
- DB2_PORT (optional; defaults to DB1 port)
- DB2_USER (optional; defaults to DB1 user)
- DB2_PASS (optional; defaults to DB1 password)
- DB2_DATABASE (optional; defaults to DB1 database)

Example (PowerShell):
```
$env:DB2_HOST="192.168.1.55"
$env:DB2_PORT="3306"
$env:DB2_USER="root"
$env:DB2_PASS="secret"
$env:DB2_DATABASE="other_db"
```

With DB2_* set, the application connects to two independent MySQL pools and matches Table 1 (DB1) to Table 2 (DB2).

---

## Export Formats
- CSV: full detail, line‑by‑line results (preferred for fuzzy runs)
- XLSX: streaming workbook with algorithm sheets and a summary (optimized for larger jobs with summaries)

---

## Performance Considerations
- Auto mode chooses between Streaming and In‑Memory based on dataset sizes and algorithm selection.
- Streaming mode indexes the smaller table and streams the larger one; ideal for large datasets and limited memory.
- In‑Memory mode loads both tables into RAM first; faster for small/medium datasets.
- Heuristics and environment flags:
  - NAME_MATCHER_STREAMING=1 forces streaming
  - NAME_MATCHER_PARTITION sets partition strategy in streaming (e.g., “last_initial”)

---

## System Requirements
- Windows/Linux/macOS supported by Rust and eframe/egui
- MySQL 5.7/8.0 or compatible
- Memory: depends on mode and dataset sizes (Streaming mode minimizes peak memory)

---

## Troubleshooting
- Connection errors: verify host/port/user/password/database; try “Test Connection” in GUI.
- Missing tables: click “Load Tables” after entering DB credentials; ensure permissions.
- Cross‑database not active: confirm DB2_* environment variables (CLI) or GUI toggle is enabled and DB2 fields filled.
- Fuzzy + XLSX: detailed per‑match CSV is recommended; XLSX provides a streaming summary.
- Performance: if memory is constrained or datasets are large, use Streaming; otherwise In‑Memory can be faster.

---

## Attribution
- Creator/Author: Matthias Tangonan
- Core Application: SRS‑II Name Matching Application (Rust)
- GUI: eframe/egui

If you use or extend this tool, please credit Matthias Tangonan and consider contributing improvements back via pull requests.



---

## Performance Tuning and Auto-Optimization

The GUI now provides an Auto Optimize button (Advanced section) that:
- Detects available RAM and computes an adaptive StreamingConfig (batch_size, flush_every)
- Suggests connection pool size from CPU cores (defaults to min(32, 2 x cores))
- Picks Streaming mode for Algorithms 1/2, and In‑Memory for Fuzzy
- Adjusts GPU memory budget suggestion (if enabled)
- SSD storage toggle tunes flush frequency for faster buffered writes

Notes:
- Streaming mode is recommended for large tables and low‑memory systems
- Fuzzy (Algorithm 3): GUI runs in In‑Memory mode for stability. CLI may stream fuzzy when heuristics or NAME_MATCHER_STREAMING=1 select streaming. CSV format only for fuzzy.

### Required Database Indexes
For best performance, ensure the following indexes exist on both tables:
- PRIMARY KEY or unique index on `id`
- BTREE indexes on: `last_name`, `first_name`, `birthdate`
- Optional but helpful for partitioning strategies: prefix/initial indexes on `last_name` and `first_name` (e.g., first character), or a computed birth year column

Example (MySQL):
```
ALTER TABLE people ADD INDEX idx_last_name(last_name),
                   ADD INDEX idx_first_name(first_name),
                   ADD INDEX idx_birthdate(birthdate);
```

### Connection Pool Sizing
- Default pool size (when not overridden) is derived from CPU cores: `min(32, 2 x cores)`
- Override via env: `NAME_MATCHER_POOL_SIZE`, `NAME_MATCHER_POOL_MIN`

- Defaults: minimum connections = 4; max defaults to min(32, 2 x cores) if not set
- Additional tuning env: NAME_MATCHER_ACQUIRE_MS, NAME_MATCHER_IDLE_MS, NAME_MATCHER_LIFETIME_MS

### Streaming Configuration
- Adaptive batch size is computed from available memory and clamped to [5,000; 100,000]
- Export writers flush every ≈ 10% of the batch (min 1,000)
- Pipelined prefetch keeps one next chunk ahead to reduce stalls while maintaining a low memory footprint

- Checkpoint files (`.nmckpt`) are written next to output files to enable resume

---
