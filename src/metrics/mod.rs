/// Production metrics collection for Name_Matcher
/// Provides lightweight metrics tracking for memory, database, throughput, and GPU

use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub mod memory;
pub mod database;
pub mod throughput;

// Re-export commonly used functions for backward compatibility
pub use memory::memory_stats_mb;

/// Global metrics collector
static METRICS: once_cell::sync::Lazy<Arc<Mutex<MetricsCollector>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(MetricsCollector::new())));

/// Get the global metrics collector
pub fn get_metrics() -> Arc<Mutex<MetricsCollector>> {
    METRICS.clone()
}

/// Central metrics collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollector {
    pub start_time: std::time::SystemTime,
    pub memory: MemoryMetrics,
    pub database: DatabaseMetrics,
    pub throughput: ThroughputMetrics,
    pub gpu: GpuMetrics,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: std::time::SystemTime::now(),
            memory: MemoryMetrics::default(),
            database: DatabaseMetrics::default(),
            throughput: ThroughputMetrics::default(),
            gpu: GpuMetrics::default(),
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.start_time = std::time::SystemTime::now();
        self.memory = MemoryMetrics::default();
        self.database = DatabaseMetrics::default();
        self.throughput = ThroughputMetrics::default();
        self.gpu = GpuMetrics::default();
    }

    /// Export metrics as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export metrics as CSV line
    pub fn to_csv_header() -> String {
        "timestamp,heap_used_mb,heap_peak_mb,vram_used_mb,vram_free_mb,db_active_conn,db_idle_conn,db_timeouts,records_processed,throughput_rps,gpu_kernel_launches,gpu_oom_events".to_string()
    }

    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            self.start_time.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            self.memory.heap_used_mb,
            self.memory.heap_peak_mb,
            self.gpu.vram_used_mb,
            self.gpu.vram_free_mb,
            self.database.active_connections,
            self.database.idle_connections,
            self.database.acquire_timeouts,
            self.throughput.records_processed,
            self.throughput.records_per_second,
            self.gpu.kernel_launches,
            self.gpu.oom_events
        )
    }

    /// Log metrics to console
    pub fn log_metrics(&self) {
        log::info!("=== Metrics Snapshot ===");
        log::info!("Memory: used={} MB, peak={} MB", self.memory.heap_used_mb, self.memory.heap_peak_mb);
        log::info!("VRAM: used={} MB, free={} MB", self.gpu.vram_used_mb, self.gpu.vram_free_mb);
        log::info!("Database: active={}, idle={}, timeouts={}", 
            self.database.active_connections, self.database.idle_connections, self.database.acquire_timeouts);
        log::info!("Throughput: {} records, {} records/sec", 
            self.throughput.records_processed, self.throughput.records_per_second);
        log::info!("GPU: {} kernel launches, {} OOM events", 
            self.gpu.kernel_launches, self.gpu.oom_events);
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub heap_used_mb: u64,
    pub heap_peak_mb: u64,
    pub allocations_total: u64,
    pub deallocations_total: u64,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            heap_used_mb: 0,
            heap_peak_mb: 0,
            allocations_total: 0,
            deallocations_total: 0,
        }
    }
}

/// Database connection pool metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetrics {
    pub active_connections: u32,
    pub idle_connections: u32,
    pub max_connections: u32,
    pub acquire_duration_ms: u64,
    pub acquire_timeouts: u64,
    pub pool_exhausted_events: u64,
}

impl Default for DatabaseMetrics {
    fn default() -> Self {
        Self {
            active_connections: 0,
            idle_connections: 0,
            max_connections: 0,
            acquire_duration_ms: 0,
            acquire_timeouts: 0,
            pool_exhausted_events: 0,
        }
    }
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub records_processed: u64,
    pub records_per_second: f64,
    pub batch_duration_ms: u64,
    pub batches_processed: u64,
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            records_processed: 0,
            records_per_second: 0.0,
            batch_duration_ms: 0,
            batches_processed: 0,
        }
    }
}

/// GPU metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
    pub kernel_launches: u64,
    pub oom_events: u64,
    pub cpu_fallbacks: u64,
}

impl Default for GpuMetrics {
    fn default() -> Self {
        Self {
            vram_total_mb: 0,
            vram_used_mb: 0,
            vram_free_mb: 0,
            kernel_launches: 0,
            oom_events: 0,
            cpu_fallbacks: 0,
        }
    }
}

/// Update memory metrics
pub fn update_memory_metrics(used_mb: u64, peak_mb: u64) {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.memory.heap_used_mb = used_mb;
    if peak_mb > m.memory.heap_peak_mb {
        m.memory.heap_peak_mb = peak_mb;
    }
}

/// Update database metrics
pub fn update_database_metrics(active: u32, idle: u32, max: u32) {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.database.active_connections = active;
    m.database.idle_connections = idle;
    m.database.max_connections = max;
}

/// Increment database timeout counter
pub fn increment_db_timeout() {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.database.acquire_timeouts += 1;
}

/// Increment pool exhausted counter
pub fn increment_pool_exhausted() {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.database.pool_exhausted_events += 1;
}

/// Update throughput metrics
pub fn update_throughput_metrics(records: u64, duration: Duration) {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.throughput.records_processed += records;
    m.throughput.batches_processed += 1;
    m.throughput.batch_duration_ms = duration.as_millis() as u64;
    
    // Calculate records per second
    if duration.as_secs_f64() > 0.0 {
        m.throughput.records_per_second = records as f64 / duration.as_secs_f64();
    }
}

/// Update GPU metrics
pub fn update_gpu_metrics(total_mb: u64, free_mb: u64) {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.gpu.vram_total_mb = total_mb;
    m.gpu.vram_free_mb = free_mb;
    m.gpu.vram_used_mb = total_mb.saturating_sub(free_mb);
}

/// Increment GPU kernel launch counter
pub fn increment_gpu_kernel_launch() {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.gpu.kernel_launches += 1;
}

/// Increment GPU OOM event counter
pub fn increment_gpu_oom() {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.gpu.oom_events += 1;
}

/// Increment GPU→CPU fallback counter
pub fn increment_gpu_fallback() {
    let metrics = get_metrics();
    let mut m = metrics.lock().unwrap();
    m.gpu.cpu_fallbacks += 1;
}

/// Metrics snapshot for periodic logging
pub struct MetricsSnapshot {
    start: Instant,
    last_log: Instant,
    log_interval: Duration,
}

impl MetricsSnapshot {
    pub fn new(log_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_log: now,
            log_interval,
        }
    }

    /// Check if it's time to log metrics
    pub fn should_log(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.last_log) >= self.log_interval {
            self.last_log = now;
            true
        } else {
            false
        }
    }

    /// Log metrics if interval elapsed
    pub fn log_if_ready(&mut self) {
        if self.should_log() {
            let metrics = get_metrics();
            let m = metrics.lock().unwrap();
            m.log_metrics();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        collector.memory.heap_used_mb = 100;
        collector.memory.heap_peak_mb = 150;
        
        let json = collector.to_json().unwrap();
        assert!(json.contains("heap_used_mb"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_csv_export() {
        let collector = MetricsCollector::new();
        let header = MetricsCollector::to_csv_header();
        let line = collector.to_csv_line();
        
        assert!(header.contains("heap_used_mb"));
        assert!(line.split(',').count() == header.split(',').count());
    }

    #[test]
    fn test_metrics_update() {
        update_memory_metrics(100, 150);
        update_database_metrics(5, 10, 64);
        increment_db_timeout();
        increment_gpu_kernel_launch();
        
        let metrics = get_metrics();
        let m = metrics.lock().unwrap();
        
        assert_eq!(m.memory.heap_used_mb, 100);
        assert_eq!(m.database.active_connections, 5);
        assert_eq!(m.database.acquire_timeouts, 1);
        assert_eq!(m.gpu.kernel_launches, 1);
    }
}

