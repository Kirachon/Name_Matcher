/// Throughput metrics tracking for Name_Matcher

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Global throughput metrics
static RECORDS_PROCESSED: AtomicU64 = AtomicU64::new(0);
static BATCHES_PROCESSED: AtomicU64 = AtomicU64::new(0);
static TOTAL_PROCESSING_TIME_MS: AtomicU64 = AtomicU64::new(0);

/// Throughput statistics
#[derive(Debug, Clone, Copy)]
pub struct ThroughputStats {
    pub records_processed: u64,
    pub batches_processed: u64,
    pub records_per_second: f64,
    pub avg_batch_duration_ms: u64,
}

/// Track batch processing
pub fn track_batch(records: u64, duration: Duration) {
    RECORDS_PROCESSED.fetch_add(records, Ordering::Relaxed);
    BATCHES_PROCESSED.fetch_add(1, Ordering::Relaxed);
    TOTAL_PROCESSING_TIME_MS.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
}

/// Get throughput statistics
pub fn get_throughput_stats() -> ThroughputStats {
    let records = RECORDS_PROCESSED.load(Ordering::Relaxed);
    let batches = BATCHES_PROCESSED.load(Ordering::Relaxed);
    let total_time_ms = TOTAL_PROCESSING_TIME_MS.load(Ordering::Relaxed);
    
    let records_per_second = if total_time_ms > 0 {
        (records as f64 * 1000.0) / total_time_ms as f64
    } else {
        0.0
    };
    
    let avg_batch_duration_ms = if batches > 0 {
        total_time_ms / batches
    } else {
        0
    };
    
    ThroughputStats {
        records_processed: records,
        batches_processed: batches,
        records_per_second,
        avg_batch_duration_ms,
    }
}

/// Reset throughput metrics
pub fn reset_throughput_metrics() {
    RECORDS_PROCESSED.store(0, Ordering::Relaxed);
    BATCHES_PROCESSED.store(0, Ordering::Relaxed);
    TOTAL_PROCESSING_TIME_MS.store(0, Ordering::Relaxed);
}

/// Batch processing timer
pub struct BatchTimer {
    start: Instant,
    records: u64,
}

impl BatchTimer {
    pub fn new(records: u64) -> Self {
        Self {
            start: Instant::now(),
            records,
        }
    }

    pub fn finish(self) {
        let duration = self.start.elapsed();
        track_batch(self.records, duration);
    }
}

/// Throughput monitor for periodic logging
pub struct ThroughputMonitor {
    start: Instant,
    last_log: Instant,
    log_interval: Duration,
    last_records: u64,
}

impl ThroughputMonitor {
    pub fn new(log_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_log: now,
            log_interval,
            last_records: 0,
        }
    }

    /// Check if it's time to log and return current throughput
    pub fn check_and_log(&mut self) -> Option<ThroughputStats> {
        let now = Instant::now();
        if now.duration_since(self.last_log) >= self.log_interval {
            self.last_log = now;
            
            let stats = get_throughput_stats();
            let records_since_last = stats.records_processed - self.last_records;
            self.last_records = stats.records_processed;
            
            log::info!(
                "Throughput: {} records/sec (total: {} records, {} batches)",
                stats.records_per_second,
                stats.records_processed,
                stats.batches_processed
            );
            
            Some(stats)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_tracking() {
        reset_throughput_metrics();
        
        track_batch(1000, Duration::from_millis(100));
        track_batch(2000, Duration::from_millis(200));
        
        let stats = get_throughput_stats();
        assert_eq!(stats.records_processed, 3000);
        assert_eq!(stats.batches_processed, 2);
        assert_eq!(stats.avg_batch_duration_ms, 150);
        
        // 3000 records in 300ms = 10,000 records/sec
        assert!((stats.records_per_second - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_batch_timer() {
        reset_throughput_metrics();
        
        let timer = BatchTimer::new(1000);
        std::thread::sleep(Duration::from_millis(10));
        timer.finish();
        
        let stats = get_throughput_stats();
        assert_eq!(stats.records_processed, 1000);
        assert_eq!(stats.batches_processed, 1);
    }
}

