/// Database connection pool metrics tracking

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Global database metrics
static ACQUIRE_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
static POOL_EXHAUSTED: AtomicU64 = AtomicU64::new(0);
static TOTAL_ACQUIRES: AtomicU64 = AtomicU64::new(0);
static TOTAL_ACQUIRE_TIME_MS: AtomicU64 = AtomicU64::new(0);

/// Database pool statistics
#[derive(Debug, Clone, Copy)]
pub struct DatabasePoolStats {
    pub active_connections: u32,
    pub idle_connections: u32,
    pub max_connections: u32,
    pub acquire_timeouts: u64,
    pub pool_exhausted_events: u64,
    pub total_acquires: u64,
    pub avg_acquire_time_ms: u64,
}

/// Track connection acquire timeout
pub fn track_acquire_timeout() {
    ACQUIRE_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
}

/// Track pool exhaustion event
pub fn track_pool_exhausted() {
    POOL_EXHAUSTED.fetch_add(1, Ordering::Relaxed);
}

/// Track connection acquire
pub fn track_acquire(duration_ms: u64) {
    TOTAL_ACQUIRES.fetch_add(1, Ordering::Relaxed);
    TOTAL_ACQUIRE_TIME_MS.fetch_add(duration_ms, Ordering::Relaxed);
}

/// Get database pool statistics
pub fn get_pool_stats(active: u32, idle: u32, max: u32) -> DatabasePoolStats {
    let total_acquires = TOTAL_ACQUIRES.load(Ordering::Relaxed);
    let total_time = TOTAL_ACQUIRE_TIME_MS.load(Ordering::Relaxed);
    let avg_time = if total_acquires > 0 {
        total_time / total_acquires
    } else {
        0
    };

    DatabasePoolStats {
        active_connections: active,
        idle_connections: idle,
        max_connections: max,
        acquire_timeouts: ACQUIRE_TIMEOUTS.load(Ordering::Relaxed),
        pool_exhausted_events: POOL_EXHAUSTED.load(Ordering::Relaxed),
        total_acquires,
        avg_acquire_time_ms: avg_time,
    }
}

/// Reset database metrics
pub fn reset_database_metrics() {
    ACQUIRE_TIMEOUTS.store(0, Ordering::Relaxed);
    POOL_EXHAUSTED.store(0, Ordering::Relaxed);
    TOTAL_ACQUIRES.store(0, Ordering::Relaxed);
    TOTAL_ACQUIRE_TIME_MS.store(0, Ordering::Relaxed);
}

/// Connection acquire timer
pub struct AcquireTimer {
    start: Instant,
}

impl AcquireTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn finish(self) {
        let duration_ms = self.start.elapsed().as_millis() as u64;
        track_acquire(duration_ms);
    }
}

impl Default for AcquireTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_tracking() {
        reset_database_metrics();
        track_acquire_timeout();
        track_acquire_timeout();
        
        let stats = get_pool_stats(5, 10, 64);
        assert_eq!(stats.acquire_timeouts, 2);
    }

    #[test]
    fn test_acquire_tracking() {
        reset_database_metrics();
        track_acquire(100);
        track_acquire(200);
        
        let stats = get_pool_stats(5, 10, 64);
        assert_eq!(stats.total_acquires, 2);
        assert_eq!(stats.avg_acquire_time_ms, 150);
    }
}

