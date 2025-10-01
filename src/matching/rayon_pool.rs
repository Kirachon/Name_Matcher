/// Dedicated Rayon thread pool for CPU-intensive matching operations
/// 
/// This module provides a custom Rayon thread pool to avoid contention with
/// the Tokio async runtime. By isolating CPU-intensive operations (like
/// parallel normalization) in a dedicated pool, we can achieve better
/// performance and avoid potential deadlocks.

use once_cell::sync::Lazy;
use rayon::ThreadPool;
use std::sync::Arc;

/// Global dedicated Rayon thread pool for matching operations
static MATCHING_POOL: Lazy<Arc<ThreadPool>> = Lazy::new(|| {
    let num_threads = get_optimal_thread_count();
    
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("name-matcher-{}", i))
            .build()
            .expect("Failed to create dedicated Rayon thread pool")
    )
});

/// Get the optimal thread count for the matching pool
fn get_optimal_thread_count() -> usize {
    // Check environment variable first
    if let Ok(val) = std::env::var("NAME_MATCHER_RAYON_THREADS") {
        if let Ok(n) = val.parse::<usize>() {
            if n > 0 {
                log::info!("[Rayon Pool] Using {} threads from NAME_MATCHER_RAYON_THREADS", n);
                return n;
            }
        }
    }
    
    // Default: use all available cores
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    
    // T1-CONC-01: Reserve at most 1 core on high-core machines; otherwise use all cores
    let reserved = if cores > 16 { 1 } else { 0 };
    let threads = cores.saturating_sub(reserved).max(1);
    
    log::info!("[Rayon Pool] Created dedicated thread pool with {} threads (total cores: {}, reserved: {})", 
        threads, cores, reserved);
    
    threads
}

/// Get the global matching thread pool
pub fn get_pool() -> Arc<ThreadPool> {
    MATCHING_POOL.clone()
}

/// Execute a closure in the dedicated matching pool
pub fn execute<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    get_pool().install(f)
}

/// Execute a parallel iterator operation in the dedicated matching pool
pub fn par_iter_execute<I, F, R>(iter: I, f: F) -> R
where
    I: rayon::iter::IntoParallelIterator + Send,
    F: FnOnce(I::Iter) -> R + Send,
    R: Send,
    I::Iter: rayon::iter::ParallelIterator,
{
    get_pool().install(|| f(iter.into_par_iter()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_pool_creation() {
        let pool = get_pool();
        assert!(pool.current_num_threads() > 0);
    }

    #[test]
    fn test_execute() {
        let result = execute(|| {
            (0..100).into_par_iter().sum::<i32>()
        });
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_par_iter_execute() {
        let data: Vec<i32> = (0..1000).collect();
        let result = par_iter_execute(&data, |iter| {
            iter.map(|&x| x * 2).sum::<i32>()
        });
        assert_eq!(result, 999000);
    }

    #[test]
    fn test_thread_isolation() {
        // Verify that the pool uses dedicated threads
        let pool = get_pool();
        let thread_name = pool.install(|| {
            std::thread::current().name().unwrap_or("").to_string()
        });
        assert!(thread_name.contains("name-matcher"));
    }
}

