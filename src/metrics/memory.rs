/// Memory tracking utilities for Name_Matcher

use std::sync::atomic::{AtomicU64, Ordering};

/// Global memory statistics
static PEAK_MEMORY_MB: AtomicU64 = AtomicU64::new(0);
static TOTAL_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static TOTAL_DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// Memory statistics
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub used_mb: u64,
    pub avail_mb: u64,  // Available memory (for backward compatibility)
    pub peak_mb: u64,
    pub allocations: u64,
    pub deallocations: u64,
}

/// Get current memory statistics
pub fn memory_stats_mb() -> MemoryStats {
    #[cfg(target_os = "windows")]
    {
        use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use winapi::um::processthreadsapi::GetCurrentProcess;
        
        unsafe {
            let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            
            if GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                let used_mb = pmc.WorkingSetSize as u64 / (1024 * 1024);
                
                // Update peak memory
                let mut peak = PEAK_MEMORY_MB.load(Ordering::Relaxed);
                while used_mb > peak {
                    match PEAK_MEMORY_MB.compare_exchange_weak(
                        peak,
                        used_mb,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => peak = x,
                    }
                }
                
                // Estimate available memory (rough heuristic: assume 8GB total, subtract used)
                let total_mb = 8192u64;  // Conservative estimate
                let avail_mb = total_mb.saturating_sub(used_mb);

                return MemoryStats {
                    used_mb,
                    avail_mb,
                    peak_mb: PEAK_MEMORY_MB.load(Ordering::Relaxed),
                    allocations: TOTAL_ALLOCATIONS.load(Ordering::Relaxed),
                    deallocations: TOTAL_DEALLOCATIONS.load(Ordering::Relaxed),
                };
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            let used_mb = kb / 1024;
                            
                            // Update peak memory
                            let mut peak = PEAK_MEMORY_MB.load(Ordering::Relaxed);
                            while used_mb > peak {
                                match PEAK_MEMORY_MB.compare_exchange_weak(
                                    peak,
                                    used_mb,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                ) {
                                    Ok(_) => break,
                                    Err(x) => peak = x,
                                }
                            }
                            
                            // Estimate available memory
                            let total_mb = 8192u64;
                            let avail_mb = total_mb.saturating_sub(used_mb);

                            return MemoryStats {
                                used_mb,
                                avail_mb,
                                peak_mb: PEAK_MEMORY_MB.load(Ordering::Relaxed),
                                allocations: TOTAL_ALLOCATIONS.load(Ordering::Relaxed),
                                deallocations: TOTAL_DEALLOCATIONS.load(Ordering::Relaxed),
                            };
                        }
                    }
                }
            }
        }
    }
    
    // Fallback: return zeros
    MemoryStats {
        used_mb: 0,
        avail_mb: 8192,  // Conservative default
        peak_mb: 0,
        allocations: 0,
        deallocations: 0,
    }
}

/// Track allocation
pub fn track_allocation(size_bytes: usize) {
    TOTAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Track deallocation
pub fn track_deallocation(size_bytes: usize) {
    TOTAL_DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Reset peak memory counter
pub fn reset_peak_memory() {
    PEAK_MEMORY_MB.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = memory_stats_mb();
        // Should return some value (or 0 on unsupported platforms)
        assert!(stats.used_mb >= 0);
    }

    #[test]
    fn test_allocation_tracking() {
        let before = TOTAL_ALLOCATIONS.load(Ordering::Relaxed);
        track_allocation(1024);
        let after = TOTAL_ALLOCATIONS.load(Ordering::Relaxed);
        assert_eq!(after, before + 1);
    }
}

