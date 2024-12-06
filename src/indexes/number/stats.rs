use std::sync::atomic::{AtomicU64, Ordering};




// const SIZE: usize = 128; // Number of seconds to track

pub struct PageUsage<const N: usize> {
    buckets: [AtomicU64; N],
}

impl<const N: usize> PageUsage<N> {
    fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; N],
        }
    }

    /// Increment the counter for the current epoch (timestamp in seconds).
    fn increment(&self, epoch: u64) {
        let idx = (epoch % N as u64) as usize;

        // Reset the bucket if it belongs to an old epoch
        let bucket_epoch = epoch - (epoch % N as u64);
        let current_value = self.buckets[idx].load(Ordering::Relaxed);

        if current_value != 0 && current_value >> 32 != bucket_epoch {
            // Reset the counter to attach the new epoch
            self.buckets[idx].store(bucket_epoch << 32, Ordering::Relaxed);
        }

        // Increment the lower 32 bits (the counter)
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
    }

    // Get the counter for a specific epoch
    // fn get(&self, epoch: u64) -> usize {
    //     let idx = (epoch % N as u64) as usize;
    //     let current_value = self.buckets[idx].load(Ordering::Relaxed);
    // 
    //     // Check if the epoch matches; otherwise, the value is outdated
    //     if current_value >> 32 == epoch {
    //         (current_value & 0xFFFF_FFFF) as usize
    //     } else {
    //         0
    //     }
    // }
}