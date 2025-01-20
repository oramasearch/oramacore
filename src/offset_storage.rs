use std::sync::atomic::{AtomicU64, Ordering};

use crate::collection_manager::sides::Offset;

#[derive(Debug, Default)]
pub struct OffsetStorage {
    offset: AtomicU64,
}

impl OffsetStorage {
    pub fn new() -> Self {
        Self {
            offset: AtomicU64::new(0),
        }
    }

    pub fn set_offset(&self, offset: Offset) {
        self.offset.store(offset.0, Ordering::SeqCst);
    }

    pub fn get_offset(&self) -> Offset {
        // Don't change `SeqCst`: it's important to have a consistent view of the offset
        // Commonly this `get_offset` is called having a lock,
        // so it's important the CPU doens't reorder the instructions
        Offset(self.offset.load(Ordering::SeqCst))
    }
}
