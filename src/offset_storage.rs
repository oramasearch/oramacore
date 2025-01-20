use std::sync::atomic::AtomicU64;

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
        self.offset
            .store(offset.0, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn get_offset(&self) -> Offset {
        Offset(self.offset.load(std::sync::atomic::Ordering::SeqCst))
    }
}
