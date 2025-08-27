//! Common offload utility functions that can be used by field types that support loading/unloading.

use crate::collection_manager::sides::read::OffloadFieldConfig;
use invocation_counter::InvocationCounter;
use std::time::UNIX_EPOCH;

/// Utility function to get current Unix timestamp
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug)]
pub enum InnerCommittedField<Loaded, Stats, FieldInfo> {
    Loaded {
        field: Loaded,
        counter: InvocationCounter,
        offload_config: OffloadFieldConfig,
    },
    Unloaded {
        offload_config: OffloadFieldConfig,
        stats: Stats,
        field_info: FieldInfo,
    },
}
impl<Loaded, Stats, FieldInfo> InnerCommittedField<Loaded, Stats, FieldInfo> {
    pub fn new_loaded(field: Loaded, offload_config: OffloadFieldConfig) -> Self {
        let counter = create_counter(offload_config.slot_count_exp, offload_config.slot_size_exp);
        InnerCommittedField::Loaded {
            field,
            counter,
            offload_config,
        }
    }

    pub fn unloaded(
        offload_config: OffloadFieldConfig,
        stats: Stats,
        field_info: FieldInfo,
    ) -> Self {
        InnerCommittedField::Unloaded {
            offload_config,
            stats,
            field_info,
        }
    }

    pub fn loaded(&self) -> bool {
        match self {
            InnerCommittedField::Loaded { .. } => true,
            InnerCommittedField::Unloaded { .. } => false,
        }
    }

    pub fn stats(&self) -> Stats
    where
        Stats: Clone,
        Loaded: LoadedField<Stats = Stats, Info = FieldInfo>,
    {
        match self {
            InnerCommittedField::Loaded { field, .. } => field.stats(),
            InnerCommittedField::Unloaded { stats, .. } => stats.clone(),
        }
    }

    pub fn info(&self) -> FieldInfo
    where
        FieldInfo: Clone,
        Loaded: LoadedField<Stats = Stats, Info = FieldInfo>,
    {
        match self {
            InnerCommittedField::Loaded { field, .. } => field.info(),
            InnerCommittedField::Unloaded { field_info, .. } => field_info.clone(),
        }
    }

    pub fn get_load_unchecked(&self) -> Option<&Loaded> {
        match self {
            InnerCommittedField::Loaded { field, counter, .. } => {
                counter.register(now());

                Some(field)
            }
            InnerCommittedField::Unloaded { .. } => None,
        }
    }

    /// Check if a counter-equipped loaded field should be unloaded based on usage.
    /// Returns true if the field should be unloaded (hasn't been used recently).
    pub fn should_unload(&self) -> bool {
        match self {
            InnerCommittedField::Loaded {
                offload_config,
                counter,
                ..
            } => {
                let now = now();
                let start = now - offload_config.unload_window.as_secs();
                let c = counter.count_in(start, now);

                c == 0
            }
            InnerCommittedField::Unloaded { .. } => false,
        }
    }
}

fn create_counter(slot_count_exp: u8, slot_size_exp: u8) -> InvocationCounter {
    // 2^8 * 2^4 = 4096 seconds = 1 hour, 6 minutes, and 36 seconds
    let counter = InvocationCounter::new(slot_count_exp, slot_size_exp);

    // Avoid to unload committed string field if it is created right now.
    counter.register(now());

    counter
}

pub trait LoadedField {
    type Stats;
    type Info;

    fn stats(&self) -> Self::Stats;
    fn info(&self) -> Self::Info;
}
