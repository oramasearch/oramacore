use std::{
    ops::{Deref, DerefMut},
    time::{Duration, UNIX_EPOCH},
};

use invocation_counter::InvocationCounter;

use crate::collection_manager::sides::read::OffloadFieldConfig;

pub fn create_counter(offload_config: OffloadFieldConfig) -> InvocationCounter {
    // 2^8 * 2^4 = 4096 seconds = 1 hour, 6 minutes, and 36 seconds
    let counter =
        InvocationCounter::new(offload_config.slot_count_exp, offload_config.slot_size_exp);

    // Avoid to unload committed string field if it is created right now.
    counter.register(now());

    counter
}
fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub enum Cow<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}
impl<'a, T> Deref for Cow<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Cow::Borrowed(b) => b,
            Cow::Owned(o) => o,
        }
    }
}

pub enum MutRef<'a, T> {
    Borrowed(&'a mut T),
    Owned(T),
}
impl<'a, T> Deref for MutRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            MutRef::Borrowed(b) => b,
            MutRef::Owned(o) => o,
        }
    }
}
impl<'a, T> DerefMut for MutRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            MutRef::Borrowed(b) => b,
            MutRef::Owned(o) => o,
        }
    }
}

pub fn should_offload(invocation_counter: &InvocationCounter, unload_window: Duration) -> bool {
    let now = now();
    let start = now - unload_window.as_secs();
    let c = invocation_counter.count_in(start, now);
    c == 0
}

pub fn update_invocation_counter(invocation_counter: &InvocationCounter) {
    invocation_counter.register(now());
}
