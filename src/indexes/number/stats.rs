const BUCKET_COUNT: usize = 128;
const SUB_BUCKET_SIZE: usize = 4;
const GROUP_SHIFT: u32 = 10;

/// 1.5 days of usage data
#[derive(Debug)]
pub struct PageUsage(invocation_counter::Counter<BUCKET_COUNT, SUB_BUCKET_SIZE>);

impl PageUsage {
    pub fn new() -> Self {
        Self(invocation_counter::Counter::new(GROUP_SHIFT))
    }

    pub fn increment(&self, epoch: u64) {
        self.0.increment_by_one(epoch);
    }
}
