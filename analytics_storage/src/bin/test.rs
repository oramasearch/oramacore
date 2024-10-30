use analytics_storage::{AnalyticsStorage, AnalyticsStorageConfig, Granularity, OffloadTarget};
use anyhow::Context;

fn main() {
    let storage = AnalyticsStorage::try_new(AnalyticsStorageConfig {
        index_id: "my_index_id".to_string(),
        granularity: Some(Granularity::Hour),
        persistence_dir: None,
        offload_after: Some(100),
        offload_to: Some(OffloadTarget::Void),
        buffer_size: Some(100),
    })
    .unwrap();
}
