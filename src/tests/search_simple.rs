use std::time::Duration;

use anyhow::Result;
use rallo::RalloAllocator;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    collection_manager::dto::ApiKey,
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::CollectionId,
};

const MAX_FRAME_LENGTH: usize = 128;
const MAX_LOG_COUNT: usize = 1_024 * 10;
#[global_allocator]
static ALLOCATOR: RalloAllocator<MAX_FRAME_LENGTH, MAX_LOG_COUNT> = RalloAllocator::new();

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_search_simpleeeee() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    config.reader_side.config.insert_batch_commit_size = 1_000_000;
    config.writer_side.config.insert_batch_commit_size = 1_000_000;

    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    let document_count = 10;
    let result = insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        (0..document_count).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
            })
        }),
    )
    .await?;
    assert_eq!(result.inserted, document_count);
    assert_eq!(result.failed, 0);
    assert_eq!(result.replaced, 0);

    write_side.commit().await?;
    read_side.commit().await?;

    sleep(Duration::from_millis(200)).await;

    for _ in 0..10 {
        read_side
            .search(
                ApiKey(Secret::new("my-read-api-key".to_string())),
                collection_id,
                json!({
                    "term": "text",
                })
                .try_into()?,
            )
            .await?;
    }

    let params = json!({
        "term": "text",
    })
    .try_into()?;
    ALLOCATOR.start_track();
    read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            params,
        )
        .await?;
    ALLOCATOR.stop_track();

    let stats = unsafe { ALLOCATOR.calculate_stats() };
    let tree = stats.into_tree().unwrap();

    tree.print_flamegraph("flamegraph.html");

    Ok(())
}
