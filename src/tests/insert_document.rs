use std::time::Duration;

use anyhow::Result;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    collection_manager::dto::ApiKey,
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_insert_duplicate_documents() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    config.reader_side.config.insert_batch_commit_size = 1_000_000;
    config.writer_side.config.insert_batch_commit_size = 1_000_000;

    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    let document_count = 10;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..document_count).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
            })
        }),
    )
    .await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count);

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..document_count).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "pippo ".repeat(i + 1),
            })
        }),
    )
    .await?;

    sleep(Duration::from_millis(200)).await;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "pippo",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count);

    Ok(())
}
