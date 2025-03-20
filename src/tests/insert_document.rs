use std::time::Duration;

use anyhow::Result;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    collection_manager::{dto::ApiKey, sides::FieldStatsType},
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

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count);

    let result = insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        (0..document_count).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "pippo ".repeat(i + 1),
            })
        }),
    )
    .await?;
    assert_eq!(result.inserted, 0);
    assert_eq!(result.failed, 0);
    assert_eq!(result.replaced, document_count);

    sleep(Duration::from_millis(200)).await;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "pippo",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_document_duplication() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id,
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![json!({
            "id": "1",
            "text": "B",
        })],
    )
    .await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![json!({
            "id": "1",
            "text": "C",
        })],
    )
    .await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "B",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 0);

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "C",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 1);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_document_chunk_long_text_for_embedding_calculation() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id,
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![json!({
            "id": "1",
            "text": "foo ".repeat(1_000),
        })],
    )
    .await?;

    let mut result = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
        )
        .await?;
    let FieldStatsType::Vector { uncommitted, .. } = result.fields_stats.remove(0).stats else {
        panic!("Expected vector field stats")
    };
    let uncommitted = uncommitted.unwrap();

    // Even if we insert only one document, we have two vectors because the text is chunked
    assert_eq!(uncommitted.len, 2);

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "foo ".repeat(256),
                "mode": "vector",
            })
            .try_into()?,
        )
        .await?;
    // In search result, the document is returned only once
    // because the id is the same
    // even it the document matches twice
    assert_eq!(result.count, 1);
    assert_eq!(result.hits[0].id, "1");

    Ok(())
}
