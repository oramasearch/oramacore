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
async fn test_delete_documents() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    config.reader_side.config.insert_batch_commit_size = 1_000_000;
    config.writer_side.config.insert_batch_commit_size = 1_000_000;

    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    let document_count = 10;
    insert_docs(
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

    write_side
        .delete_documents(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id,
            vec![(document_count - 1).to_string()],
        )
        .await?;
    sleep(Duration::from_millis(100)).await;
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
    assert_eq!(result.count, document_count - 1);

    write_side.commit().await.unwrap();
    read_side.commit().await.unwrap();

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
    assert_eq!(result.count, document_count - 1);

    write_side
        .delete_documents(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id,
            vec![(document_count - 2).to_string()],
        )
        .await?;
    sleep(Duration::from_millis(100)).await;

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
    assert_eq!(result.count, document_count - 2);

    let (write_side, read_side) = create(config.clone()).await?;
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
    assert_eq!(result.count, document_count - 1);

    write_side
        .delete_documents(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id,
            vec![(document_count - 2).to_string()],
        )
        .await?;
    sleep(Duration::from_millis(500)).await;

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
    assert_eq!(result.count, document_count - 2);

    write_side.commit().await.unwrap();
    read_side.commit().await.unwrap();

    let (_, read_side) = create(config.clone()).await?;
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
    assert_eq!(result.count, document_count - 2);

    // Deletion of non-existent document should not fail
    let output = write_side
        .delete_documents(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id,
            vec![(document_count - 2).to_string()],
        )
        .await;
    assert!(output.is_ok());

    Ok(())
}
