use std::time::Duration;

use anyhow::Result;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    tests::utils::{create, create_oramacore_config, insert_docs},
    types::{ApiKey, CollectionId},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_search_empty_document_fields() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());

    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id,
                "write_api_key": "write",
                "read_api_key": "read",
                "embeddings": {
                    "document_fields" : [],
                    "model": "BGESmall"
                }
            })
            .try_into()?,
        )
        .await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("write".to_string())),
        collection_id,
        vec![
            json!({
                "id": "1",
                "text": "The cat is sleeping on the table.",
            }),
            json!({
                "id": "2",
                "text": "A cat rests peacefully on the sofa.",
            }),
            json!({
                "id": "3",
                "text": "The dog is barking loudly in the yard.",
            }),
        ],
    )
    .await?;

    sleep(Duration::from_millis(500)).await;

    let output = read_side
        .search(
            ApiKey(Secret::new("read".to_string())),
            collection_id,
            json!({
                "term": "A cat sleeps",
                "mode": "vector"
            })
            .try_into()?,
        )
        .await?;

    assert_eq!(output.count, 1);

    Ok(())
}
