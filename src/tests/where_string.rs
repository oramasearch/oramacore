use anyhow::Result;
use redact::Secret;
use serde_json::json;

use crate::{
    collection_manager::dto::ApiKey,
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_where_string() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "section": "my-section-1",
                "name": "John Doe",
            }),
            json!({
                "id": "2",
                "section": "my-section-2",
                "name": "Jane Doe",
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "section": "my-section-1"
                }
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    Ok(())
}
