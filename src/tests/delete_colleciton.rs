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

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_collection() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "name": "John Doe",
            }),
            json!({
                "id": "2",
                "name": "Jane Doe",
            }),
        ],
    )
    .await?;

    write_side.commit().await?;
    read_side.commit().await?;

    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
        )
        .await?;
    assert_eq!(stats.document_count, 2);

    write_side
        .delete_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            collection_id.clone(),
        )
        .await?;

    let stats = write_side
        .get_collection_dto(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            collection_id.clone(),
        )
        .await;
    assert!(matches!(stats, Ok(None)));

    sleep(Duration::from_millis(200)).await;

    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
        )
        .await;

    assert!(stats.is_err());

    Ok(())
}
