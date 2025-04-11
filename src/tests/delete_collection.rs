use std::time::Duration;

use anyhow::Result;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::{ApiKey, CollectionId},
};

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_delete_collection() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    insert_docs(
        write_side.clone(),
        ApiKey::try_from("my-write-api-key").unwrap(),
        collection_id,
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
        .collection_stats(ApiKey::try_from("my-read-api-key").unwrap(), collection_id)
        .await?;
    assert_eq!(stats.document_count, 2);

    write_side
        .delete_collection(
            ApiKey::try_from("my-master-api-key").unwrap(),
            collection_id,
        )
        .await?;

    sleep(Duration::from_millis(2_000)).await;

    sleep(Duration::from_millis(200)).await;
    let stats = write_side
        .get_collection_dto(
            ApiKey::try_from("my-master-api-key").unwrap(),
            collection_id,
        )
        .await;
    assert!(matches!(stats, Ok(None)));
    let stats = read_side
        .collection_stats(ApiKey::try_from("my-read-api-key").unwrap(), collection_id)
        .await;
    assert!(stats.is_err());

    write_side.commit().await?;
    read_side.commit().await?;

    let result = create_collection(write_side.clone(), collection_id).await;
    assert!(result.is_ok());

    sleep(Duration::from_millis(600)).await;

    let stats = read_side
        .collection_stats(ApiKey::try_from("my-read-api-key").unwrap(), collection_id)
        .await
        .unwrap();
    assert_eq!(stats.document_count, 0);

    let (write_side, read_side) = create(config.clone()).await?;
    let stats = write_side
        .get_collection_dto(
            ApiKey::try_from("my-master-api-key").unwrap(),
            collection_id,
        )
        .await;
    assert!(matches!(stats, Ok(None)));
    let stats = read_side
        .collection_stats(ApiKey::try_from("my-read-api-key").unwrap(), collection_id)
        .await;
    assert!(stats.is_err());

    Ok(())
}
