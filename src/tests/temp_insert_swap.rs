use std::time::Duration;

use anyhow::Result;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    collection_manager::dto::{ApiKey, CreateCollectionFrom, SwapCollections},
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_temp_insert_swap() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    let write_api_key = ApiKey(Secret::new("my-write-api-key".to_string()));
    let read_api_key = ApiKey(Secret::new("my-read-api-key".to_string()));

    let temp_coll_id = write_side
        .create_collection_from(
            write_api_key.clone(),
            CreateCollectionFrom {
                from: collection_id,
                embeddings: None,
                language: None,
            },
        )
        .await?;

    insert_docs(
        write_side.clone(),
        write_api_key.clone(),
        temp_coll_id,
        vec![json!({
            "title": "avvocata",
        })],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            temp_coll_id,
            json!({
                "term": "avvocata",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    write_side
        .swap_collections(
            write_api_key.clone(),
            SwapCollections {
                from: temp_coll_id,
                to: collection_id,
            },
        )
        .await?;

    sleep(Duration::from_millis(100)).await;

    let output = read_side
        .search(
            read_api_key.clone(),
            collection_id,
            json!({
                "term": "avvocata",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    let stats = read_side
        .collection_stats(read_api_key.clone(), collection_id)
        .await?;
    assert_eq!(stats.document_count, 1);

    let output = write_side
        .list_collections(ApiKey(Secret::new("my-master-api-key".to_string())))
        .await?;

    assert!(!output.iter().any(|c| c.id == temp_coll_id));

    Ok(())
}
