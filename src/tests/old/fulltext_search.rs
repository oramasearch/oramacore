use anyhow::Result;
use serde_json::json;

use crate::{
    tests::utils::{create, create_oramacore_config, insert_docs},
    types::{ApiKey, CollectionId},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_fulltext_search_offset() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());

    write_side
        .create_collection(
            ApiKey::try_from("my-master-api-key").unwrap(),
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
        ApiKey::try_from("write").unwrap(),
        collection_id,
        vec![
            json!({
                "id": "1",
                "text": "cat",
            }),
            json!({
                "id": "2",
                "text": "cat cat",
            }),
            json!({
                "id": "3",
                "text": "cat cat cat",
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey::try_from("read").unwrap(),
            collection_id,
            json!({
                "term": "A cat sleeps",
                "limit": 1,
                "offset": 0,
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.hits.len(), 1);
    assert_eq!(
        output.hits[0].document.as_ref().unwrap().id,
        Some("3".to_string())
    );

    let output = read_side
        .search(
            ApiKey::try_from("read").unwrap(),
            collection_id,
            json!({
                "term": "A cat sleeps",
                "limit": 1,
                "offset": 1,
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.hits.len(), 1);
    assert_eq!(
        output.hits[0].document.as_ref().unwrap().id,
        Some("2".to_string())
    );

    let output = read_side
        .search(
            ApiKey::try_from("read").unwrap(),
            collection_id,
            json!({
                "term": "A cat sleeps",
                "limit": 1,
                "offset": 2,
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.hits.len(), 1);
    assert_eq!(
        output.hits[0].document.as_ref().unwrap().id,
        Some("1".to_string())
    );

    Ok(())
}
