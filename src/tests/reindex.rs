use std::time::Duration;

use anyhow::Result;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;
use types::HookName;
use crate::{
    collection_manager::{
        dto::{ApiKey, LanguageDTO, ReindexConfig},
    },
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_reindex_change_language() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![json!({
            "title": "avvocata",
        })],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "avvocato",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 0);

    write_side
        .reindex(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id,
            ReindexConfig {
                description: None,
                embeddings: None,
                language: Some(LanguageDTO::Italian),
                reference: None,
            },
        )
        .await
        .unwrap();

    sleep(Duration::from_millis(300)).await;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "avvocato",
            })
            .try_into()?,
        )
        .await?;
    // We changed the language to Italian, so the search should return the document
    // because the "avvocata" shares the same root of "avvocato"
    assert_eq!(output.count, 1);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reindex_field_reorder() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![json!({
            "title1": "title1",
            "number1": 1,
            "bool1": true,
            "title2": "title2",
            "number2": 2,
            "bool2": false,
            "title3": "title3",
            "number3": 3,
            "bool3": true,
        })],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "title1",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    for _ in 0..50 {
        write_side
            .reindex(
                ApiKey(Secret::new("my-write-api-key".to_string())),
                collection_id,
                ReindexConfig {
                    description: None,
                    embeddings: None,
                    language: Some(LanguageDTO::Italian),
                    reference: None,
                },
            )
            .await
            .unwrap();
    }

    sleep(Duration::from_millis(300)).await;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "title1",
            })
            .try_into()?,
        )
        .await?;
    // We changed the language to Italian, so the search should return the document
    // because the "avvocata" shares the same root of "avvocato"
    assert_eq!(output.count, 1);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reindex_with_hooks() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    let code = r#"
function selectEmbeddingsProperties() {
    return "The pen is on the table.";
}
export default {
    selectEmbeddingsProperties
}
"#;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![json!({
            "title": "Today I want to listen only Max Pezzali.",
        })],
    )
    .await?;

    sleep(Duration::from_millis(500)).await;

    write_side
        .insert_javascript_hook(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id,
            HookName::SelectEmbeddingsProperties,
            code.to_string(),
        )
        .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 0);

    // Hook change the meaning of the text, so the exact match should not work
    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "mode": "vector",
                "term": "Today I want to listen only Max Pezzali.",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    write_side
        .reindex(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id,
            ReindexConfig {
                description: None,
                embeddings: None,
                language: None,
                reference: None,
            },
        )
        .await
        .unwrap();
    sleep(Duration::from_millis(500)).await;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    // Hook change the meaning of the text, so the exact match should not work
    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "mode": "vector",
                "term": "Today I want to listen only Max Pezzali.",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 0);

    Ok(())
}
