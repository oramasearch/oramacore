use anyhow::Result;
use dircpy::copy_dir;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    collection_manager::dto::ApiKey,
    tests::utils::{create, create_oramacore_config},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_ensure_back_compatibility() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    config.writer_side.config.data_dir = "./src/tests/dump/v1/write_side".to_string().into();
    config.reader_side.config.data_dir = "./src/tests/dump/v1/read_side".to_string().into();
    let (_, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "d",
                "where": {
                    "number": { "gt": 1 },
                    "numbers": { "gt": 1 },
                    "bool": true,
                    "bools": true,
                }
            })
            .try_into()?,
        )
        .await;

    assert_eq!(output.is_ok(), true);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_ensure_back_compatibility_and_upgrate() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();

    copy_dir("./src/tests/dump", "./src/tests/dump-test/").unwrap();

    config.writer_side.config.data_dir = "./src/tests/dump-test/v1/write_side".to_string().into();
    config.reader_side.config.data_dir = "./src/tests/dump-test/v1/read_side".to_string().into();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "d",
                "where": {
                    "number": { "gt": 1 },
                    "numbers": { "gt": 1 },
                    "bool": true,
                    "bools": true,
                }
            })
            .try_into()?,
        )
        .await;
    assert_eq!(output.is_ok(), true);

    write_side.commit().await?;
    read_side.commit().await?;

    let (_, read_side) = create(config.clone()).await?;
    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "d",
                "where": {
                    "number": { "gt": 1 },
                    "numbers": { "gt": 1 },
                    "bool": true,
                    "bools": true,
                }
            })
            .try_into()?,
        )
        .await;
    assert_eq!(output.is_ok(), true);

    Ok(())
}
