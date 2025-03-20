use anyhow::Result;
use dircpy::copy_dir;
use redact::Secret;
use serde_json::json;

use crate::{
    collection_manager::dto::ApiKey,
    tests::utils::{create, create_oramacore_config},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_ensure_back_compatibility() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    let cwd = std::env::current_dir().unwrap();
    config.writer_side.config.data_dir = cwd.join("./src/tests/dump/v1/write_side");
    config.reader_side.config.data_dir = cwd.join("./src/tests/dump/v1/read_side");
    let (_, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());

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

    assert!(output.is_ok());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_ensure_back_compatibility_and_upgrate() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    let cwd = std::env::current_dir().unwrap();

    copy_dir(
        cwd.join("./src/tests/dump"),
        cwd.join("./src/tests/dump-test/"),
    )
    .unwrap();

    config.writer_side.config.data_dir = cwd.join("./src/tests/dump-test/v1/write_side");
    config.reader_side.config.data_dir = cwd.join("./src/tests/dump-test/v1/read_side");
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "d",
                "where": {
                    "number": { "eq": 3 },
                    "numbers": { "eq": 3 },
                    "bool": false,
                    "bools": false,
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

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
                    "number": { "eq": 3 },
                    "numbers": { "eq": 3 },
                    "bool": false,
                    "bools": false,
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    Ok(())
}
