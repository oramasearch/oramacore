use anyhow::Result;
use serde_json::json;

use crate::{tests::utils::{create_oramacore_config, TestContext}, types::{ApiKey, CollectionId}, OramacoreConfig};

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_doc_migration() -> Result<()> {
    let mut config: OramacoreConfig = create_oramacore_config();
    config.writer_side.config.data_dir = std::env::current_dir().unwrap()
        .join("src")
        .join("tests")
        .join("dump-test")
        .join("migration_docs")
        .join("writer");

    println!("Config: {:#?}", config.writer_side.config.data_dir);

    let test_context = TestContext::new_with_config(config.clone()).await;

    let colls = test_context.get_writer_collections().await;

    assert_eq!(colls.len(), 1);
    assert_eq!(colls[0].document_count, 5);
    assert_eq!(colls[0].indexes.len(), 1);
    assert_eq!(colls[0].indexes[0].document_count, 5);

    Ok(())
}
