use anyhow::Result;
use serde_json::json;

use crate::{tests::utils::{create_oramacore_config, TestContext}, types::{ApiKey, CollectionId}, OramacoreConfig};

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_doc_migration() -> Result<()> {
    let mut config: OramacoreConfig = create_oramacore_config();
    config.writer_side.config.data_dir = std::env::current_dir().unwrap()
        .join("dump-test")
        .join("migration_docs")
        .join("writer");

    println!("Config: {:#?}", config.writer_side.config.data_dir);

    let test_context = TestContext::new_with_config(config.clone()).await;

    let colls = test_context.get_writer_collections().await;

    println!("Collections: {:#?}", colls);
    assert_eq!(colls.len(), 1);

    


    /*
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let document_count = 10;
    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let docs = test_context
        .writer
        .list_document(
            collection_client.write_api_key,
            collection_client.collection_id,
        )
        .await?;
    assert_eq!(docs.len(), document_count);
    */

    Ok(())
}
