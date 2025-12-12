use anyhow::Result;
use serde_json::json;
use tracing::info;

use crate::tests::utils::{init_log, TestContext};

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_list_documents() -> Result<()> {
    init_log();

    let test_context = TestContext::new().await;
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

    info!("Documents inserted, listing documents...");
    let docs = test_context
        .writer
        .list_document(
            collection_client.write_api_key,
            collection_client.collection_id,
        )
        .await?;
    info!("Done {:?}", docs);
    assert_eq!(docs.len(), document_count);

    Ok(())
}
