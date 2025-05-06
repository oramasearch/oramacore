use anyhow::bail;
use futures::FutureExt;
use serde_json::json;

use crate::collection_manager::sides::IndexFieldStatsType;
use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_writer_graceful_shutdown() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .unchecked_insert_documents(
            json!([
                {
                    "id": "1",
                    "text": "Hi",
                },
                {
                    "id": "2",
                    "text": "Hello",
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.writer.stop().await.unwrap();

    wait_for(&test_context, |_| {
        async {
            let mut stats = collection_client.reader_stats().await.unwrap();

            let mut index = stats.indexes_stats.remove(0);
            // It is always the first field because we order them by field_id
            let embedding_field = index.fields_stats.remove(0);
            let IndexFieldStatsType::UncommittedVector(stats) = embedding_field.stats else {
                bail!("Expected UncommittedVector stats");
            };

            if stats.document_count == 2 {
                return Ok(());
            }
            bail!("Expected 2 documents, got {}", stats.document_count)
        }
        .boxed()
    })
    .await
    .unwrap();

    drop(test_context);
}
