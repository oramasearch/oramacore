use std::time::Duration;

use crate::collection_manager::sides::write::TempIndexCleanupConfig;
use crate::tests::utils::create_oramacore_config;
use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use futures::FutureExt;
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
async fn test_temp_index_cleanup_enabled() {
    init_log();

    let mut config = create_oramacore_config();
    config.writer_side.config.temp_index_cleanup = TempIndexCleanupConfig {
        cleanup_interval: Duration::from_millis(200),
        max_age: Duration::from_millis(1_000),
    };

    let test_context = TestContext::new_with_config(config).await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let copy_from = index_client.index_id;

    // Create a temp index
    let index_client = collection_client
        .create_temp_index(copy_from)
        .await
        .unwrap();
    index_client
        .writer
        .insert_documents(
            index_client.write_api_key,
            index_client.collection_id,
            index_client.index_id,
            json!([
                {
                    "id": 1,
                    "title": "Document 1",
                    "content": "This is the content of document 1"
                },
                {
                    "id": 2,
                    "title": "Document 2",
                    "content": "This is the content of document 2"
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    wait_for(
        &collection_client,
        |collection_client: &crate::tests::utils::TestCollectionClient| {
            async move {
                let stats = collection_client.reader_stats().await.unwrap();
                if stats.indexes_stats.len() == 2 {
                    // index + temp index
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("No temp index found"))
                }
            }
            .boxed()
        },
    )
    .await
    .unwrap();

    // Waiting for temp index to be cleaned up
    wait_for(&collection_client, |collection_client| {
        async move {
            let stats = collection_client.reader_stats().await.unwrap();
            if stats.indexes_stats.len() == 1 {
                // only the index: the temp index should gone
                Ok(())
            } else {
                Err(anyhow::anyhow!("Temp index found but should not be"))
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    drop(test_context);
}
