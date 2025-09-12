use std::time::Duration;

use serde_json::json;
use tokio::time::sleep;

use crate::collection_manager::sides::read::SearchRequest;
use crate::tests::utils::create_oramacore_config;
use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::DocumentList;
use crate::types::WriteApiKey;
use crate::OramacoreConfig;

#[tokio::test(flavor = "multi_thread")]
async fn test_master_api_key_to_insert_document() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let output = index_client
        .writer
        .insert_documents(
            WriteApiKey::ApiKey(test_context.master_api_key),
            index_client.collection_id,
            index_client.index_id,
            DocumentList(vec![]),
        )
        .await
        .unwrap();

    assert_eq!(output.failed, 0);
    assert_eq!(output.inserted, 0);
    assert_eq!(output.replaced, 0);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_api_key_to_search() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .writer
        .insert_documents(
            WriteApiKey::ApiKey(test_context.master_api_key),
            index_client.collection_id,
            index_client.index_id,
            json!([
                {"name": "Tommaso"}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    sleep(Duration::from_millis(100)).await;

    let WriteApiKey::ApiKey(write_api_key) = collection_client.write_api_key else {
        panic!("Write Api key is expected to be an ApiKey");
    };
    let output = collection_client
        .reader
        .search(
            write_api_key,
            collection_client.collection_id,
            SearchRequest {
                search_params: json!({
                    "term": "T",
                })
                .try_into()
                .unwrap(),
                analytics_metadata: None,
                interaction_id: None,
                search_analytics_event_origin: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_master_api_key_to_search() {
    init_log();

    let mut config: OramacoreConfig = create_oramacore_config();
    config.reader_side.master_api_key = Some(config.writer_side.master_api_key);

    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .writer
        .insert_documents(
            WriteApiKey::ApiKey(test_context.master_api_key),
            index_client.collection_id,
            index_client.index_id,
            json!([
                {"name": "Tommaso"}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    sleep(Duration::from_millis(100)).await;

    let output = collection_client
        .reader
        .search(
            test_context.master_api_key,
            collection_client.collection_id,
            SearchRequest {
                search_params: json!({
                    "term": "T",
                })
                .try_into()
                .unwrap(),
                analytics_metadata: None,
                interaction_id: None,
                search_analytics_event_origin: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    drop(test_context);
}
