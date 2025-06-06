use std::time::Duration;

use anyhow::bail;
use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::create_oramacore_config;
use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use crate::types::LanguageDTO;

#[tokio::test(flavor = "multi_thread")]
async fn test_change_language_without_commit() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "avvocata" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(json!({ "term": "avvocato" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    collection_client
        .rebuild_index(LanguageDTO::Italian)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    wait_for(&test_context, |test_context| {
        let collection_client = test_context
            .get_test_collection_client(
                collection_client.collection_id,
                collection_client.write_api_key,
                collection_client.read_api_key,
            )
            .unwrap();

        async move {
            let output = collection_client
                .search(json!({ "term": "avvocato" }).try_into().unwrap())
                .await
                .unwrap();
            if output.count == 1 {
                return Ok(());
            }
            bail!("Index not rebuilt yet")
        }
        .boxed()
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_change_language_with_commit() {
    init_log();
    let mut config = create_oramacore_config();
    config.writer_side.config.insert_batch_commit_size = 5;
    config.reader_side.config.insert_batch_commit_size = 5;

    let test_context = TestContext::new_with_config(config.clone()).await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                json!({ "id": "1", "text": "avvocata" }),
                json!({ "id": "2", "text": "avvocata" }),
                json!({ "id": "3", "text": "avvocata" }),
                json!({ "id": "4", "text": "avvocata" }),
                json!({ "id": "5", "text": "avvocata" }),
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    let output = collection_client
        .search(json!({ "term": "avvocato" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    let output = collection_client
        .search(json!({ "term": "avvocata" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(output.count, 5);

    collection_client
        .rebuild_index(LanguageDTO::Italian)
        .await
        .unwrap();

    wait_for(&test_context, |test_context| {
        let collection_client = test_context
            .get_test_collection_client(
                collection_client.collection_id,
                collection_client.write_api_key,
                collection_client.read_api_key,
            )
            .unwrap();

        async move {
            let output = collection_client
                .search(json!({ "term": "avvocato" }).try_into().unwrap())
                .await
                .unwrap();
            if output.count == 5 {
                return Ok(());
            }
            bail!("Index not rebuilt yet")
        }
        .boxed()
    })
    .await
    .unwrap();

    sleep(Duration::from_secs(1)).await;

    test_context.commit_all().await.unwrap();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let output = collection_client
        .search(json!({ "term": "avvocato" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(output.count, 5);
}
