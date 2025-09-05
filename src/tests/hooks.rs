use oramacore_lib::hook_storage::HookType;

use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use crate::types::CollectionStatsRequest;
use anyhow::Context;
use futures::FutureExt;

#[tokio::test(flavor = "multi_thread")]
async fn test_hooks() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::BeforeRetrieval,
            r#"
const beforeRetrieval = function () { }
export default { beforeRetrieval }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key;
    let collection_id = collection_client.collection_id;
    let stats = wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        async move {
            let stats = reader
                .collection_stats(
                    read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("")?;

            if stats.hooks.is_empty() {
                return Err(anyhow::anyhow!("hooks not arrived yet"));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    assert_eq!(stats.hooks, vec![HookType::BeforeRetrieval,]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hooks_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::BeforeRetrieval,
            r#"
const beforeRetrieval = function () { }
export default { beforeRetrieval }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key;
    let write_api_key = collection_client.write_api_key;
    let collection_id = collection_client.collection_id;
    let stats = wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        async move {
            let stats = reader
                .collection_stats(
                    read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("")?;

            if stats.hooks.is_empty() {
                return Err(anyhow::anyhow!("hooks not arrived yet"));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    assert_eq!(stats.hooks, vec![HookType::BeforeRetrieval,]);

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.hooks, vec![HookType::BeforeRetrieval,]);

    drop(test_context);
}
