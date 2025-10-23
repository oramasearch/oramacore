use serde_json::json;

use crate::collection_manager::sides::read::IndexFieldStatsType;
use crate::tests::utils::create_oramacore_config;
use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::OramacoreConfig;

#[tokio::test(flavor = "multi_thread")]
async fn test_commit_after_operation_limit_reached() {
    init_log();
    let mut config: OramacoreConfig = create_oramacore_config();
    config.writer_side.config.insert_batch_commit_size = 5;
    config.reader_side.config.insert_batch_commit_size = 5;
    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!((0..5)
                .map(|i| {
                    json!({
                        "id": i.to_string(),
                        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    })
                })
                .collect::<Vec<_>>())
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let stats = collection_client.reader_stats().await.unwrap();

    let at_least_one_committed = stats.indexes_stats.iter().any(|index| {
        index.fields_stats.iter().any(|f| {
            matches!(
                f.stats,
                IndexFieldStatsType::CommittedBoolean(_)
                    | IndexFieldStatsType::CommittedNumber(_)
                    | IndexFieldStatsType::CommittedStringFilter(_)
                    | IndexFieldStatsType::CommittedString(_)
                    | IndexFieldStatsType::CommittedVector(_)
            )
        })
    });
    assert!(at_least_one_committed, "No committed fields found");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_empty_index_reload() {
    init_log();
    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    collection_client.create_index().await.unwrap();

    test_context.commit_all().await.unwrap();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let stats = collection_client.reader_stats().await.unwrap();

    let total_fields: usize = stats
        .indexes_stats
        .iter()
        .map(|index| index.fields_stats.len())
        .sum();

    // Embedding fields (committed and uncommitted)
    assert_eq!(total_fields, 2, "Expected no fields in the index");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_empty_collection_reload() {
    init_log();
    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    test_context.commit_all().await.unwrap();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let stats = collection_client
        .reader_stats()
        .await
        .expect("Collection must exists");

    assert_eq!(stats.indexes_stats.len(), 0, "Expected no indexes");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_continue_commit() {
    init_log();
    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    for i in 0..4 {
        index_client
            .insert_documents(
                vec![json!({
                    "id": format!("{}", i),
                    "text": "text",
                })]
                .try_into()
                .unwrap(),
            )
            .await
            .unwrap();
        test_context.commit_all().await.unwrap();
    }

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let search_result = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .expect("Collection must exists");

    assert_eq!(search_result.count, 4);

    drop(test_context);
}
