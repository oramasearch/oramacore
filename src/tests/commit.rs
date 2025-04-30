use serde_json::json;

use crate::collection_manager::sides::IndexFieldStatsType;
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
