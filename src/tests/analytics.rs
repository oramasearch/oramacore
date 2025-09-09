use std::time::Duration;

use futures::StreamExt;
use serde_json::json;
use tokio::time::sleep;

use crate::collection_manager::sides::read::{OramaCoreAnalyticConfig, SearchAnalyticEventOrigin};
use crate::tests::utils::{create_oramacore_config, init_log, TestContext};
use crate::types::ApiKey;
use crate::OramacoreConfig;

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_analytics_search() {
    init_log();

    let mut config: OramacoreConfig = create_oramacore_config();
    config.reader_side.analytics = Some(OramaCoreAnalyticConfig {
        api_key: ApiKey::try_new("test_analytics_api_key").unwrap(),
    });

    let test_context = TestContext::new_with_config(config.clone()).await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs = (0..10)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    collection_client
        .reader
        .search(
            collection_client.read_api_key,
            collection_client.collection_id,
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
            Some(SearchAnalyticEventOrigin::Direct),
        )
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    let analytics_logs = collection_client.reader.get_analytics_logs().unwrap();
    let mut logs = analytics_logs
        .get_and_erase(config.reader_side.analytics.unwrap().api_key)
        .await
        .unwrap();

    let mut all = vec![];
    while let Some(Ok(bytes)) = logs.next().await {
        all.extend(bytes);
    }

    println!("Logs: {}", String::from_utf8_lossy(&all).as_ref());

    drop(test_context);
}
