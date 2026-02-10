use oramacore_lib::hook_storage::HookType;

use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use crate::types::{CollectionStatsRequest, DocumentList};
use crate::HooksConfig;
use anyhow::Context;
use futures::FutureExt;
use serde_json::json;
use std::time::Duration;

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

    let read_api_key = collection_client.read_api_key.clone();
    let collection_id = collection_client.collection_id;
    let stats = wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        let read_api_key = read_api_key.clone();
        async move {
            let stats = reader
                .collection_stats(
                    &read_api_key,
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

#[test]
fn test_hooks_config_defaults() {
    // Test that HooksConfig::default() provides sensible defaults
    let default_config = HooksConfig::default();

    assert_eq!(default_config.allowed_domains.len(), 0);
    assert_eq!(default_config.denied_domains.len(), 0);
    assert_eq!(
        default_config.evaluation_timeout,
        Duration::from_millis(200)
    );
    assert_eq!(
        default_config.execution_timeout,
        Duration::from_millis(1000)
    );
}

#[test]
fn test_hooks_config_deserialization_with_values() {
    // Test that custom values are properly deserialized
    let json = serde_json::json!({
        "denied_domains": ["127.*", "localhost"],
        "evaluation_timeout": "500ms",
        "execution_timeout": "2s"
    });

    let config: HooksConfig = serde_json::from_value(json).expect("Failed to deserialize config");

    assert_eq!(config.denied_domains, vec!["127.*", "localhost"]);
    assert_eq!(config.allowed_domains.len(), 0);
    assert_eq!(config.evaluation_timeout, Duration::from_millis(500));
    assert_eq!(config.execution_timeout, Duration::from_secs(2));
}

#[test]
fn test_hooks_config_deserialization_empty() {
    // Test that empty config uses all defaults
    let json = serde_json::json!({});

    let config: HooksConfig = serde_json::from_value(json).expect("Failed to deserialize config");

    assert_eq!(config.allowed_domains.len(), 0);
    assert_eq!(config.denied_domains.len(), 0);
    assert_eq!(config.evaluation_timeout, Duration::from_millis(200));
    assert_eq!(config.execution_timeout, Duration::from_millis(1000));
}

#[test]
fn test_hooks_config_validation_mutual_exclusion() {
    // Test that both allowed_domains and denied_domains cannot be set together
    let config = HooksConfig {
        allowed_domains: vec!["example.com".to_string()],
        denied_domains: vec!["bad.com".to_string()],
        evaluation_timeout: Duration::from_millis(200),
        execution_timeout: Duration::from_millis(1000),
    };

    assert!(config.validate().is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hooks_isolated_between_collections() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_a = test_context.create_collection().await.unwrap();
    collection_a
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    return documents.map(doc => {
        doc.title = doc.title.toUpperCase();
        doc.collection = "A";
        return doc;
    });
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let collection_b = test_context.create_collection().await.unwrap();
    collection_b
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    return documents.map(doc => {
        doc.title = doc.title.toLowerCase();
        doc.collection = "B";
        return doc;
    });
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let index_a = collection_a.create_index().await.unwrap();
    let index_b = collection_b.create_index().await.unwrap();

    let documents: DocumentList = json!([{"id": "1", "title": "Hello World"}])
        .try_into()
        .unwrap();

    index_a.insert_documents(documents.clone()).await.unwrap();
    index_b.insert_documents(documents).await.unwrap();

    // Verify collection A has uppercase title
    let results_a = collection_a
        .search(json!({"term": "HELLO"}).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(results_a.count, 1);
    let doc_a = results_a.hits[0].document.as_ref().unwrap();
    assert_eq!(doc_a.get("title").unwrap(), "HELLO WORLD");
    assert_eq!(doc_a.get("collection").unwrap(), "A");

    // Verify collection B has lowercase title
    let results_b = collection_b
        .search(json!({"term": "hello"}).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(results_b.count, 1);
    let doc_b = results_b.hits[0].document.as_ref().unwrap();
    assert_eq!(doc_b.get("title").unwrap(), "hello world");
    assert_eq!(doc_b.get("collection").unwrap(), "B");
}
