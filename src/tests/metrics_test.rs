use serde_json::json;
use metrics_exporter_prometheus::PrometheusBuilder;

use crate::tests::utils::{init_log, TestContext};
use crate::types::{DocumentList, UpdateDocumentRequest, UpdateStrategy};

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics() {
    init_log();

    // Install PrometheusBuilder to capture metrics
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install recorder");

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // First insertion: 3 documents
    let documents1: DocumentList = json!([
        {"id": "1", "title": "First Document", "content": "Content 1"},
        {"id": "2", "title": "Second Document", "content": "Content 2"},
        {"id": "3", "title": "Third Document", "content": "Content 3"}
    ]).try_into().unwrap();
    index_client.insert_documents(documents1).await.unwrap();

    // Second insertion: 5 documents
    let documents2: DocumentList = json!([
        {"id": "4", "title": "Fourth Document", "content": "Content 4"},
        {"id": "5", "title": "Fifth Document", "content": "Content 5"},
        {"id": "6", "title": "Sixth Document", "content": "Content 6"},
        {"id": "7", "title": "Seventh Document", "content": "Content 7"},
        {"id": "8", "title": "Eighth Document", "content": "Content 8"}
    ]).try_into().unwrap();
    
    index_client.insert_documents(documents2).await.unwrap();

    index_client.update_documents(UpdateDocumentRequest {
        strategy: UpdateStrategy::Merge,
        documents: json!([
            {"id": "1", "number": 3},
        ]).try_into().unwrap(),
    }).await.unwrap();

    // Delete some documents: delete documents with ids "2" and "3"
    let documents_to_delete = vec!["2".to_string(), "3".to_string()];
    index_client.delete_documents(documents_to_delete).await.unwrap();

    // Get metrics output as string
    let metrics = prometheus_handle.render();

    let lines: Vec<_> = metrics.lines().collect();

    // Insert docs
    lines.iter()
        .find(|l| l.contains("oramacore_insert_documents_total{") && l.contains(collection_client.collection_id.as_str()))
        .expect("oramacore_insert_documents_total for this collection has to be present");
    let oramacore_insert_documents_duration_seconds_count = lines.iter()
        .filter(|l| l.contains("oramacore_insert_documents_total{") && l.contains(collection_client.collection_id.as_str()))
        .count();
    assert!(oramacore_insert_documents_duration_seconds_count > 0);

    // update docs
    lines.iter()
        .find(|l| l.contains("oramacore_update_documents_total{") && l.contains(collection_client.collection_id.as_str()))
        .expect("oramacore_update_documents_total for this collection has to be present");

    let oramacore_update_documents_duration_seconds_count = lines.iter()
        .filter(|l| l.contains("oramacore_update_documents_duration_seconds{") && l.contains(collection_client.collection_id.as_str()))
        .count();
    assert!(oramacore_update_documents_duration_seconds_count > 0);

    // delete docs
    lines.iter()
        .find(|l| l.contains("oramacore_delete_documents_total{") && l.contains(collection_client.collection_id.as_str()))
        .expect("oramacore_delete_documents_total for this collection has to be present");

    let oramacore_delete_documents_duration_seconds_count = lines.iter()
        .filter(|l| l.contains("oramacore_delete_documents_duration_seconds{") && l.contains(collection_client.collection_id.as_str()))
        .count();
    assert!(oramacore_delete_documents_duration_seconds_count > 0);

    drop(test_context);
}