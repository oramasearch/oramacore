use serde_json::json;

use crate::tests::utils::{init_log, TestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_documents_success() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert test documents
    index_client
        .insert_documents(
            json!([
                { "id": "doc1", "title": "First", "value": 100 },
                { "id": "doc2", "title": "Second", "value": 200 },
                { "id": "doc3", "title": "Third", "value": 300 },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Batch retrieve
    let result = collection_client
        .batch_get_documents(vec!["doc1".to_string(), "doc3".to_string()])
        .await
        .unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains_key("doc1"));
    assert!(result.contains_key("doc3"));
    assert_eq!(result["doc1"].inner.get("title").unwrap(), "First");
    assert_eq!(result["doc3"].inner.get("value").unwrap(), &json!(300));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_empty_request() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let result = collection_client.batch_get_documents(vec![]).await.unwrap();

    assert_eq!(result.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_non_existent_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                { "id": "doc1", "title": "First" },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .batch_get_documents(vec![
            "doc1".to_string(),
            "missing1".to_string(),
            "missing2".to_string(),
        ])
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    assert!(result.contains_key("doc1"));
    assert!(!result.contains_key("missing1"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_all_missing() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    let result = collection_client
        .batch_get_documents(vec!["missing1".to_string(), "missing2".to_string()])
        .await
        .unwrap();

    assert_eq!(result.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_with_deleted_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                { "id": "doc1", "title": "First" },
                { "id": "doc2", "title": "Second" },
                { "id": "doc3", "title": "Third" },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Delete doc2
    index_client
        .delete_documents(vec!["doc2".to_string()])
        .await
        .unwrap();

    let result = collection_client
        .batch_get_documents(vec![
            "doc1".to_string(),
            "doc2".to_string(),
            "doc3".to_string(),
        ])
        .await
        .unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains_key("doc1"));
    assert!(!result.contains_key("doc2"));
    assert!(result.contains_key("doc3"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_duplicate_ids() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                { "id": "doc1", "title": "First" },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .batch_get_documents(vec![
            "doc1".to_string(),
            "doc1".to_string(),
            "doc1".to_string(),
        ])
        .await
        .unwrap();

    // Should return only one entry
    assert_eq!(result.len(), 1);
    assert!(result.contains_key("doc1"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                { "id": "doc1", "title": "First" },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Force commit
    test_context.commit_all().await.unwrap();

    // Should still work after commit
    let result = collection_client
        .batch_get_documents(vec!["doc1".to_string()])
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    assert!(result.contains_key("doc1"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_exactly_1000() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert 100 documents
    let docs: Vec<_> = (0..100)
        .map(|i| json!({ "id": format!("doc{}", i), "value": i }))
        .collect();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Request exactly 1000 IDs (most won't exist)
    let ids: Vec<String> = (0..1000).map(|i| format!("doc{i}")).collect();

    let result = collection_client.batch_get_documents(ids).await.unwrap();

    assert_eq!(result.len(), 100);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_get_multiple_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert many documents
    let docs: Vec<_> = (0..50)
        .map(|i| json!({ "id": format!("doc{}", i), "title": format!("Title {}", i), "value": i }))
        .collect();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Retrieve a subset
    let ids: Vec<String> = (0..50).step_by(5).map(|i| format!("doc{i}")).collect();

    let result = collection_client
        .batch_get_documents(ids.clone())
        .await
        .unwrap();

    assert_eq!(result.len(), 10); // Every 5th document from 0 to 49
    for id in ids {
        assert!(result.contains_key(&id));
    }
}
