use std::time::Duration;

use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, wait_for, TestContext};

/// Tests that bool fields in a temp index are correctly promoted to the runtime index.
///
/// The promotion flow:
/// 1. Create a runtime index with some documents (including a bool field).
/// 2. Create a temp index from the runtime index.
/// 3. Insert documents with bool fields into the temp index.
/// 4. Commit the temp index to persist BoolStorage to temp directory.
/// 5. Replace (promote) the temp index into the runtime index.
/// 6. Verify the bool field is searchable after promotion.
/// 7. Commit again and verify data survives.
/// 8. Reload the system and verify bool field data persists after reload.
#[tokio::test(flavor = "multi_thread")]
async fn test_bool_field_promotion_from_temp_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    // Insert an initial document with a bool field in the runtime index.
    runtime_index_client
        .insert_documents(
            json!([
                { "id": "original-1", "name": "Original", "active": true },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Create a temp index based on the runtime index.
    let temp_index_client = collection_client
        .create_temp_index(runtime_index_client.index_id)
        .await
        .unwrap();

    // Insert documents with bool field into the temp index.
    let docs: Vec<_> = (0..20)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "name": format!("doc {i}"),
                "active": i % 2 == 0,
            })
        })
        .collect();
    temp_index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Commit to persist the bool field data in the temp directory.
    test_context.commit_all().await.unwrap();

    // Verify the original runtime index still returns its own data through search.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "active": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1, "Runtime index should still have only its own document");

    // Promote: replace the runtime index with the temp index.
    collection_client
        .replace_index(runtime_index_client.index_id, temp_index_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    // After promotion, the bool filter should reflect the temp index data.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "active": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // Documents 0,2,4,6,8,10,12,14,16,18 have active=true → 10 docs
    assert_eq!(result.count, 10, "After promotion, 10 documents should have active=true");

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "active": false },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // Documents 1,3,5,7,9,11,13,15,17,19 have active=false → 10 docs
    assert_eq!(result.count, 10, "After promotion, 10 documents should have active=false");

    // Commit after promotion to verify compaction works at the new path.
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "active": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 10, "After post-promotion commit, bool filter should still work");

    // Reload the entire system and verify data persists.
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "active": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 10, "After reload, bool field data should persist");

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "active": false },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 10, "After reload, bool field false filter should persist");

    drop(test_context);
}

/// Tests promotion of a temp index with uncommitted bool field data.
///
/// This verifies that bool field data inserted after the last commit (still in BoolStorage's
/// pending ops) is correctly carried over during promotion and can be committed afterwards.
#[tokio::test(flavor = "multi_thread")]
async fn test_bool_field_promotion_with_uncommitted_data() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{ "id": "seed", "name": "seed", "enabled": false }])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let temp_index_client = collection_client
        .create_temp_index(runtime_index_client.index_id)
        .await
        .unwrap();

    // Insert first batch and commit.
    temp_index_client
        .insert_documents(
            json!([
                { "id": "1", "name": "first", "enabled": true },
                { "id": "2", "name": "second", "enabled": false },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Insert second batch WITHOUT committing - these remain as pending ops in BoolStorage.
    temp_index_client
        .insert_documents(
            json!([
                { "id": "3", "name": "third", "enabled": true },
                { "id": "4", "name": "fourth", "enabled": false },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Promote with uncommitted data still in the temp index.
    collection_client
        .replace_index(runtime_index_client.index_id, temp_index_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    // All 4 documents should be present (2 committed + 2 uncommitted from temp).
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "enabled": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2, "Should find 2 documents with enabled=true");

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "enabled": false },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2, "Should find 2 documents with enabled=false");

    // Commit after promotion to persist all data (including the previously uncommitted batch).
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "enabled": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2, "After commit, should still find 2 documents with enabled=true");

    // Verify data persists after reload.
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "enabled": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2, "After reload, should persist enabled=true documents");

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "enabled": false },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2, "After reload, should persist enabled=false documents");

    drop(test_context);
}

/// Tests that inserting new documents with bool fields into a promoted index works correctly.
///
/// After promotion, the bool field storage points to the new runtime path.
/// New inserts and subsequent commits must operate on the promoted path without errors.
#[tokio::test(flavor = "multi_thread")]
async fn test_bool_field_insert_after_promotion() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{ "id": "seed", "name": "seed", "available": true }])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let temp_index_client = collection_client
        .create_temp_index(runtime_index_client.index_id)
        .await
        .unwrap();

    temp_index_client
        .insert_documents(
            json!([
                { "id": "1", "name": "alpha", "available": true },
                { "id": "2", "name": "beta", "available": false },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Promote the temp index.
    collection_client
        .replace_index(runtime_index_client.index_id, temp_index_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    // Insert new documents into the now-promoted runtime index.
    runtime_index_client
        .insert_documents(
            json!([
                { "id": "3", "name": "gamma", "available": true },
                { "id": "4", "name": "delta", "available": false },
                { "id": "5", "name": "epsilon", "available": true },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Wait for eventual consistency after inserting into the promoted index.
    wait_for(&collection_client, |c| {
        async move {
            let result = c
                .search(json!({ "term": "" }).try_into().unwrap())
                .await
                .unwrap();
            if result.count != 5 {
                return Err(anyhow::anyhow!(
                    "Expected 5 documents, got {}",
                    result.count
                ));
            }
            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "available": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // From temp: id=1 (true). Post-promotion: id=3 (true), id=5 (true). Total: 3
    assert_eq!(
        result.count, 3,
        "Should find 3 documents with available=true after post-promotion insert"
    );

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "available": false },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // From temp: id=2 (false). Post-promotion: id=4 (false). Total: 2
    assert_eq!(
        result.count, 2,
        "Should find 2 documents with available=false after post-promotion insert"
    );

    // Commit to persist the post-promotion data.
    test_context.commit_all().await.unwrap();

    // Reload and verify everything is correct.
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "available": true },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 3, "After reload, available=true count should be 3");

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": { "available": false },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2, "After reload, available=false count should be 2");

    drop(test_context);
}
