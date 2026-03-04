use std::time::Duration;

use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, wait_for, TestContext};

/// Tests that date fields in a temp index are correctly promoted to the runtime index.
///
/// The promotion flow:
/// 1. Create a runtime index with some documents (including a date field).
/// 2. Create a temp index from the runtime index.
/// 3. Insert documents with date fields into the temp index.
/// 4. Commit the temp index to persist DateFieldStorage to temp directory.
/// 5. Replace (promote) the temp index into the runtime index.
/// 6. Verify the date field is searchable after promotion via date filter.
/// 7. Commit again and verify data survives.
/// 8. Reload the system and verify date field data persists after reload.
#[tokio::test(flavor = "multi_thread")]
async fn test_date_field_promotion_from_temp_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    // Insert an initial document with a date field in the runtime index.
    runtime_index_client
        .insert_documents(
            json!([
                {
                    "id": "original-1",
                    "name": "Original",
                    "created_at": "2024-01-01T00:00:00Z",
                },
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

    // Insert documents with date fields into the temp index.
    // Dates spread across 2024 for temporal filtering tests.
    let docs: Vec<_> = vec![
        ("0", "January", "2024-01-15T00:00:00Z"),
        ("1", "February", "2024-02-15T00:00:00Z"),
        ("2", "March", "2024-03-15T00:00:00Z"),
        ("3", "April", "2024-04-15T00:00:00Z"),
        ("4", "May", "2024-05-15T00:00:00Z"),
        ("5", "June", "2024-06-15T00:00:00Z"),
        ("6", "July", "2024-07-15T00:00:00Z"),
        ("7", "August", "2024-08-15T00:00:00Z"),
        ("8", "September", "2024-09-15T00:00:00Z"),
        ("9", "October", "2024-10-15T00:00:00Z"),
        ("10", "November", "2024-11-15T00:00:00Z"),
        ("11", "December", "2024-12-15T00:00:00Z"),
        ("12", "Jan2025", "2025-01-15T00:00:00Z"),
        ("13", "Feb2025", "2025-02-15T00:00:00Z"),
        ("14", "Mar2025", "2025-03-15T00:00:00Z"),
        ("15", "Apr2025", "2025-04-15T00:00:00Z"),
        ("16", "May2025", "2025-05-15T00:00:00Z"),
        ("17", "Jun2025", "2025-06-15T00:00:00Z"),
        ("18", "Jul2025", "2025-07-15T00:00:00Z"),
        ("19", "Aug2025", "2025-08-15T00:00:00Z"),
    ]
    .into_iter()
    .map(|(id, name, date)| {
        json!({
            "id": id,
            "name": name,
            "created_at": date,
        })
    })
    .collect();

    temp_index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Commit to persist the date field data in the temp directory.
    test_context.commit_all().await.unwrap();

    // Verify the original runtime index still returns its own data through search.
    // Search with a date filter for >= 2024-01-01 - should find just the original document.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "created_at": {
                        "gte": "2024-01-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "Runtime index should still have only its own document"
    );

    // Promote: replace the runtime index with the temp index.
    collection_client
        .replace_index(runtime_index_client.index_id, temp_index_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    // After promotion, search with date >= 2024-06-01.
    // Should find June(5) through Aug2025(19) = 15 documents.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "created_at": {
                        "gte": "2024-06-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 14,
        "After promotion, should find at least 14 documents with date >= 2024-06-01, got {}",
        result.count
    );
    let gte_june_count = result.count;

    // Search with date < 2024-06-01 to get Q1-Q2 2024 documents.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "created_at": {
                        "lt": "2024-06-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 5,
        "After promotion, should find at least 5 documents with date < 2024-06-01, got {}",
        result.count
    );

    // Commit after promotion to verify compaction works at the new path.
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "created_at": {
                        "gte": "2024-06-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, gte_june_count,
        "After post-promotion commit, date filter should still work"
    );

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
                "where": {
                    "created_at": {
                        "gte": "2024-06-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, gte_june_count,
        "After reload, date field data should persist"
    );

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "created_at": {
                        "lt": "2024-06-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 5,
        "After reload, should still find documents with date < 2024-06-01, got {}",
        result.count
    );

    drop(test_context);
}

/// Tests that inserting new documents with date fields into a promoted index works correctly.
///
/// After promotion, the date field storage points to the new runtime path.
/// New inserts and subsequent commits must operate on the promoted path without errors.
#[tokio::test(flavor = "multi_thread")]
async fn test_date_field_insert_after_promotion() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{
                "id": "seed",
                "name": "seed",
                "created_at": "2024-01-01T00:00:00Z",
            }])
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
                { "id": "1", "name": "Doc1", "created_at": "2024-03-01T00:00:00Z" },
                { "id": "2", "name": "Doc2", "created_at": "2024-06-01T00:00:00Z" },
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
                { "id": "3", "name": "Doc3", "created_at": "2024-09-01T00:00:00Z" },
                { "id": "4", "name": "Doc4", "created_at": "2024-12-01T00:00:00Z" },
                { "id": "5", "name": "Doc5", "created_at": "2025-03-01T00:00:00Z" },
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

    // Search with date >= 2024-06-01 - should find Doc2(6/1), Doc3(9/1), Doc4(12/1), Doc5(3/25) = 4
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "created_at": {
                        "gte": "2024-06-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 4,
        "Should find at least 4 documents with date >= 2024-06-01 after post-promotion insert, got {}",
        result.count
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
                "where": {
                    "created_at": {
                        "gte": "2024-01-01T00:00:00Z"
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 5,
        "After reload, all 5 documents should have dates >= 2024-01-01"
    );

    drop(test_context);
}
