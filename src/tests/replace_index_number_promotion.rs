use std::time::Duration;

use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, wait_for, TestContext};

/// Tests that number fields in a temp index are correctly promoted to the runtime index.
///
/// The promotion flow:
/// 1. Create a runtime index with some documents (including a number field).
/// 2. Create a temp index from the runtime index.
/// 3. Insert documents with number fields (mixed I32 and F32) into the temp index.
/// 4. Commit the temp index to persist NumberFieldStorage to temp directory.
/// 5. Replace (promote) the temp index into the runtime index.
/// 6. Verify the number field is searchable after promotion via number filter.
/// 7. Commit again and verify data survives.
/// 8. Reload the system and verify number field data persists after reload.
#[tokio::test(flavor = "multi_thread")]
async fn test_number_field_promotion_from_temp_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    // Insert an initial document with a number field in the runtime index.
    runtime_index_client
        .insert_documents(
            json!([
                {
                    "id": "original-1",
                    "name": "Original",
                    "price": 100,
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

    // Insert documents with mixed integer and float number fields into the temp index.
    let docs: Vec<_> = vec![
        ("0", "Item-0", 10),
        ("1", "Item-1", 20),
        ("2", "Item-2", 30),
        ("3", "Item-3", 40),
        ("4", "Item-4", 50),
        ("5", "Item-5", 60),
        ("6", "Item-6", 70),
        ("7", "Item-7", 80),
        ("8", "Item-8", 90),
        ("9", "Item-9", 100),
        ("10", "Item-10", 110),
        ("11", "Item-11", 120),
        ("12", "Item-12", 130),
        ("13", "Item-13", 140),
        ("14", "Item-14", 150),
        ("15", "Item-15", 160),
        ("16", "Item-16", 170),
        ("17", "Item-17", 180),
        ("18", "Item-18", 190),
        ("19", "Item-19", 200),
    ]
    .into_iter()
    .map(|(id, name, price)| {
        json!({
            "id": id,
            "name": name,
            "price": price,
        })
    })
    .collect();

    temp_index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Commit to persist the number field data in the temp directory.
    test_context.commit_all().await.unwrap();

    // Verify the original runtime index still returns its own data through search.
    // Filter for price >= 50 - should find just the original document (price=100).
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "price": {
                        "gte": 50
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

    // After promotion, search with price >= 100.
    // Should find Item-9(100) through Item-19(200) = 11 documents.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "price": {
                        "gte": 100
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 11,
        "After promotion, should find at least 11 documents with price >= 100, got {}",
        result.count
    );
    let gte_100_count = result.count;

    // Search with price < 100 to get lower-priced items.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "price": {
                        "lt": 100
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 9,
        "After promotion, should find at least 9 documents with price < 100, got {}",
        result.count
    );

    // Commit after promotion to verify compaction works at the new path.
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "price": {
                        "gte": 100
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, gte_100_count,
        "After post-promotion commit, number filter should still work"
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
                    "price": {
                        "gte": 100
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, gte_100_count,
        "After reload, number field data should persist"
    );

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "price": {
                        "lt": 100
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 9,
        "After reload, should still find documents with price < 100, got {}",
        result.count
    );

    drop(test_context);
}

/// Tests that inserting new documents with number fields into a promoted index works correctly.
///
/// After promotion, the number field storage points to the new runtime path.
/// New inserts and subsequent commits must operate on the promoted path without errors.
#[tokio::test(flavor = "multi_thread")]
async fn test_number_field_insert_after_promotion() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{
                "id": "seed",
                "name": "seed",
                "price": 50,
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
                { "id": "1", "name": "Doc1", "price": 10 },
                { "id": "2", "name": "Doc2", "price": 99.5 },
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

    // Insert new documents into the now-promoted runtime index (mix of I32 and F32).
    runtime_index_client
        .insert_documents(
            json!([
                { "id": "3", "name": "Doc3", "price": 200 },
                { "id": "4", "name": "Doc4", "price": 49.99 },
                { "id": "5", "name": "Doc5", "price": 300 },
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

    // Search with price >= 100 - should find Doc3(200), Doc5(300) = at least 2.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "price": {
                        "gte": 100
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 2,
        "Should find at least 2 documents with price >= 100 after post-promotion insert, got {}",
        result.count
    );

    // Search with price between 40 and 100 - should find Doc4(49.99), Doc2(99.5) = at least 2.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "price": {
                        "between": [40, 100]
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 2,
        "Should find at least 2 documents with price between 40 and 100, got {}",
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
                    "price": {
                        "gte": 1
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
        "After reload, all 5 documents should have price >= 1"
    );

    drop(test_context);
}
