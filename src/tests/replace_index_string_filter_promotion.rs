use std::time::Duration;

use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, wait_for, TestContext};

/// Tests that string_filter fields in a temp index are correctly promoted to the runtime index.
///
/// The promotion flow:
/// 1. Create a runtime index with some documents (including a string_filter field).
/// 2. Create a temp index from the runtime index.
/// 3. Insert documents with string_filter fields into the temp index.
/// 4. Commit the temp index to persist StringFilterStorage to temp directory.
/// 5. Replace (promote) the temp index into the runtime index.
/// 6. Verify the string_filter field is searchable after promotion via where filter.
/// 7. Commit again and verify data survives.
/// 8. Reload the system and verify string_filter field data persists after reload.
#[tokio::test(flavor = "multi_thread")]
async fn test_string_filter_field_promotion_from_temp_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    // Insert an initial document with a string_filter field in the runtime index.
    runtime_index_client
        .insert_documents(
            json!([
                {
                    "id": "original-1",
                    "name": "Original",
                    "color": "red",
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

    // Insert documents with string_filter fields into the temp index.
    let docs: Vec<_> = vec![
        ("0", "Apple", "red"),
        ("1", "Sky", "blue"),
        ("2", "Grass", "green"),
        ("3", "Fire", "red"),
        ("4", "Ocean", "blue"),
        ("5", "Leaf", "green"),
        ("6", "Cherry", "red"),
        ("7", "Sapphire", "blue"),
        ("8", "Emerald", "green"),
        ("9", "Ruby", "red"),
        ("10", "Cobalt", "blue"),
        ("11", "Jade", "green"),
        ("12", "Rose", "red"),
        ("13", "Azure", "blue"),
        ("14", "Mint", "green"),
        ("15", "Crimson", "red"),
        ("16", "Teal", "blue"),
        ("17", "Olive", "green"),
        ("18", "Sunset", "orange"),
        ("19", "Lemon", "yellow"),
    ]
    .into_iter()
    .map(|(id, name, color)| {
        json!({
            "id": id,
            "name": name,
            "color": color,
        })
    })
    .collect();

    temp_index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Commit to persist the string_filter field data in the temp directory.
    test_context.commit_all().await.unwrap();

    // Verify the original runtime index still returns its own data through search.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "red"
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

    // After promotion, search for "red" - should find 6 documents.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "red"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 6,
        "After promotion, should find 6 red documents, got {}",
        result.count
    );

    // Search for "blue" - should find 6 documents (ids 1,4,7,10,13,16).
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "blue"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 6,
        "After promotion, should find 6 blue documents, got {}",
        result.count
    );

    // Search for "green" - should find 6 documents (ids 2,5,8,11,14,17).
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "green"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 6,
        "After promotion, should find 6 green documents, got {}",
        result.count
    );

    // Commit after promotion to verify compaction works at the new path.
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "red"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 6,
        "After post-promotion commit, string_filter should still work"
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
                    "color": "red"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 6,
        "After reload, string_filter field data should persist"
    );

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "blue"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 6,
        "After reload, blue filter should persist, got {}",
        result.count
    );

    drop(test_context);
}

/// Tests that inserting new documents with string_filter fields into a promoted index works correctly.
///
/// After promotion, the string_filter field storage points to the new runtime path.
/// New inserts and subsequent commits must operate on the promoted path without errors.
#[tokio::test(flavor = "multi_thread")]
async fn test_string_filter_field_insert_after_promotion() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{
                "id": "seed",
                "name": "seed",
                "color": "red",
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
                { "id": "1", "name": "Apple", "color": "red" },
                { "id": "2", "name": "Sky", "color": "blue" },
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
                { "id": "3", "name": "Grass", "color": "green" },
                { "id": "4", "name": "Ocean", "color": "blue" },
                { "id": "5", "name": "Cherry", "color": "red" },
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

    // Search for "red" - should find 2 (Apple + Cherry).
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "red"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 2,
        "Should find 2 red documents after post-promotion insert, got {}",
        result.count
    );

    // Search for "blue" - should find 2 (Sky + Ocean).
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "blue"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 2,
        "Should find 2 blue documents after post-promotion insert, got {}",
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
        .search(json!({ "term": "" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 5,
        "After reload, all 5 documents should be present"
    );

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "color": "red"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 2,
        "After reload, should still find 2 red documents"
    );

    drop(test_context);
}
