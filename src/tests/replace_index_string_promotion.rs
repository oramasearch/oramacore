use std::time::Duration;

use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, wait_for, TestContext};

/// Tests that string (fulltext) fields in a temp index are correctly promoted to the runtime index.
///
/// The promotion flow:
/// 1. Create a runtime index with some documents (including a text field).
/// 2. Create a temp index from the runtime index.
/// 3. Insert documents with text fields into the temp index.
/// 4. Commit the temp index to persist StringStorage to temp directory.
/// 5. Replace (promote) the temp index into the runtime index.
/// 6. Verify fulltext search works after promotion.
/// 7. Commit again and verify data survives.
/// 8. Reload the system and verify string field data persists after reload.
#[tokio::test(flavor = "multi_thread")]
async fn test_string_field_promotion_from_temp_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    // Insert an initial document with a text field in the runtime index.
    runtime_index_client
        .insert_documents(
            json!([
                { "id": "original-1", "description": "The quick brown fox jumps over the lazy dog" },
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

    // Insert documents with varied text into the temp index.
    let docs: Vec<_> = vec![
        json!({ "id": "1", "description": "Artificial intelligence and machine learning algorithms" }),
        json!({ "id": "2", "description": "Neural networks process deep learning models efficiently" }),
        json!({ "id": "3", "description": "Computer vision transforms image recognition tasks" }),
        json!({ "id": "4", "description": "Natural language processing enables text understanding" }),
        json!({ "id": "5", "description": "Reinforcement learning optimizes decision making policies" }),
        json!({ "id": "6", "description": "Data science combines statistics and programming skills" }),
        json!({ "id": "7", "description": "Cloud computing provides scalable infrastructure services" }),
        json!({ "id": "8", "description": "Cybersecurity protects digital assets from threats" }),
        json!({ "id": "9", "description": "Blockchain technology enables decentralized applications" }),
        json!({ "id": "10", "description": "Quantum computing explores new computational paradigms" }),
    ];
    temp_index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Commit to persist the string field data in the temp directory.
    test_context.commit_all().await.unwrap();

    // Verify the original runtime index still returns its own data through search.
    let result = collection_client
        .search(json!({ "term": "quick brown fox" }).try_into().unwrap())
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

    // After promotion, fulltext search should reflect the temp index data.
    let result = collection_client
        .search(json!({ "term": "artificial intelligence" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After promotion, should find 'artificial intelligence' document"
    );

    let result = collection_client
        .search(json!({ "term": "learning" }).try_into().unwrap())
        .await
        .unwrap();
    // "machine learning", "deep learning", "reinforcement learning" = 3 docs
    assert_eq!(
        result.count, 3,
        "After promotion, should find 3 documents containing 'learning'"
    );

    // Commit after promotion to verify compaction works at the new path.
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(json!({ "term": "neural networks" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After post-promotion commit, fulltext search should still work"
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
        .search(json!({ "term": "blockchain technology" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After reload, string field data should persist"
    );

    let result = collection_client
        .search(json!({ "term": "blockchain" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After reload, should find 1 document about blockchain"
    );

    drop(test_context);
}

/// Tests promotion of a temp index with uncommitted string field data.
///
/// This verifies that string field data inserted after the last commit (still in StringStorage's
/// pending ops) is correctly carried over during promotion and can be committed afterwards.
#[tokio::test(flavor = "multi_thread")]
async fn test_string_field_promotion_with_uncommitted_data() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{ "id": "seed", "description": "seed document for schema" }])
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
                { "id": "1", "description": "Database optimization techniques for PostgreSQL" },
                { "id": "2", "description": "Redis caching strategies for web applications" },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Insert second batch WITHOUT committing - these remain as pending ops in StringStorage.
    temp_index_client
        .insert_documents(
            json!([
                { "id": "3", "description": "MongoDB document storage architecture patterns" },
                { "id": "4", "description": "Elasticsearch distributed search engine internals" },
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
        .search(json!({ "term": "PostgreSQL" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "Should find committed document about PostgreSQL"
    );

    let result = collection_client
        .search(json!({ "term": "Elasticsearch" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "Should find uncommitted document about Elasticsearch"
    );

    // Commit after promotion to persist all data (including the previously uncommitted batch).
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(json!({ "term": "MongoDB" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After commit, should still find MongoDB document"
    );

    // Verify data persists after reload.
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(json!({ "term": "Redis" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After reload, should persist Redis document"
    );

    let result = collection_client
        .search(json!({ "term": "Elasticsearch" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After reload, should persist Elasticsearch document"
    );

    drop(test_context);
}

/// Tests that inserting new documents with string fields into a promoted index works correctly.
///
/// After promotion, the string field storage points to the new runtime path.
/// New inserts and subsequent commits must operate on the promoted path without errors.
#[tokio::test(flavor = "multi_thread")]
async fn test_string_field_insert_after_promotion() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{ "id": "seed", "description": "initial seed document" }])
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
                { "id": "1", "description": "Functional programming with Haskell monads" },
                { "id": "2", "description": "Object oriented design patterns in Java" },
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
                { "id": "3", "description": "Rust systems programming with memory safety" },
                { "id": "4", "description": "Python scripting for automation tasks" },
                { "id": "5", "description": "Go concurrency patterns with goroutines" },
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
        .search(json!({ "term": "programming" }).try_into().unwrap())
        .await
        .unwrap();
    // "functional programming" (id=1), "systems programming" (id=3) = 2 docs
    assert_eq!(
        result.count, 2,
        "Should find 2 documents containing 'programming' after post-promotion insert"
    );

    let result = collection_client
        .search(json!({ "term": "Rust memory safety" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "Should find the post-promotion Rust document"
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
        .search(json!({ "term": "Haskell monads" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After reload, should find original temp document"
    );

    let result = collection_client
        .search(json!({ "term": "Go concurrency" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "After reload, should find post-promotion document"
    );

    let result = collection_client
        .search(json!({ "term": "programming" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 2,
        "After reload, programming count should still be 2"
    );

    drop(test_context);
}
