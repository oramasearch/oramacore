use anyhow::bail;
use futures::FutureExt;
use serde_json::json;

use crate::tests::utils::{init_log, wait_for, TestContext};
use crate::types::{DocumentList, ReadApiKey};

#[tokio::test(flavor = "multi_thread")]
async fn test_regenerate_read_api_key() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert test documents
    let documents: DocumentList = json!([
        { "id": "1", "text": "The quick brown fox" },
        { "id": "2", "text": "jumps over the lazy dog" },
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    // Verify search works with original read key
    let result = collection_client
        .search(json!({ "term": "fox" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "Search should find one document with the original key"
    );

    // Save the original read API key
    let original_read_api_key = collection_client.read_api_key.clone();

    // Regenerate the read API key
    let new_read_api_key = test_context
        .writer
        .regenerate_read_api_key(
            collection_client.write_api_key,
            collection_client.collection_id,
        )
        .await
        .expect("Should regenerate read API key successfully");

    // Wait for the read side to pick up the new key
    let new_read_api_key_for_wait = ReadApiKey::from_api_key(new_read_api_key);
    let collection_id = collection_client.collection_id;
    wait_for(&test_context, |ctx| {
        let reader = ctx.reader.clone();
        let new_key = new_read_api_key_for_wait.clone();
        async move {
            // Try using the new key - if it works, the update has propagated
            let stats = reader
                .collection_stats(
                    &new_key,
                    collection_id,
                    crate::types::CollectionStatsRequest { with_keys: false },
                )
                .await;
            match stats {
                Ok(_) => Ok(()),
                Err(_) => bail!("New read API key not yet accepted by reader"),
            }
        }
        .boxed()
    })
    .await
    .expect("Reader should accept the new read API key");

    // Verify the old key no longer works
    let old_key_result = test_context
        .reader
        .collection_stats(
            &original_read_api_key,
            collection_id,
            crate::types::CollectionStatsRequest { with_keys: false },
        )
        .await;
    assert!(
        old_key_result.is_err(),
        "Old read API key should no longer work after regeneration"
    );

    // Verify search works with the new key
    let new_collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            new_read_api_key_for_wait.clone(),
        )
        .unwrap();

    let result = new_collection_client
        .search(json!({ "term": "fox" }).try_into().unwrap())
        .await
        .unwrap();
    assert_eq!(result.count, 1, "Search should work with the new key");

    // Commit and reload to verify persistence
    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;

    // After reload, the new key should still work
    let result = test_context
        .reader
        .collection_stats(
            &new_read_api_key_for_wait,
            collection_id,
            crate::types::CollectionStatsRequest { with_keys: false },
        )
        .await;
    assert!(
        result.is_ok(),
        "New read API key should persist across restarts"
    );

    // After reload, the old key should still not work
    let old_key_result = test_context
        .reader
        .collection_stats(
            &original_read_api_key,
            collection_id,
            crate::types::CollectionStatsRequest { with_keys: false },
        )
        .await;
    assert!(
        old_key_result.is_err(),
        "Old read API key should still not work after restart"
    );
}
