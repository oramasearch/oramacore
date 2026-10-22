use std::time::Duration;

use serde_json::json;
use tokio::time::timeout;

use crate::collection_manager::sides::read::SearchRequest;
use crate::tests::utils::{extrapolate_ids_from_result, init_log, TestContext};

/// A search waiting on document retrieval must not hold the collection's index read locks:
/// `commit` takes the index write lock and has to get it while the fetch is still pending.
#[tokio::test(flavor = "multi_thread")]
async fn test_slow_document_fetch_does_not_block_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    index_client
        .insert_documents(
            json!([
                { "id": "1", "text": "the quick brown fox" },
                { "id": "2", "text": "jumps over the lazy dog" },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    test_context.commit_all().await.unwrap();

    // Stand-in for slow storage: the next document fetch waits until released.
    let fetch_gate = test_context
        .reader
        .get_collection(
            collection_client.collection_id,
            &collection_client.read_api_key,
        )
        .await
        .unwrap()
        .get_document_storage()
        .fetch_gate();
    fetch_gate.arm();

    let reader = test_context.reader.clone();
    let read_api_key = collection_client.read_api_key.clone();
    let collection_id = collection_client.collection_id;
    let search = tokio::spawn(async move {
        reader
            .search(
                &read_api_key,
                collection_id,
                SearchRequest {
                    search_params: json!({ "term": "fox" }).try_into().unwrap(),
                    search_analytics_event_origin: None,
                    analytics_metadata: None,
                    interaction_id: None,
                },
            )
            .await
    });

    timeout(Duration::from_secs(10), fetch_gate.wait_entered())
        .await
        .expect("The search did not reach document retrieval");

    let commit = timeout(Duration::from_secs(1), test_context.reader.commit(true)).await;

    // Release the fetch before asserting, so a failure cannot leave the search holding locks.
    fetch_gate.release();
    let result = search.await.unwrap().unwrap();

    commit
        .expect("Commit was blocked by a search waiting on document retrieval")
        .unwrap();
    assert_eq!(result.count, 1);
    assert_eq!(extrapolate_ids_from_result(&result), vec!["1"]);
}
