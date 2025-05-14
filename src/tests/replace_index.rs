use std::time::Duration;

use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, TestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_index_replacement() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    index_1_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Tommaso" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let index_2_client = collection_client
        .create_temp_index(index_1_client.index_id)
        .await
        .unwrap();
    index_2_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Michele" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);

    collection_client
        .replace_index(index_1_client.index_id, index_2_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_replacement_2() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    index_1_client
        .insert_documents(
            json!([json!({ "id": "1", "name": "Tommaso", "surname": "Allevi" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let index_2_client = collection_client
        .create_temp_index(index_1_client.index_id)
        .await
        .unwrap();
    index_2_client
        .insert_documents(
            json!([
                json!({ "id": "1", "name": "Tommaso", "surname": "Allevi" }),
                json!({ "id": "2", "name": "Michele", "surname": "Riva" }),
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .replace_index(index_1_client.index_id, index_2_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, format!("{}:1", index_1_client.index_id));
}
