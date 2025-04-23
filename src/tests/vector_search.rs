use assert_approx_eq::assert_approx_eq;
use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_search() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                json!({
                    "id": "1",
                    "text": "The cat is sleeping on the table.",
                }),
                json!({
                    "id": "2",
                    "text": "A cat rests peacefully on the sofa.",
                }),
                json!({
                    "id": "3",
                    "text": "The dog is barking loudly in the yard.",
                }),
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "A cat sleeps",
                "mode": "vector"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert!(output.hits[0].score > 0.);

    let output = collection_client
        .search(
            json!({
                "term": "A cat sleeps",
                "mode": "vector",
                "similarity": 0.0001
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 3);
    assert_eq!(output.hits.len(), 3);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[2].id,
        format!("{}:{}", index_client.index_id, "3")
    );
    assert!(output.hits[0].score > 0.);
    assert!(output.hits[1].score > 0.);
    assert!(output.hits[2].score > 0.);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_search_should_work_after_commit() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                json!({
                    "id": "1",
                    "text": "The cat is sleeping on the table.",
                }),
                json!({
                    "id": "2",
                    "text": "A cat rests peacefully on the sofa.",
                }),
                json!({
                    "id": "3",
                    "text": "The dog is barking loudly in the yard.",
                }),
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output1 = collection_client
        .search(
            json!({
                "term": "A cat sleeps",
                "mode": "vector"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output1.count, 1);
    assert_eq!(output1.hits.len(), 1);
    assert_eq!(
        output1.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert!(output1.hits[0].score > 0.);

    test_context.commit_all().await.unwrap();

    let output2 = collection_client
        .search(
            json!({
                "term": "A cat sleeps",
                "mode": "vector",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output2.count, 1);
    assert_eq!(output2.hits.len(), 1);
    assert_eq!(
        output2.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert!(output2.hits[0].score > 0.);

    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let output3 = collection_client
        .search(
            json!({
                "term": "A cat sleeps",
                "mode": "vector",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output3.count, 1);
    assert_eq!(output3.hits.len(), 1);
    assert_eq!(
        output3.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert!(output3.hits[0].score > 0.);

    assert_approx_eq!(output1.hits[0].score, output2.hits[0].score);
    assert_approx_eq!(output2.hits[0].score, output3.hits[0].score);

    drop(test_context);
}
