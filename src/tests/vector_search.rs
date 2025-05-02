use assert_approx_eq::assert_approx_eq;
use serde_json::json;
use tokio::time::sleep;

use crate::collection_manager::sides::hooks::HookName;
use crate::collection_manager::sides::IndexFieldStatsType;
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

    // Generage embeddings keeps time
    sleep(std::time::Duration::from_millis(500)).await;

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

    sleep(std::time::Duration::from_millis(500)).await;

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

#[tokio::test(flavor = "multi_thread")]
async fn test_commit_hooks() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let code = r#"
function selectEmbeddingsProperties() {
    return "The pen is on the table.";
}
export default {
    selectEmbeddingsProperties
}
"#;

    test_context
        .writer
        .insert_javascript_hook(
            collection_client.write_api_key,
            collection_client.collection_id,
            index_client.index_id,
            HookName::SelectEmbeddingsProperties,
            code.to_string(),
        )
        .await
        .unwrap();

    index_client
        .insert_documents(
            json!([json!({
                "title": "Today I want to listen only Max Pezzali.",
            })])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Generage embeddings keeps time
    sleep(std::time::Duration::from_millis(500)).await;

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    // Hook change the meaning of the text, so the exact match should not work
    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "Today I want to listen only Max Pezzali.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    test_context.commit_all().await.unwrap();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();
    let index_client = collection_client
        .get_test_index_client(index_client.index_id)
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "My dog is barking.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    index_client
        .insert_documents(
            json!([json!({
                "title": "My dog is barking.",
            })])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 2);

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "My dog is barking.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "The pen is on the table.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 2);

    let output = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "My dog is barking.",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_document_chunk_long_text_for_embedding_calculation() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([json!({
            "id": "1",
            "text": "foo ".repeat(1_000),
            })])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    sleep(std::time::Duration::from_millis(1_500)).await;

    let result = collection_client.reader_stats().await.unwrap();

    let mut result = result
        .indexes_stats
        .into_iter()
        .find(|i| i.id == index_client.index_id)
        .unwrap();
    let IndexFieldStatsType::UncommittedVector(stats) = result.fields_stats.remove(0).stats else {
        panic!("Expected vector field stats")
    };

    // Even if we insert only one document, we have two vectors because the text is chunked
    assert_eq!(stats.vector_count, 2);

    let result = collection_client
        .search(
            json!({
                "term": "foo ".repeat(256),
                "mode": "vector",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // In search result, the document is returned only once
    // because the id is the same
    // even it the document matches twice
    assert_eq!(result.count, 1);
    assert_eq!(result.hits[0].id, format!("{}:1", index_client.index_id));

    drop(test_context);
}
