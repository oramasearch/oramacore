use assert_approx_eq::assert_approx_eq;
use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_fulltext_search() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client.insert_documents(json!([
        {
            "id": "1",
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        },
        {
            "id": "2",
            "text": "Curabitur sem tortor, interdum in rutrum in, dignissim vestibulum metus.",
        }
    ]).try_into().unwrap()).await.unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "Lorem ipsum",
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

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fulltext_search_should_work_after_commit() {
    init_log();
    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client.insert_documents(json!([
        {
            "id": "1",
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        },
        {
            "id": "2",
            "text": "Curabitur sem tortor, interdum in rutrum in, dignissim vestibulum metus.",
        }
    ]).try_into().unwrap()).await.unwrap();

    let output1 = collection_client
        .search(
            json!({
                "term": "Lorem ipsum",
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
                "term": "Lorem ipsum",
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
                "term": "Lorem ipsum",
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

    // Committed before and after the reload should be the same
    // NB: the uncommitted could be different because it doens't implement phase matching
    assert_approx_eq!(output2.hits[0].score, output3.hits[0].score);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_documents_order() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                json!({
                    "id": "1",
                    "text": "This is a long text with a lot of words",
                }),
                json!({
                    "id": "2",
                    "text": "This is a smaller text",
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
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 2);
    assert_eq!(output.hits.len(), 2);
    assert_eq!(output.hits[0].id, format!("{}:2", index_client.index_id));
    assert_eq!(output.hits[1].id, format!("{}:1", index_client.index_id));
    assert!(output.hits[0].score > output.hits[1].score);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_documents_limit() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "limit": 10,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 100);
    assert_eq!(output.hits.len(), 10);
    assert_eq!(
        output.hits[0].id,
        format!("{}:99", index_client.index_id.as_str())
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:98", index_client.index_id.as_str())
    );
    assert_eq!(
        output.hits[2].id,
        format!("{}:97", index_client.index_id.as_str())
    );
    assert_eq!(
        output.hits[3].id,
        format!("{}:96", index_client.index_id.as_str())
    );
    assert_eq!(
        output.hits[4].id,
        format!("{}:95", index_client.index_id.as_str())
    );

    assert!(output.hits[0].score > output.hits[1].score);
    assert!(output.hits[1].score > output.hits[2].score);
    assert!(output.hits[2].score > output.hits[3].score);
    assert!(output.hits[3].score > output.hits[4].score);
}
