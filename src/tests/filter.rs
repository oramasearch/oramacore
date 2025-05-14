use serde_json::json;

use crate::tests::utils::{init_log, TestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_search_on_unknown_field() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    collection_client.create_index().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Doe",
                "where": {
                    "unknown_field": json!({
                        "eq": 1,
                    }),
                },
            })
            .try_into()
            .unwrap(),
        )
        .await;

    println!("result: {:?}", result);

    assert!(result.is_err());
    assert_eq!(
        format!("{}", result.unwrap_err()),
        "Cannot filter by \"unknown_field\": unknown field".to_string(),
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_number() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // EQ

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "eq": 50,
                    },
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.hits[0].id, format!("{}:50", index_client.index_id));

    // GT

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gt": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 100 - 3);

    // GTE

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gte": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 100 - 2);

    // LT

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "lt": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 2);

    // LTE

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "lte": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 3);

    // BETWEEN

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "between": [2, 4],
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_bool() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "bool": i % 2 == 0,
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
                "where": {
                    "bool": true,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 50);
    assert_eq!(output.hits.len(), 10);
    for hit in output.hits.iter() {
        let id =
            serde_json::from_str::<serde_json::Value>(hit.document.as_ref().unwrap().inner.get())
                .unwrap()
                .get("id")
                .unwrap()
                .as_str()
                .unwrap()
                .parse::<usize>()
                .unwrap();
        assert_eq!(id % 2, 0);
    }

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "bool": false,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 50);
    assert_eq!(output.hits.len(), 10);
    for hit in output.hits.iter() {
        let id =
            serde_json::from_str::<serde_json::Value>(hit.document.as_ref().unwrap().inner.get())
                .unwrap()
                .get("id")
                .unwrap()
                .as_str()
                .unwrap()
                .parse::<usize>()
                .unwrap();
        assert_eq!(id % 2, 1);
    }

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_string() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": format!("text {}", i % 2 == 0),
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
                "where": {
                    "text": "text true",
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 50);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_array_types() {
    init_log();

    let document_count = 10;

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": vec!["text ".repeat(i + 1)],
                "number": vec![i],
                "bool": vec![i % 2 == 0],
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, document_count);

    let result = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "eq": 5,
                    },
                }
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
                "term": "text",
                "where": {
                    "bool": true,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 5);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_and_or_not() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": format!("text {}", i % 3 == 0),
                "number": i,
                "bool": i % 2 == 0,
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
                "where": {
                    "and": [
                        { "bool": true },
                        { "number": { "gte": 50 } },
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 25);

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "or": [
                        { "bool": true },
                        { "number": { "gte": 50 } },
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 75);

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "not": {
                        "bool": true,
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 50);

    drop(test_context);
}
