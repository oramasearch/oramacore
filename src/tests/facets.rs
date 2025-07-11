use std::collections::HashMap;

use serde_json::json;

use crate::tests::utils::{init_log, TestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_number() {
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

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "facets": {
                    "number": {
                        "ranges": [
                            {
                                "from": 0,
                                "to": 10,
                            },
                            {
                                "from": 0.5,
                                "to": 10.5,
                            },
                            {
                                "from": -10,
                                "to": 10,
                            },
                            {
                                "from": -10,
                                "to": -1,
                            },
                            {
                                "from": 1,
                                "to": 100,
                            },
                            {
                                "from": 99,
                                "to": 105,
                            },
                            {
                                "from": 102,
                                "to": 105,
                            },
                        ],
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let facets = output.facets.expect("Facet should be there");
    let number_facet = facets
        .get("number")
        .expect("Facet on field 'number' should be there");

    assert_eq!(number_facet.count, 7);
    assert_eq!(number_facet.values.len(), 7);

    assert_eq!(
        number_facet.values,
        HashMap::from_iter(vec![
            ("-10--1".to_string(), 0),
            ("-10-10".to_string(), 11),
            ("0-10".to_string(), 11),
            ("0.5-10.5".to_string(), 10),
            ("1-100".to_string(), 99),
            ("102-105".to_string(), 0),
            ("99-105".to_string(), 1),
        ])
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_bool() {
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
                "facets": {
                    "bool": {
                        "true": true,
                        "false": true,
                    },
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let facets = output.facets.expect("Facet should be there");
    let bool_facet = facets
        .get("bool")
        .expect("Facet on field 'bool' should be there");

    assert_eq!(bool_facet.count, 2);
    assert_eq!(bool_facet.values.len(), 2);

    assert_eq!(
        bool_facet.values,
        HashMap::from_iter(vec![("true".to_string(), 50), ("false".to_string(), 50),])
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_string() {
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
                "facets": {
                    "text": {},
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let facets = output.facets.expect("Facet should be there");
    let string_facet = facets
        .get("text")
        .expect("Facet on field 'bool' should be there");

    assert_eq!(string_facet.count, 2);
    assert_eq!(string_facet.values.len(), 2);

    assert_eq!(
        string_facet.values,
        HashMap::from_iter(vec![
            ("text true".to_string(), 50),
            ("text false".to_string(), 50),
        ])
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_unknown_field() {
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
                "facets": {
                    "unknown": {},
                }
            })
            .try_into()
            .unwrap(),
        )
        .await;

    assert!(format!("{output:?}").contains("Unknown field name 'unknown'"));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_should_based_on_term() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                json!({
                    "id": "1",
                    "text": "text",
                    "bool": true,
                    "number": 1,
                }),
                json!({
                    "id": "2",
                    "text": "text text",
                    "bool": false,
                    "number": 2,
                }),
                // This document doens't match the term
                // so it should not be counted in the facets
                json!({
                    "id": "3",
                    "text": "another",
                    "bool": true,
                    "number": 1,
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
                "facets": {
                    "bool": {
                        "true": true,
                        "false": true,
                    },
                    "number": {
                        "ranges": [
                            {
                                "from": 0,
                                "to": 10,
                            },
                        ],
                    },
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let facets = output.facets.expect("Facet should be there");
    let bool_facet = facets
        .get("bool")
        .expect("Facet on field 'bool' should be there");

    assert_eq!(bool_facet.count, 2);
    assert_eq!(bool_facet.values.len(), 2);

    assert_eq!(
        bool_facet.values,
        HashMap::from_iter(vec![("true".to_string(), 1), ("false".to_string(), 1),])
    );

    let number_facet = facets
        .get("number")
        .expect("Facet on field 'number' should be there");

    assert_eq!(number_facet.count, 1);
    assert_eq!(number_facet.values.len(), 1);

    assert_eq!(
        number_facet.values,
        HashMap::from_iter(vec![("0-10".to_string(), 2),])
    );

    drop(test_context);
}
