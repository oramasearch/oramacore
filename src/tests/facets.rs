use std::collections::HashMap;

use serde_json::json;

use crate::collection_manager::sides::read::ReadError;
use crate::tests::utils::{init_log, TestContext};
use crate::types::FacetResult;

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

    let err = output.expect_err("Should return an error");

    assert!(matches!(
        err,
        ReadError::FacetFieldNotFound(_)
    ));

    assert!(format!("{err:?}").contains("unknown"));

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

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_real_case() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs = (0..10)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "path": format!("path{}", i),
                "siteSection": format!("Section {}", i),
                "pageTitle": format!("The title {}", i),
                "pageSectionTitle": format!("The page section title {}", i),
                "pageSectionContent": format!("The page section content {}", i),
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "title",
                "facets": {
                    "siteSection": {}
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        output.facets,
        Some(HashMap::from([(
            "siteSection".to_string(),
            FacetResult {
                count: 10,
                values: HashMap::from([
                    ("Section 0".to_string(), 1),
                    ("Section 1".to_string(), 1),
                    ("Section 2".to_string(), 1),
                    ("Section 3".to_string(), 1),
                    ("Section 4".to_string(), 1),
                    ("Section 5".to_string(), 1),
                    ("Section 6".to_string(), 1),
                    ("Section 7".to_string(), 1),
                    ("Section 8".to_string(), 1),
                    ("Section 9".to_string(), 1),
                ])
            }
        )]))
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_with_filters() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs = (0..10)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "title": format!("title {}", i),
                "category": if i % 2 == 0 { "A" } else { "B" },
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "category": "A",
                },
                "facets": {
                    "category": {}
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        output.facets,
        Some(HashMap::from([(
            "category".to_string(),
            FacetResult {
                count: 2,
                values: HashMap::from([("A".to_string(), 5), ("B".to_string(), 5),])
            }
        )]))
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_on_multiple_index_collection() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_client1 = collection_client.create_index().await.unwrap();
    let docs = (0..10)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "title": format!("title {}", i),
                "category": if i % 2 == 0 { "A" } else { "B" },
            })
        })
        .collect::<Vec<_>>();
    index_client1
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let index_client2 = collection_client.create_index().await.unwrap();
    let docs = (0..10)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "title": format!("title {}", i),
                "category": if i % 2 == 0 { "A" } else { "B" },
            })
        })
        .collect::<Vec<_>>();
    index_client2
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "facets": {
                    "category": {}
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let Some(facets) = output.facets else {
        panic!("Facet should be there");
    };
    let category_facet = facets
        .get("category")
        .expect("Facet on field 'category' should be there");
    assert_eq!(
        category_facet.values,
        HashMap::from([("A".to_string(), 10), ("B".to_string(), 10),])
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_with_different_shaped_index() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    // index 1 is empty
    let _ = collection_client.create_index().await.unwrap();

    let index_client2 = collection_client.create_index().await.unwrap();
    let docs = (0..10)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "title": format!("title {}", i),
                "category": if i % 2 == 0 { "A" } else { "B" },
            })
        })
        .collect::<Vec<_>>();
    index_client2
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "facets": {
                    "category": {}
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let Some(facets) = output.facets else {
        panic!("Facet should be there");
    };
    let category_facet = facets
        .get("category")
        .expect("Facet on field 'category' should be there");
    assert_eq!(
        category_facet.values,
        HashMap::from([("A".to_string(), 5), ("B".to_string(), 5),])
    );

    drop(test_context);
}
