use crate::tests::utils::TestContext;
use crate::tests::utils::{extrapolate_ids_from_result, init_log};
use serde_json::json;
use std::convert::TryInto;

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_simple() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
                "run": format!("run-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
                "id": "rule-1",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "is"
                    },
                    {
                        "pattern": "running",
                        "anchoring": "is",
                        "normalization": "stem",
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "5",
                            "position": 1
                        },
                        {
                            "doc_id": "7",
                            "position": 2
                        }
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // The pin rule should promote document with id "5" to position 1
    // The actual ID will be in format "index_id:5"
    assert!(
        result.hits[1].id.ends_with(":5"),
        "Expected document ID ending with ':5', got: {}",
        result.hits[1].id
    );
    assert!(
        result.hits[2].id.ends_with(":7"),
        "Expected document ID ending with ':7', got: {}",
        result.hits[2].id
    );

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "5", "7", "1", "2", "3", "4", "6", "8", "9"]);

    let stemmed_result = collection_client
        .search(
            json!({
                "term": "runs"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let stemmed_ids = extrapolate_ids_from_result(&stemmed_result);
    assert_eq!(&stemmed_ids, &ids);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_multiple_indexes() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client1 = collection_client.create_index().await.unwrap();
    let index_client2 = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client1
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let docs: Vec<_> = (10_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client2
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client1
        .insert_pin_rules(
            json!({
                "id": "rule-1",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "is"
                    },
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "5",
                            "position": 1
                        },
                        {
                            "doc_id": "7",
                            "position": 4
                        }
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    index_client2
        .insert_pin_rules(
            json!({
                "id": "rule-2",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "startsWith",
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "11",
                            "position": 2
                        },
                        {
                            "doc_id": "15",
                            "position": 3
                        }
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // The pin rule should promote document with id "5" to position 1
    // The actual ID will be in format "index_id:5"
    assert!(
        result.hits[1].id.ends_with(":5"),
        "Expected document ID ending with ':5', got: {}",
        result.hits[1].id
    );
    assert!(
        result.hits[2].id.ends_with(":11"),
        "Expected document ID ending with ':7', got: {}",
        result.hits[2].id
    );
    assert!(
        result.hits[3].id.ends_with(":15"),
        "Expected document ID ending with ':15', got: {}",
        result.hits[3].id
    );
    assert!(
        result.hits[4].id.ends_with(":7"),
        "Expected document ID ending with ':7', got: {}",
        result.hits[4].id
    );

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "5", "11", "15", "7", "1", "2", "3", "4", "6"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_already_returned() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
                "id": "rule-1",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "is"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "0",
                            "position": 3
                        }
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2", "3", "0", "4", "5", "6", "7", "8", "9"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_after_the_pagination_window() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
                "id": "rule-1",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "is"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "0",
                            "position": 3000
                        },
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_pagination() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
                "id": "rule-1",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "is"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "0",
                            "position": 3
                        },
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2", "3", "0", "4", "5", "6", "7", "8", "9"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 0,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 1,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["2", "3"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 2,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["3", "0"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 3,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "4"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 4,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["4", "5"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_before_insert_simple() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_pin_rules(
            json!({
                "id": "rule-1",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "is"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "5",
                            "position": 1
                        },
                        {
                            "doc_id": "7",
                            "position": 2
                        }
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "5", "7", "1", "2", "3", "4", "6", "8", "9"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_with_sort() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_pin_rules(
            json!({
                "id": "rule-1",
                "conditions": [
                    {
                        "pattern": "c",
                        "anchoring": "is"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "5",
                            "position": 1
                        },
                        {
                            "doc_id": "7",
                            "position": 2
                        }
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
                "n": i,
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "sortBy": {
                    "order": "ASC",
                    "property": "n",
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "5", "7", "1", "2", "3", "4", "6", "8", "9"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "sortBy": {
                    "order": "DESC",
                    "property": "n",
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(
        &ids,
        &["19", "5", "7", "18", "17", "16", "15", "14", "13", "12"]
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_promote_non_matching_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents
    index_client
        .insert_documents(
            json!([
                { "id": "1", "name": "Blue Jeans" },
                { "id": "2", "name": "Red T-Shirt" },
                { "id": "3", "name": "Green Hoodie" },
                { "id": "4", "name": "Yellow Socks" }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Create pin rule that promotes document "2" when searching for "Blue Jeans"
    // Document "2" won't be in the original search results since it doesn't match "Blue Jeans"
    index_client
        .insert_pin_rules(
            json!({
                "id": "test_rule",
                "conditions": [
                    {
                        "anchoring": "is",
                        "pattern": "Blue Jeans"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "2",
                            "position": 1
                        }
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Blue Jeans"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    // We expect both documents: "1" (matches the search) and "2" (promoted by pin rule)
    assert_eq!(result.hits.len(), 2);
    assert_eq!(&ids, &["1", "2"]);

    // Verify the promoted document has a score of 0.0 since it wasn't in original results
    let doc_2_hit = result
        .hits
        .iter()
        .find(|hit| hit.id.ends_with(":2"))
        .unwrap();
    assert_eq!(doc_2_hit.score, 0.0);

    // Verify the original matching document has a positive score
    let doc_1_hit = result
        .hits
        .iter()
        .find(|hit| hit.id.ends_with(":1"))
        .unwrap();
    assert!(doc_1_hit.score > 0.0);

    drop(test_context);
}
