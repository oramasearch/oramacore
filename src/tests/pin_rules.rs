use crate::tests::utils::{extrapolate_ids_from_result, init_log};
use crate::tests::utils::{wait_for, TestContext};
use anyhow::bail;
use futures::FutureExt;
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
        "Expected document ID ending with ':11', got: {}",
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

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rule_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents
    let docs: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "name": format!("Product {}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    // trigger the commit for the index and collection
    test_context.commit_all().await.unwrap();

    // Insert pin rule
    const TEST_COMMIT_RULE_ID: &str = "test-commit-rule";
    index_client
        .insert_pin_rules(
            json!({
                "id": TEST_COMMIT_RULE_ID,
                "conditions": [
                    {
                        "pattern": "product",
                        "anchoring": "contains"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "5",
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

    // commit only the pin rules, to be sure it is triggered
    test_context.commit_all().await.unwrap();

    // Check that the rule file exists in the reader side
    let reader_data_dir = &test_context.config.reader_side.config.data_dir;
    let reader_pin_rules_dir = reader_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("pin_rules");
    let reader_rule_file = reader_pin_rules_dir.join(format!("{TEST_COMMIT_RULE_ID}.rule"));
    assert!(
        reader_rule_file.exists(),
        "Pin rule file should exist in reader side: {reader_rule_file:?}"
    );

    // Check that the rule file exists in the writer side
    let writer_data_dir = &test_context.config.writer_side.config.data_dir;
    let writer_pin_rules_dir = writer_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("pin_rules");
    let writer_rule_file = writer_pin_rules_dir.join(format!("{TEST_COMMIT_RULE_ID}.rule"));
    assert!(
        writer_rule_file.exists(),
        "Pin rule file should exist in writer side: {writer_rule_file:?}"
    );

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;

    // Now reload the system to verify pin rules are loaded from disk
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    // Wait for the reader to fully load the collection and pin rules
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let pin_rules_reader = collection.get_pin_rules_reader("reload-test").await;
            let rule_ids = pin_rules_reader.get_rule_ids();

            if !rule_ids.contains(&TEST_COMMIT_RULE_ID.to_string()) {
                anyhow::bail!("{TEST_COMMIT_RULE_ID} not found after reload");
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Pin rules should be loaded from disk after reload");

    // Verify the rules still work correctly after reload
    let result_after_reload = collection_client
        .search(
            json!({
                "term": "product"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids_after_reload = extrapolate_ids_from_result(&result_after_reload);
    assert_eq!(
        &ids_after_reload[0], "0",
        "First result should be natural result after reload"
    );
    assert_eq!(
        &ids_after_reload[1], "5",
        "Second result should be promoted {TEST_COMMIT_RULE_ID} after reload"
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rule_delete_removes_files() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents
    let docs: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "name": format!("Product {}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    // Insert pin rule
    const TEST_DELETE_RULE_ID: &str = "test-delete-rule";
    index_client
        .insert_pin_rules(
            json!({
                "id": TEST_DELETE_RULE_ID,
                "conditions": [
                    {
                        "pattern": "product",
                        "anchoring": "contains"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "5",
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

    // Commit to ensure files are written to disk
    test_context.commit_all().await.unwrap();

    // Verify that the rule files exist in both reader and writer sides before deletion
    let reader_data_dir = &test_context.config.reader_side.config.data_dir;
    let reader_pin_rules_dir = reader_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("pin_rules");
    let reader_rule_file = reader_pin_rules_dir.join(format!("{TEST_DELETE_RULE_ID}.rule"));
    assert!(
        reader_rule_file.exists(),
        "Pin rule file should exist in reader side before deletion: {reader_rule_file:?}"
    );

    let writer_data_dir = &test_context.config.writer_side.config.data_dir;
    let writer_pin_rules_dir = writer_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("pin_rules");
    let writer_rule_file = writer_pin_rules_dir.join(format!("{TEST_DELETE_RULE_ID}.rule"));
    assert!(
        writer_rule_file.exists(),
        "Pin rule file should exist in writer side before deletion: {writer_rule_file:?}"
    );

    // Delete the pin rule
    index_client
        .delete_pin_rules(TEST_DELETE_RULE_ID.to_string())
        .await
        .unwrap();

    // Commit to ensure deletion is persisted to disk
    test_context.commit_all().await.unwrap();

    // Verify that the rule files are removed from both reader and writer sides
    assert!(
        !reader_rule_file.exists(),
        "Pin rule file should be removed from reader side: {reader_rule_file:?}"
    );
    assert!(
        !writer_rule_file.exists(),
        "Pin rule file should be removed from writer side: {writer_rule_file:?}"
    );

    // Verify the rule no longer affects search results
    let result = collection_client
        .search(
            json!({
                "term": "product"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(
        &ids[0], "0",
        "First result should be natural order after rule deletion"
    );
    assert_ne!(
        &ids[1], "5",
        "Document 5 should not be promoted after rule deletion"
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rule_updated_when_document_id_changes() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..5_u8)
        .map(|i| {
            json!({
                "id": format!("doc-{}", i),
                "name": format!("Product {}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    const TEST_UPDATE_RULE_ID: &str = "test-update-doc-rule";
    index_client
        .insert_pin_rules(
            json!({
                "id": TEST_UPDATE_RULE_ID,
                "conditions": [
                    {
                        "pattern": "product",
                        "anchoring": "contains"
                    }
                ],
                "consequence": {
                    "promote": [
                        {
                            "doc_id": "doc-1",
                            "position": 0
                        },
                        {
                            "doc_id": "doc-3",
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

    let doc_ids_before = {
        let collection = collection_client
            .reader
            .get_collection(
                collection_client.collection_id,
                collection_client.read_api_key,
            )
            .await
            .unwrap();
        let pin_rules_reader = collection.get_pin_rules_reader("test").await;
        let rule = pin_rules_reader.get_by_id(TEST_UPDATE_RULE_ID).unwrap();

        rule.consequence.promote.clone()
    };

    // Now update one of the documents that's in the rule, changing its ID
    index_client
        .update_documents(
            serde_json::from_value(json!({
                "strategy": "merge",
                "documents": [
                    {
                        "id": "doc-3", // Update the document 3 with a new one
                        "name": "Updated Product 3",
                    }
                ]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Wait for the pin rule to be updated in the reader
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        let expected_doc_ids = doc_ids_before.clone();
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let pin_rules_reader = collection.get_pin_rules_reader("test").await;
            let rule = pin_rules_reader.get_by_id(TEST_UPDATE_RULE_ID).unwrap();

            if rule.consequence.promote.len() != 2 {
                bail!(
                    "Expected 2 promotions in pin rule after update, got {}",
                    rule.consequence.promote.len()
                );
            }

            if rule.consequence.promote[0].doc_id != expected_doc_ids[0].doc_id {
                bail!(
                    "First document ID changed unexpectedly: {:?} != {:?}",
                    rule.consequence.promote[0].doc_id,
                    expected_doc_ids[0].doc_id
                );
            }

            // Check if the document IDs have been updated
            // The second document should have changed its internal ID but still be referenced
            // because the document was replaced with a new one
            if rule.consequence.promote[1].doc_id == expected_doc_ids[1].doc_id {
                bail!(
                    "Second document ID should have changed after update: {:?}",
                    rule.consequence.promote[1].doc_id,
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Pin rule should be updated after document ID change");

    drop(test_context);
}
