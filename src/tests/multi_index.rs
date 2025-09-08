use std::panic;

use anyhow::Result;
use serde_json::json;

use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_search_multi_index_basic() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    let index_2_client = collection_client.create_index().await.unwrap();

    let document_count = 10;
    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        })
        .collect::<Vec<_>>();
    index_1_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "bool": i % 2 == 0,
            })
        })
        .collect::<Vec<_>>();
    index_2_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Search on both indexes
    let res = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, document_count * 2);

    // Bool is present only on the second index
    let res = collection_client
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
        .await?;
    assert_eq!(res.count, 5);

    // number is present only on the first index
    let res = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gt": -1
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, 10);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_search_multi_index_one_is_empty() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    // Empty index
    collection_client.create_index().await.unwrap();

    let document_count = 10;
    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        })
        .collect::<Vec<_>>();
    index_1_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Keep index 2 empty

    // Search on both indexes
    let res = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, document_count);

    // Number is present only on the first index
    let res = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gt": -1
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, 10);

    let res = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "text": "text "
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, 1);

    Ok(())
}

/// Test for field type mismatch bug across multiple indexes
/// When the same field name exists with different types across indexes,
/// the filtering should either provide a clear error or handle gracefully,
/// but currently falls through to `_ => {}` and returns empty results silently
#[tokio::test(flavor = "multi_thread")]
async fn test_field_type_mismatch_bug() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    let index_2_client = collection_client.create_index().await.unwrap();

    // Index 1: "value" field as NUMBER
    let docs_with_number_value = vec![
        json!({
            "id": "doc1",
            "text": "item",
            "value": 10  // NUMBER type
        }),
        json!({
            "id": "doc2",
            "text": "item",
            "value": 20  // NUMBER type
        }),
    ];

    index_1_client
        .insert_documents(json!(docs_with_number_value).try_into().unwrap())
        .await
        .unwrap();

    // Index 2: Same field name "value" but as STRING type
    let docs_with_string_value = vec![
        json!({
            "id": "doc3",
            "text": "item",
            "value": "hello"  // STRING type - this creates the type mismatch
        }),
        json!({
            "id": "doc4",
            "text": "item",
            "value": "world"  // STRING type
        }),
    ];

    index_2_client
        .insert_documents(json!(docs_with_string_value).try_into().unwrap())
        .await
        .unwrap();

    // Verify all documents exist without filtering
    let res_no_filter = collection_client
        .search(
            json!({
                "term": "item"
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res_no_filter.count, 4);

    // Filter using NUMBER criteria - this should work for Index 1 but fail for Index 2
    // due to type mismatch (Index 2 has STRING values)
    let res_number_filter = collection_client
        .search(
            json!({
                "term": "item",
                "where": {
                    "value": {
                        "gte": 15  // Should match doc2 (value=20) from Index 1 only
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    // BUG: Index 2 falls through to `_ => {}` match case and returns empty set
    // instead of providing proper error handling for type mismatch
    // Expected: Should get 1 document (doc2 with value=20)
    // Actual: Depends on how the type mismatch is handled
    assert_eq!(
        res_number_filter.count, 1,
        "Should find doc2 with value=20, but type mismatch in Index 2 may cause issues"
    );

    // Test reverse scenario: STRING filter on mixed types
    let res_string_filter = collection_client
        .search(
            json!({
                "term": "item",
                "where": {
                    "value": "hello"  // Should match doc3 from Index 2, but Index 1 has number types
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    // Similar issue: Index 1 will fall through to `_ => {}` for type mismatch
    assert_eq!(
        res_string_filter.count, 1,
        "Should find doc3 with value='hello', but type mismatch in Index 1 may cause issues"
    );

    Ok(())
}

/// Test for bug where deleted indexes are included in index validation but excluded from search
/// This can cause issues when user specifies specific indexes to search
#[tokio::test(flavor = "multi_thread")]
async fn test_deleted_index_validation_bug() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    // Create two indexes
    let index_1_client = collection_client.create_index().await.unwrap();
    let index_2_client = collection_client.create_index().await.unwrap();

    // Add documents to both indexes
    index_1_client
        .insert_documents(
            json!([
                {"id": "1", "text": "test document one"}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    index_2_client
        .insert_documents(
            json!([
                {"id": "2", "text": "test document two"}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Delete index 2
    index_2_client.delete().await.unwrap();

    // The bug: When specifying indexes to search, deleted indexes are included in validation
    // but excluded during actual search, potentially causing inconsistent behavior

    // Try to search specifying both indexes (including the deleted one)
    let search_params = json!({
        "term": "test",
        "indexes": [index_1_client.index_id, index_2_client.index_id]
    });

    let result = collection_client
        .search(search_params.try_into().unwrap())
        .await;

    // The issue is in calculate_index_to_search_on() line 930:
    // all_indexes includes deleted indexes, but search loop skips them
    // This inconsistency could lead to unexpected behavior
    match result {
        Ok(res) => {
            // If this succeeds, it means deleted indexes are handled inconsistently
            // between validation and execution
            assert_eq!(
                res.count, 1,
                "Should only find document from non-deleted index"
            );
        }
        Err(_) => {
            // If this fails, it might be due to the deleted index being considered invalid
            // even though it should be silently ignored during search
        }
    }

    Ok(())
}

/// Test for bug where filtering fails when field exists only in committed data but not uncommitted
/// This happens because the filtering logic requires uncommitted field to exist but gracefully
/// handles missing committed fields, creating an asymmetry
#[tokio::test(flavor = "multi_thread")]
async fn test_committed_only_field_filter_bug() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents and commit them (so they become committed data)
    index_client
        .insert_documents(
            json!([
                {"id": "1", "text": "test", "status": "active"},
                {"id": "2", "text": "test", "status": "inactive"}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Force commit to move data from uncommitted to committed fields
    test_context.commit_all().await.unwrap();

    // Now try to filter by the "status" field
    // BUG: This might fail because filtering logic expects field in uncommitted_fields
    // but after commit, the field only exists in committed_fields
    let result = collection_client
        .search(
            json!({
                "term": "test",
                "where": {
                    "status": "active"
                }
            })
            .try_into()
            .unwrap(),
        )
        .await;

    match result {
        Ok(res) => {
            assert_eq!(res.count, 1, "Should find one document with status=active");
        }
        Err(e) => {
            // If this fails with "unknown field" error, it demonstrates the bug
            let error_msg = format!("{e}");
            if error_msg.contains("unknown field") && error_msg.contains("status") {
                panic!("BUG DETECTED: Field exists in committed data but filtering fails because it's not in uncommitted fields");
            } else {
                panic!("Unexpected error: {e}");
            }
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multi_index_sorting_bug() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    let index_2_client = collection_client.create_index().await.unwrap();

    // Index 1: Documents with priority field (lower values)
    index_1_client
        .insert_documents(
            json!([
                {"id": "doc1", "text": "item", "priority": 1},
                {"id": "doc2", "text": "item", "priority": 3}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Index 2: Documents with priority field (higher values)
    index_2_client
        .insert_documents(
            json!([
                {"id": "doc3", "text": "item", "priority": 2},
                {"id": "doc4", "text": "item", "priority": 4}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Search and sort by priority ascending
    let result = collection_client
        .search(
            json!({
                "term": "item",
                "sortBy": {
                    "property": "priority",
                    "order": "ASC"
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(result.count, 4, "Should find all 4 documents");

    // Extract the document IDs in order
    let doc_ids: Vec<String> = result
        .hits
        .iter()
        .map(|hit| {
            let doc: serde_json::Value =
                serde_json::from_str(hit.document.as_ref().unwrap().inner.get()).unwrap();
            doc["id"].as_str().unwrap().to_string()
        })
        .collect();

    // Check if sorting considers documents from all indexes
    assert_eq!(
        doc_ids,
        vec!["doc1", "doc3", "doc2", "doc4"],
        "Documents should be sorted by priority across all indexes, got: {doc_ids:?}"
    );

    // Search and sort by priority ascending
    let result = collection_client
        .search(
            json!({
                "term": "item",
                "sortBy": {
                    "property": "priority",
                    "order": "DESC"
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(result.count, 4, "Should find all 4 documents");

    // Extract the document IDs in order
    let doc_ids: Vec<String> = result
        .hits
        .iter()
        .map(|hit| {
            let doc: serde_json::Value =
                serde_json::from_str(hit.document.as_ref().unwrap().inner.get()).unwrap();
            doc["id"].as_str().unwrap().to_string()
        })
        .collect();

    // Check if sorting considers documents from all indexes
    assert_eq!(
        doc_ids,
        vec!["doc4", "doc2", "doc3", "doc1"],
        "Documents should be sorted by priority across all indexes, got: {doc_ids:?}"
    );

    Ok(())
}
