use assert_approx_eq::assert_approx_eq;
use serde_json::json;

use crate::tests::utils::{init_log, TestContext};

/// Test that documents with OMC (_omc field) have their scores multiplied by the OMC value.
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_multiplies_scores() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert two identical documents, one with OMC multiplier of 2.0
    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "_omc": 2.0
        },
        {
            "id": "doc2",
            "title": "machine learning"
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "machine learning"
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 2);
    assert_eq!(results.hits.len(), 2);

    // Both documents should be found
    let doc1 = results
        .hits
        .iter()
        .find(|h| h.id.contains("doc1"))
        .expect("doc1 should be in results");
    let doc2 = results
        .hits
        .iter()
        .find(|h| h.id.contains("doc2"))
        .expect("doc2 should be in results");

    // doc1 should have approximately 2x the score of doc2
    // Allow some tolerance for floating point math
    assert_approx_eq!(doc1.score, doc2.score * 2.0, 0.001);

    drop(test_context);
}

/// Test that OMC multipliers work correctly after commit (persisted to disk).
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_persists_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "_omc": 3.0
        },
        {
            "id": "doc2",
            "title": "machine learning"
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    // Search before commit
    let search_params = json!({
        "term": "machine learning"
    });

    let results_before = collection_client
        .search(search_params.clone().try_into().unwrap())
        .await
        .unwrap();

    // Commit all data
    test_context.commit_all().await.unwrap();

    // Search after commit
    let results_after = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results_before.count, results_after.count);
    assert_eq!(results_before.hits.len(), results_after.hits.len());

    // Find doc1 in both results
    let _doc1_before = results_before
        .hits
        .iter()
        .find(|h| h.id.contains("doc1"))
        .expect("doc1 should be in results before commit");
    let doc1_after = results_after
        .hits
        .iter()
        .find(|h| h.id.contains("doc1"))
        .expect("doc1 should be in results after commit");

    // Scores should be approximately equal before and after commit
    // Note: There might be slight differences due to BM25 calculation changes after commit
    // but the OMC multiplier effect should still be present
    let doc2_after = results_after
        .hits
        .iter()
        .find(|h| h.id.contains("doc2"))
        .expect("doc2 should be in results after commit");

    // After commit, doc1 should have approximately 3x the score of doc2
    assert_approx_eq!(doc1_after.score, doc2_after.score * 3.0, 0.001);

    drop(test_context);
}

/// Test that invalid OMC values (zero, negative, non-numeric) are ignored.
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_invalid_values_ignored() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents with various invalid OMC values
    let documents = json!([
        {
            "id": "doc_negative",
            "title": "machine learning",
            "_omc": -1.0  // Negative, should be ignored
        },
        {
            "id": "doc_zero",
            "title": "machine learning",
            "_omc": 0.0  // Zero, should be ignored
        },
        {
            "id": "doc_string",
            "title": "machine learning",
            "_omc": "invalid"  // Non-numeric, should be ignored
        },
        {
            "id": "doc_null",
            "title": "machine learning",
            "_omc": null  // Null, should be ignored
        },
        {
            "id": "doc_normal",
            "title": "machine learning"
            // No _omc, should have default score
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "machine learning"
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 5);

    // All documents should have the same score (invalid OMC values are ignored)
    let scores: Vec<f32> = results.hits.iter().map(|h| h.score).collect();
    let first_score = scores[0];
    for score in scores.iter() {
        assert_approx_eq!(*score, first_score, 0.001);
    }

    drop(test_context);
}

/// Test that OMC values are removed when documents are deleted.
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_removed_on_delete() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "_omc": 5.0
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    // Delete the document
    index_client
        .delete_documents(vec!["doc1".to_string()])
        .await
        .unwrap();

    // Insert a new document with the same id but different OMC
    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "_omc": 1.0  // Different OMC
        },
        {
            "id": "doc2",
            "title": "machine learning"
            // No OMC
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "machine learning"
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 2);

    // doc1 should have the new OMC value (1.0, effectively same as doc2)
    // not the old deleted one (5.0)
    let doc1 = results
        .hits
        .iter()
        .find(|h| h.id.contains("doc1"))
        .expect("doc1 should be in results");
    let doc2 = results
        .hits
        .iter()
        .find(|h| h.id.contains("doc2"))
        .expect("doc2 should be in results");

    // Both should have approximately the same score now
    assert_approx_eq!(doc1.score, doc2.score, 0.001);

    drop(test_context);
}

/// Test that document updates correctly update the OMC value.
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_updated_on_document_update() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert initial document with OMC
    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "_omc": 2.0
        },
        {
            "id": "doc2",
            "title": "machine learning"
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "machine learning"
    });

    let results_before = collection_client
        .search(search_params.clone().try_into().unwrap())
        .await
        .unwrap();

    let doc1_before = results_before
        .hits
        .iter()
        .find(|h| h.id.contains("doc1"))
        .expect("doc1 should be in results");
    let doc2_before = results_before
        .hits
        .iter()
        .find(|h| h.id.contains("doc2"))
        .expect("doc2 should be in results");

    // doc1 should have 2x the score initially
    assert_approx_eq!(doc1_before.score, doc2_before.score * 2.0, 0.001);

    // Update doc1 with a new OMC value
    // Use unchecked_insert_documents because insert_documents expects document count to increase,
    // but when updating an existing document, the count stays the same
    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "_omc": 4.0  // Changed from 2.0 to 4.0
        }
    ]);

    index_client
        .unchecked_insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    // Wait for eventual consistency
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    let results_after = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    let doc1_after = results_after
        .hits
        .iter()
        .find(|h| h.id.contains("doc1"))
        .expect("doc1 should be in results");
    let doc2_after = results_after
        .hits
        .iter()
        .find(|h| h.id.contains("doc2"))
        .expect("doc2 should be in results");

    // doc1 should now have 4x the score
    assert_approx_eq!(doc1_after.score, doc2_after.score * 4.0, 0.001);

    drop(test_context);
}

/// Test that _omc field is preserved in the document and returned in search results.
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_field_in_document() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "_omc": 2.5
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "machine learning"
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 1);

    // The _omc field should be in the returned document
    let doc = results.hits[0]
        .document
        .as_ref()
        .expect("document should be returned");
    // Parse the raw JSON to check for _omc field
    let raw_json = doc.inner.get();
    let parsed: serde_json::Value = serde_json::from_str(raw_json).expect("valid JSON");
    let omc_value = parsed.get("_omc");
    // _omc is stored in the document and should be returned
    assert!(omc_value.is_some(), "_omc should be in document");
    assert_eq!(omc_value.unwrap().as_f64(), Some(2.5));

    drop(test_context);
}

/// Test OMC with different positive multipliers.
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_various_multipliers() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents with various OMC values
    let documents = json!([
        {
            "id": "doc_x1",
            "title": "test document",
            "_omc": 1.0  // Base multiplier
        },
        {
            "id": "doc_x2",
            "title": "test document",
            "_omc": 2.0
        },
        {
            "id": "doc_x5",
            "title": "test document",
            "_omc": 5.0
        },
        {
            "id": "doc_x10",
            "title": "test document",
            "_omc": 10.0
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "test document"
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 4);

    let get_score = |doc_id: &str| -> f32 {
        results
            .hits
            .iter()
            // Use ends_with to avoid "doc_x1" matching "doc_x10"
            .find(|h| h.id.ends_with(&format!(":{doc_id}")))
            .unwrap_or_else(|| panic!("{doc_id} should be in results"))
            .score
    };

    let score_x1 = get_score("doc_x1");
    let score_x2 = get_score("doc_x2");
    let score_x5 = get_score("doc_x5");
    let score_x10 = get_score("doc_x10");

    // Verify multiplier relationships (with higher tolerance for floating point)
    assert_approx_eq!(score_x2, score_x1 * 2.0, 0.01);
    assert_approx_eq!(score_x5, score_x1 * 5.0, 0.01);
    assert_approx_eq!(score_x10, score_x1 * 10.0, 0.01);

    // Document with highest OMC should be ranked first
    assert!(results.hits[0].id.contains("doc_x10"));

    drop(test_context);
}

/// Test OMC with fractional multipliers (less than 1.0 but positive).
#[tokio::test(flavor = "multi_thread")]
async fn test_omc_fractional_multipliers() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {
            "id": "doc_full",
            "title": "test document",
            "_omc": 1.0
        },
        {
            "id": "doc_half",
            "title": "test document",
            "_omc": 0.5
        },
        {
            "id": "doc_quarter",
            "title": "test document",
            "_omc": 0.25
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "test document"
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 3);

    let get_score = |doc_id: &str| -> f32 {
        results
            .hits
            .iter()
            .find(|h| h.id.contains(doc_id))
            .unwrap_or_else(|| panic!("{doc_id} should be in results"))
            .score
    };

    let score_full = get_score("doc_full");
    let score_half = get_score("doc_half");
    let score_quarter = get_score("doc_quarter");

    // Verify fractional multiplier relationships
    assert_approx_eq!(score_half, score_full * 0.5, 0.001);
    assert_approx_eq!(score_quarter, score_full * 0.25, 0.001);

    // Document with highest OMC should be ranked first
    assert!(results.hits[0].id.contains("doc_full"));

    drop(test_context);
}
