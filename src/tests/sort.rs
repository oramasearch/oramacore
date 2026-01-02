use serde_json::json;

use crate::collection_manager::sides::read::ReadError;
use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_number() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "year": 1990,
                },
                {
                    "id": "2",
                    "name": "Michele",
                    "year": 1994,
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "year",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "year",
                    "order": "ASC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "year",
                    "order": "DESC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_date() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    // The date is not correct, otherwise we leak personal data
                    "birthday": "1990-01-01T00:00:00Z",
                },
                {
                    "id": "2",
                    "name": "Michele",
                    // The date is not correct, otherwise we leak personal data
                    "birthday": "1994-01-01T00:00:00Z",
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "birthday",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "birthday",
                    "order": "ASC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "birthday",
                    "order": "DESC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_bool() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "is_good": false,
                },
                {
                    "id": "2",
                    "name": "Michele",
                    "is_good": true,
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "is_good",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "is_good",
                    "order": "ASC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "is_good",
                    "order": "DESC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_unknown_field() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    collection_client.create_index().await.unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "unknown_field",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await;

    let ReadError::SortFieldNotFound(k) = output.unwrap_err() else {
        panic!("Expected ReadError::SortFieldNotFound");
    };

    assert_eq!(k, "unknown_field",);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_unsupported_field() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "position": {
                        "lat": 45.0,
                        "lon": 7.0,
                    }
                },
                {
                    "id": "2",
                    "name": "Michele",
                    "position": {
                        "lat": 46.0,
                        "lon": 8.0,
                    }
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "position",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await;

    println!("Output: {output:?}");

    let ReadError::InvalidSortField(k, t) = output.unwrap_err() else {
        panic!("Expected ReadError::InvalidSortField");
    };

    assert_eq!(k, "position",);
    assert_eq!(t, "GeoPoint",);

    drop(test_context);
}

/// Test that filtering works correctly when combined with sorting
/// when documents share the same sort key value.
/// This tests the interaction between MergeSortedIterator's batch-level filtering
/// and the document-level filtering in truncate().
#[tokio::test(flavor = "multi_thread")]
async fn test_sort_with_filter_same_key_value() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert 3 documents with the same number value but different boolean values
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Document One",
                    "number": 2,
                    "is_active": true,
                },
                {
                    "id": "2",
                    "name": "Document Two",
                    "number": 2,
                    "is_active": false,
                },
                {
                    "id": "3",
                    "name": "Document Three",
                    "number": 2,
                    "is_active": true,
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Search with sorting by number and filtering by is_active = true
    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "number",
                    "order": "ASC"
                },
                "where": {
                    "is_active": true
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Should return only documents with is_active = true (doc 1 and 3)
    assert_eq!(output.count, 2, "Expected 2 documents matching the filter");
    assert_eq!(output.hits.len(), 2, "Expected 2 hits");

    // Verify that document 2 (is_active = false) is NOT in the results
    let hit_ids: Vec<String> = output.hits.iter().map(|h| h.id.clone()).collect();
    assert!(
        hit_ids.contains(&format!("{}:{}", index_client.index_id, "1")),
        "Document 1 should be in results"
    );
    assert!(
        hit_ids.contains(&format!("{}:{}", index_client.index_id, "3")),
        "Document 3 should be in results"
    );
    assert!(
        !hit_ids.contains(&format!("{}:{}", index_client.index_id, "2")),
        "Document 2 should NOT be in results"
    );

    drop(test_context);
}
