use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::SearchParams;

#[tokio::test(flavor = "multi_thread")]
async fn test_geosearch() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "T",
                    "location": {
                        "lat": 9.0814233,
                        "lon": 45.2623823,
                    },
                },
                {
                    "id": "2",
                    "name": "T",
                    "location": {
                        "lat": 9.0979028,
                        "lon": 45.1995182,
                    },
                },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let p = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": {
                        "lat": 9.1418481,
                        "lon": 45.2324096
                    },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            },
        },
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p).await.unwrap();

    assert_eq!(output.count, 2);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_geosearch_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "T",
                    "location": {
                        "lat": 9.0814233,
                        "lon": 45.2623823,
                    },
                },
                {
                    "id": "2",
                    "name": "T",
                    "location": {
                        "lat": 9.0979028,
                        "lon": 45.1995182,
                    },
                },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let p: SearchParams = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": {
                        "lat": 9.1418481,
                        "lon": 45.2324096
                    },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            },
        },
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p.clone()).await.unwrap();
    assert_eq!(output.count, 2);

    test_context.commit_all().await.unwrap();

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;

    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let output = collection_client.search(p).await.unwrap();
    assert_eq!(output.count, 2);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_add_delete_search_no_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Add two documents
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "A",
                    "location": { "lat": 10.0, "lon": 20.0 }
                },
                {
                    "id": "2",
                    "name": "B",
                    "location": { "lat": 10.1, "lon": 20.1 }
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Delete one document
    index_client
        .delete_documents(vec!["1".to_string()])
        .await
        .unwrap();

    // Search: should only find the second document
    let p = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": { "lat": 10.05, "lon": 20.05 },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            }
        }
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p).await.unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(output.hits[0].id, format!("{}:2", index_client.index_id));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_add_delete_commit_reload_search() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Add two documents
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "A",
                    "location": { "lat": 10.0, "lon": 20.0 }
                },
                {
                    "id": "2",
                    "name": "B",
                    "location": { "lat": 10.1, "lon": 20.1 }
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Delete one document
    index_client
        .delete_documents(vec!["1".to_string()])
        .await
        .unwrap();

    // Commit changes
    test_context.commit_all().await.unwrap();

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;
    let index_id = index_client.index_id;

    // Reload context
    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    // Search: should only find the second document
    let p = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": { "lat": 10.05, "lon": 20.05 },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            }
        }
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p).await.unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(output.hits[0].id, format!("{index_id}:2"));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_add_commit_delete_search_no_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Add two documents and commit
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "A",
                    "location": { "lat": 10.0, "lon": 20.0 }
                },
                {
                    "id": "2",
                    "name": "B",
                    "location": { "lat": 10.1, "lon": 20.1 }
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    test_context.commit_all().await.unwrap();

    // Delete one document (no commit)
    index_client
        .delete_documents(vec!["1".to_string()])
        .await
        .unwrap();

    // Search: should only find the second document
    let p = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": { "lat": 10.05, "lon": 20.05 },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            }
        }
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p).await.unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(output.hits[0].id, format!("{}:2", index_client.index_id));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_add_commit_delete_commit_reload_search() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Add two documents and commit
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "A",
                    "location": { "lat": 10.0, "lon": 20.0 }
                },
                {
                    "id": "2",
                    "name": "B",
                    "location": { "lat": 10.1, "lon": 20.1 }
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    test_context.commit_all().await.unwrap();

    // Delete one document and commit
    index_client
        .delete_documents(vec!["1".to_string()])
        .await
        .unwrap();
    test_context.commit_all().await.unwrap();

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;
    let index_id = index_client.index_id;

    // Reload context
    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    // Search: should only find the second document
    let p = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": { "lat": 10.05, "lon": 20.05 },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            }
        }
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p).await.unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(output.hits[0].id, format!("{index_id}:2"));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_add_delete_add_again_search() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Add a document
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "location": { "lat": 10.0, "lon": 20.0 }
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Delete the document
    index_client
        .delete_documents(vec!["1".to_string()])
        .await
        .unwrap();

    // Add the same document again
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "location": { "lat": 10.0, "lon": 20.0 }
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Search: should find the document
    let p = json!({
        "term": "Tommaso",
        "where": {
            "location": {
                "radius": {
                    "coordinates": { "lat": 10.0, "lon": 20.0 },
                    "unit": "km",
                    "value": 1,
                    "inside": true,
                }
            }
        }
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p).await.unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(output.hits[0].id, format!("{}:1", index_client.index_id));

    drop(test_context);
}
