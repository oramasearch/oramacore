use crate::tests::utils::{init_log, wait_for, TestContext};
use anyhow::bail;
use futures::FutureExt;
use serde_json::json;
use std::convert::TryInto;

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_insert_simple() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert some documents that will be in the shelf
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

    // Create a shelf with specific document order
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "bestsellers",
                "documents": ["5", "3", "1", "7"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify the shelf was created
    let shelf = collection_client
        .get_shelf("bestsellers".to_string())
        .await
        .unwrap();

    assert_eq!(shelf.id.as_str(), "bestsellers");
    assert_eq!(shelf.documents, vec!["5", "3", "1", "7"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_update_existing() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Insert initial shelf
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "featured",
                "documents": ["1", "2", "3"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Update the shelf with new documents
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "featured",
                "documents": ["4", "5", "6", "7"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify the shelf was updated
    let shelf = collection_client
        .get_shelf("featured".to_string())
        .await
        .unwrap();

    assert_eq!(shelf.id.as_str(), "featured");
    assert_eq!(shelf.documents, vec!["4", "5", "6", "7"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_list_multiple() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Initially should have no shelves
    let shelves = collection_client.list_shelves().await.unwrap();
    assert_eq!(shelves.len(), 0);

    // Insert multiple shelves
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-1",
                "documents": ["1", "2"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-2",
                "documents": ["3", "4", "5"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-3",
                "documents": ["6"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify all shelves are listed
    let shelves = collection_client.list_shelves().await.unwrap();
    assert_eq!(shelves.len(), 3);

    let shelf_ids: Vec<String> = shelves.iter().map(|s| s.id.as_str().to_string()).collect();
    assert!(shelf_ids.contains(&"shelf-1".to_string()));
    assert!(shelf_ids.contains(&"shelf-2".to_string()));
    assert!(shelf_ids.contains(&"shelf-3".to_string()));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_delete() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Insert multiple shelves
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-to-keep",
                "documents": ["1", "2"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-to-delete",
                "documents": ["3", "4"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify both shelves exist
    let shelves = collection_client.list_shelves().await.unwrap();
    assert_eq!(shelves.len(), 2);

    // Delete one shelf
    collection_client
        .delete_shelf("shelf-to-delete".to_string())
        .await
        .unwrap();

    // Verify only one shelf remains
    let shelves = collection_client.list_shelves().await.unwrap();
    assert_eq!(shelves.len(), 1);
    assert_eq!(shelves[0].id.as_str(), "shelf-to-keep");

    // Verify getting the deleted shelf returns an error
    let result = collection_client
        .get_shelf("shelf-to-delete".to_string())
        .await;
    assert!(result.is_err());

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Insert a shelf
    const TEST_COMMIT_SHELF_ID: &str = "test-commit-shelf";
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": TEST_COMMIT_SHELF_ID,
                "documents": ["1", "2", "3", "4", "5"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Commit to ensure files are written to disk
    test_context.commit_all().await.unwrap();

    // Check that the shelf file exists in the writer side
    let writer_data_dir = &test_context.config.writer_side.config.data_dir;
    let writer_shelf_dir = writer_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("shelves");
    let writer_shelf_file = writer_shelf_dir.join(format!("{TEST_COMMIT_SHELF_ID}.shelf"));
    assert!(
        writer_shelf_file.exists(),
        "Shelf file should exist in writer side: {writer_shelf_file:?}"
    );

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;

    // Now reload the system to verify shelves are loaded from disk
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    // Wait for the writer to fully load the collection and shelves
    wait_for(&collection_client, |c| {
        let writer = c.writer;
        let write_api_key = c.write_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = writer.get_collection(collection_id, write_api_key).await?;
            let shelves = collection.list_shelves().await?;

            if !shelves
                .iter()
                .any(|s| s.id.as_str() == TEST_COMMIT_SHELF_ID)
            {
                bail!("{TEST_COMMIT_SHELF_ID} not found after reload");
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelves should be loaded from disk after reload");

    // Verify the shelf still exists and has correct data after reload
    let shelf_after_reload = collection_client
        .get_shelf(TEST_COMMIT_SHELF_ID.to_string())
        .await
        .unwrap();

    assert_eq!(shelf_after_reload.id.as_str(), TEST_COMMIT_SHELF_ID);
    assert_eq!(shelf_after_reload.documents, vec!["1", "2", "3", "4", "5"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_delete_removes_files() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Insert a shelf
    const TEST_DELETE_SHELF_ID: &str = "test-delete-shelf";
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": TEST_DELETE_SHELF_ID,
                "documents": ["1", "2", "3"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Commit to ensure files are written to disk
    test_context.commit_all().await.unwrap();

    // Verify that the shelf file exists in the writer side before deletion
    let writer_data_dir = &test_context.config.writer_side.config.data_dir;
    let writer_shelf_dir = writer_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("shelves");
    let writer_shelf_file = writer_shelf_dir.join(format!("{TEST_DELETE_SHELF_ID}.shelf"));
    assert!(
        writer_shelf_file.exists(),
        "Shelf file should exist in writer side before deletion: {writer_shelf_file:?}"
    );

    // Delete the shelf
    collection_client
        .delete_shelf(TEST_DELETE_SHELF_ID.to_string())
        .await
        .unwrap();

    // Commit to ensure deletion is persisted to disk
    test_context.commit_all().await.unwrap();

    // Verify that the shelf file is removed from the writer side
    assert!(
        !writer_shelf_file.exists(),
        "Shelf file should be removed from writer side: {writer_shelf_file:?}"
    );

    // Verify the shelf no longer exists
    let result = collection_client
        .get_shelf(TEST_DELETE_SHELF_ID.to_string())
        .await;
    assert!(result.is_err(), "Shelf should not exist after deletion");

    // Verify it's not in the list
    let shelves = collection_client.list_shelves().await.unwrap();
    assert!(
        !shelves
            .iter()
            .any(|s| s.id.as_str() == TEST_DELETE_SHELF_ID),
        "Deleted shelf should not be in list"
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_empty_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Create a shelf with no documents
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "empty-shelf",
                "documents": []
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify the shelf was created
    let shelf = collection_client
        .get_shelf("empty-shelf".to_string())
        .await
        .unwrap();

    assert_eq!(shelf.id.as_str(), "empty-shelf");
    assert_eq!(shelf.documents.len(), 0);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_multiple_collections() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client_1 = test_context.create_collection().await.unwrap();
    let collection_client_2 = test_context.create_collection().await.unwrap();

    // Create shelves in different collections
    collection_client_1
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-a",
                "documents": ["1", "2"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client_2
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-b",
                "documents": ["3", "4"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify each collection has only its own shelf
    let shelves_1 = collection_client_1.list_shelves().await.unwrap();
    assert_eq!(shelves_1.len(), 1);
    assert_eq!(shelves_1[0].id.as_str(), "shelf-a");

    let shelves_2 = collection_client_2.list_shelves().await.unwrap();
    assert_eq!(shelves_2.len(), 1);
    assert_eq!(shelves_2[0].id.as_str(), "shelf-b");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_large_document_list() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // should return an error, too many documents
    let large_doc_list: Vec<String> = (0..1001).map(|i| format!("doc-{i}")).collect();

    assert!(collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "large-shelf",
                "documents": large_doc_list
            }))
            .unwrap(),
        )
        .await
        .is_err());

    drop(test_context);
}
