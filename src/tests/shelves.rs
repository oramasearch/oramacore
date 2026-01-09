use crate::tests::utils::{init_log, wait_for, TestContext};
use crate::types::DocumentId;
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

    // Verify the shelf was created in writer
    let shelf = collection_client
        .get_shelf_from_writer("bestsellers".to_string())
        .await
        .unwrap();

    assert_eq!(shelf.id.as_str(), "bestsellers");
    assert_eq!(shelf.documents, vec!["5", "3", "1", "7"]);

    // Wait for the reader to receive the shelf message and verify it has the expected document IDs
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelf_id = oramacore_lib::shelf::ShelfId::try_new("bestsellers")?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.id.as_str() != "bestsellers" {
                bail!(
                    "Shelf ID mismatch in reader: expected 'bestsellers', got '{}'",
                    shelf.id
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf should be available in reader with correct document IDs");

    // Verify the shelf from reader using the helper method
    let shelf_reader = collection_client
        .get_shelf_from_reader("bestsellers".to_string())
        .await
        .unwrap();

    // Test also the conversion to document id
    assert_eq!(shelf_reader.id.as_str(), "bestsellers");
    assert_eq!(
        shelf_reader.documents,
        vec![DocumentId(6), DocumentId(4), DocumentId(2), DocumentId(8)]
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_list_multiple() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Initially should have no shelves
    let shelves = collection_client.list_shelves_from_writer().await.unwrap();
    assert_eq!(shelves.len(), 0);

    let shelves_reader = collection_client.list_shelves_from_reader().await.unwrap();
    assert_eq!(shelves_reader.len(), 0);

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

    let expected_shelves = vec!["shelf-1", "shelf-2", "shelf-3"];

    // Verify all shelves are listed in writer
    wait_for(&collection_client, |c| {
        let writer = c.writer;
        let write_api_key = c.write_api_key;
        let collection_id = c.collection_id;
        let expected_shelves = expected_shelves.clone();
        async move {
            let collection = writer.get_collection(collection_id, write_api_key).await?;
            let shelves = collection.list_shelves().await?;

            if shelves.len() != 3 {
                bail!("Expected 3 shelves in writer, found {}", shelves.len());
            }

            let shelf_ids: Vec<&str> = shelves.iter().map(|s| s.id.as_str()).collect();
            for expected in expected_shelves {
                if !shelf_ids.contains(&expected) {
                    bail!("{expected} not found in writer");
                }
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("All shelves should be listed in writer");

    // Verify all shelves are listed in reader
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        let expected_shelves = expected_shelves.clone();
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelves = collection.list_shelves().await?;

            if shelves.len() != 3 {
                bail!("Expected 3 shelves in reader, found {}", shelves.len());
            }

            let shelf_ids: Vec<&str> = shelves.iter().map(|s| s.id.as_str()).collect();
            for expected in expected_shelves {
                if !shelf_ids.contains(&expected) {
                    bail!("{expected} not found in reader");
                }
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("All shelves should be listed in reader");

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
    let shelves = collection_client.list_shelves_from_writer().await.unwrap();
    assert_eq!(shelves.len(), 2);

    // Delete one shelf
    collection_client
        .delete_shelf("shelf-to-delete".to_string())
        .await
        .unwrap();

    // Verify only one shelf remains
    let shelves = collection_client.list_shelves_from_writer().await.unwrap();
    assert_eq!(shelves.len(), 1);
    assert_eq!(shelves[0].id.as_str(), "shelf-to-keep");

    // Verify getting the deleted shelf returns an error
    let result = collection_client
        .get_shelf_from_writer("shelf-to-delete".to_string())
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

    test_context.commit_all().await.unwrap();

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

    // Check that the shelf file exists
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

    let reader_data_dir = &test_context.config.reader_side.config.data_dir;
    let reader_shelf_dir = reader_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("shelves");
    let reader_shelf_file = reader_shelf_dir.join(format!("{TEST_COMMIT_SHELF_ID}.shelf"));
    assert!(
        reader_shelf_file.exists(),
        "Shelf file should exist in reader side: {reader_shelf_file:?}"
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
                bail!("{TEST_COMMIT_SHELF_ID} not found after reload in writer");
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelves should be loaded from disk after reload in writer");

    // Wait for the reader to fully load the collection and shelves
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelves = collection.list_shelves().await?;

            if !shelves
                .iter()
                .any(|s| s.id.as_str() == TEST_COMMIT_SHELF_ID)
            {
                bail!("{TEST_COMMIT_SHELF_ID} not found after reload in reader");
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelves should be loaded from disk after reload in reader");

    // Verify the shelf still exists and has correct data after reload
    let shelf_after_reload_writer = collection_client
        .get_shelf_from_writer(TEST_COMMIT_SHELF_ID.to_string())
        .await
        .unwrap();

    assert_eq!(shelf_after_reload_writer.id.as_str(), TEST_COMMIT_SHELF_ID);
    assert_eq!(shelf_after_reload_writer.documents.len(), 5);

    let shelf_after_reload_reader = collection_client
        .get_shelf_from_reader(TEST_COMMIT_SHELF_ID.to_string())
        .await
        .unwrap();

    assert_eq!(shelf_after_reload_reader.id.as_str(), TEST_COMMIT_SHELF_ID);
    assert_eq!(shelf_after_reload_reader.documents.len(), 5);

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

    let reader_data_dir = &test_context.config.reader_side.config.data_dir;
    let reader_shelf_dir = reader_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("shelves");
    let reader_shelf_file = reader_shelf_dir.join(format!("{TEST_DELETE_SHELF_ID}.shelf"));
    assert!(
        reader_shelf_file.exists(),
        "Shelf file should exists in reader side before deletion: {reader_shelf_file:?}"
    );

    collection_client
        .delete_shelf(TEST_DELETE_SHELF_ID.to_string())
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Verify that the shelf file is removed
    assert!(
        !writer_shelf_file.exists(),
        "Shelf file should be removed from writer side: {writer_shelf_file:?}"
    );
    assert!(
        !reader_shelf_file.exists(),
        "Shelf file should be removed from reader side: {reader_shelf_file:?}"
    );

    // Verify the shelf no longer exists
    let result = collection_client
        .get_shelf_from_writer(TEST_DELETE_SHELF_ID.to_string())
        .await;
    assert!(
        result.is_err(),
        "Shelf should not exist after deletion in writer"
    );

    let result_reader = collection_client
        .get_shelf_from_reader(TEST_DELETE_SHELF_ID.to_string())
        .await;
    assert!(
        result_reader.is_err(),
        "Shelf should not exist after deletion in reader"
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_update() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    const TEST_UPDATE_SHELF_ID: &str = "test-update-shelf";
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": TEST_UPDATE_SHELF_ID,
                "documents": ["1", "2", "3"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify shelf from writer
    let shelf_writer = collection_client
        .get_shelf_from_writer(TEST_UPDATE_SHELF_ID.to_string())
        .await
        .unwrap();
    assert_eq!(shelf_writer.id.as_str(), TEST_UPDATE_SHELF_ID);
    assert_eq!(shelf_writer.documents, vec!["1", "2", "3"]);

    // Wait for reader to update
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelf_id = oramacore_lib::shelf::ShelfId::try_new(TEST_UPDATE_SHELF_ID)?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.documents.len() != 3 {
                bail!("Shelf not updated in reader yet");
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf should be updated in reader");

    // Update shelf
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": TEST_UPDATE_SHELF_ID,
                "documents": ["4", "5"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Wait for reader to update
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelf_id = oramacore_lib::shelf::ShelfId::try_new(TEST_UPDATE_SHELF_ID)?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.documents.len() != 2 {
                bail!("Shelf not updated in reader yet");
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf should be updated in reader after commit");

    // Verify updated shelf from reader
    let shelf_reader_updated = collection_client
        .get_shelf_from_reader(TEST_UPDATE_SHELF_ID.to_string())
        .await
        .unwrap();
    assert_eq!(shelf_reader_updated.id.as_str(), TEST_UPDATE_SHELF_ID);
    assert_eq!(shelf_reader_updated.documents.len(), 2);

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
        .get_shelf_from_writer("empty-shelf".to_string())
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
    let shelves_1 = collection_client_1
        .list_shelves_from_writer()
        .await
        .unwrap();
    assert_eq!(shelves_1.len(), 1);
    assert_eq!(shelves_1[0].id.as_str(), "shelf-a");

    let shelves_2 = collection_client_2
        .list_shelves_from_writer()
        .await
        .unwrap();
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

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_updated_when_document_id_changes() {
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

    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "featured",
                "documents": ["doc-1", "doc-3"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify initial shelf state in reader
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelf_id = oramacore_lib::shelf::ShelfId::try_new("featured")?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.documents.len() != 2 {
                bail!(
                    "Expected 2 documents in shelf, got {}",
                    shelf.documents.len()
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf should be available in reader");

    let shelf_before = collection_client
        .get_shelf_from_reader("featured".to_string())
        .await
        .unwrap();
    let doc_ids_before = shelf_before.documents.clone();

    // Now update one of the documents that's in the shelf, changing its ID
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

    // Wait for the shelf to be updated in the reader
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        let expected_doc_ids = doc_ids_before.clone();
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelf_id = oramacore_lib::shelf::ShelfId::try_new("featured")?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.documents.len() != 2 {
                bail!(
                    "Expected 2 documents in shelf after update, got {}",
                    shelf.documents.len()
                );
            }

            if shelf.documents[0] != expected_doc_ids[0] {
                bail!(
                    "First document ID changed unexpectedly: {:?} != {:?}",
                    shelf.documents[0],
                    expected_doc_ids[0]
                );
            }

            // Check if the document IDs have been updated
            // The second document should have changed its internal ID but still be referenced
            // because the document was replaced with a new one
            if shelf.documents[1] == expected_doc_ids[1] {
                bail!(
                    "Second document ID should have changed after update: {:?}",
                    shelf.documents[1]
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf should be updated after document ID change");

    drop(test_context);
}
