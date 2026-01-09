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
                "doc_ids": ["5", "3", "1", "7"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Wait for the reader to receive the shelf message and verify it has the expected document IDs
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelf_id = oramacore_lib::shelves::ShelfId::try_new("bestsellers")?;
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
        .get_shelf("bestsellers".to_string())
        .await
        .unwrap();

    // Test also the conversion to document id
    assert_eq!(shelf_reader.id.as_str(), "bestsellers");
    assert_eq!(
        shelf_reader.doc_ids,
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

    let shelves_reader = collection_client.list_shelves().await.unwrap();
    assert_eq!(shelves_reader.len(), 0);

    // Insert multiple shelves
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-1",
                "doc_ids": ["1", "2"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-2",
                "doc_ids": ["3", "4", "5"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-3",
                "doc_ids": ["6"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    let expected_shelves = vec!["shelf-1", "shelf-2", "shelf-3"];

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
                bail!("expected 3 shelves in reader, found {}", shelves.len());
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
                "doc_ids": ["1", "2"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-to-delete",
                "doc_ids": ["3", "4"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    // Verify both shelves exist
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelves = collection.list_shelves().await?;

            if shelves.len() != 2 {
                bail!("expected 2 shelves in reader, found {}", shelves.len());
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("All shelves should be listed in reader");

    // Delete one shelf
    collection_client
        .delete_shelf("shelf-to-delete".to_string())
        .await
        .unwrap();

    // Verify only one shelf remains
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelves = collection.list_shelves().await?;

            if shelves.len() != 1 {
                bail!("expected 1 shelves in reader, found {}", shelves.len());
            }

            if shelves.first().unwrap().id.as_str() != "shelf-to-keep" {
                bail!(
                    "expected 'shelf-to-keep' to be in reader, found {}",
                    shelves.first().unwrap().id.as_str()
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("All shelves should be listed in reader");

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

    test_context.commit_all().await.unwrap();

    const TEST_COMMIT_SHELF_ID: &str = "test-commit-shelf";
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": TEST_COMMIT_SHELF_ID,
                "doc_ids": ["1", "2", "3", "4", "5"]
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

    let shelf_after_reload_reader = collection_client
        .get_shelf(TEST_COMMIT_SHELF_ID.to_string())
        .await
        .unwrap();

    assert_eq!(shelf_after_reload_reader.id.as_str(), TEST_COMMIT_SHELF_ID);
    assert_eq!(shelf_after_reload_reader.doc_ids.len(), 5);

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
                "doc_ids": ["1", "2", "3"]
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

    let result_reader = collection_client
        .get_shelf(TEST_DELETE_SHELF_ID.to_string())
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
                "doc_ids": ["1", "2", "3"]
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
            let shelf_id = oramacore_lib::shelves::ShelfId::try_new(TEST_UPDATE_SHELF_ID)?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.doc_ids.len() != 3 {
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
                "doc_ids": ["4", "5"]
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
            let shelf_id = oramacore_lib::shelves::ShelfId::try_new(TEST_UPDATE_SHELF_ID)?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.doc_ids.len() != 2 {
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
        .get_shelf(TEST_UPDATE_SHELF_ID.to_string())
        .await
        .unwrap();
    assert_eq!(shelf_reader_updated.id.as_str(), TEST_UPDATE_SHELF_ID);
    assert_eq!(shelf_reader_updated.doc_ids.len(), 2);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shelf_empty_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Create a shelf with no doc_ids
    collection_client
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "empty-shelf",
                "doc_ids": []
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelves = collection.list_shelves().await.unwrap();

            if shelves.len() != 1 {
                bail!("Expected 1 shelf, got {}", shelves.len());
            }

            if shelves[0].id.as_str() != "empty-shelf" {
                bail!(
                    "Expected shelf 'empty-shelf', got {}",
                    shelves[0].id.as_str()
                );
            }

            if !shelves[0].doc_ids.is_empty() {
                bail!(
                    "Expected shelf to be empty, got {}",
                    shelves[0].doc_ids.len()
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf should be available in reader");

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
                "doc_ids": ["1", "2"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client_2
        .insert_shelf(
            serde_json::from_value(json!({
                "id": "shelf-b",
                "doc_ids": ["3", "4"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();

    wait_for(&collection_client_1, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelves = collection.list_shelves().await.unwrap();

            if shelves.len() != 1 {
                bail!("Expected 1 shelf, got {}", shelves.len());
            }

            if shelves[0].id.as_str() != "shelf-a" {
                bail!("Expected shelf 'shelf-a', got {}", shelves[0].id.as_str());
            }

            if shelves[0].doc_ids.len() != 2 {
                bail!(
                    "Expected shelf to be of len 2, got {}",
                    shelves[0].doc_ids.len()
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf-a should be available in reader");

    wait_for(&collection_client_2, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key;
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, read_api_key).await?;
            let shelves = collection.list_shelves().await.unwrap();

            if shelves.len() != 1 {
                bail!("Expected 1 shelf, got {}", shelves.len());
            }

            if shelves[0].id.as_str() != "shelf-b" {
                bail!("Expected shelf 'shelf-b', got {}", shelves[0].id.as_str());
            }
            if shelves[0].doc_ids.len() != 2 {
                bail!(
                    "Expected shelf to be of len 2, got {}",
                    shelves[0].doc_ids.len()
                );
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf-b should be available in reader");

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
                "doc_ids": large_doc_list
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
                "doc_ids": ["doc-1", "doc-3"]
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
            let shelf_id = oramacore_lib::shelves::ShelfId::try_new("featured")?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.doc_ids.len() != 2 {
                bail!("Expected 2 doc_ids in shelf, got {}", shelf.doc_ids.len());
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Shelf should be available in reader");

    let shelf_before = collection_client
        .get_shelf("featured".to_string())
        .await
        .unwrap();
    let doc_ids_before = shelf_before.doc_ids.clone();

    // Now update one of the documents that's in the shelf, changing its ID
    index_client
        .update_documents(
            serde_json::from_value(json!({
                "strategy": "merge",
                "doc_ids": [
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
            let shelf_id = oramacore_lib::shelves::ShelfId::try_new("featured")?;
            let shelf = collection.get_shelf(shelf_id).await?;

            if shelf.doc_ids.len() != 2 {
                bail!(
                    "Expected 2 doc_ids in shelf after update, got {}",
                    shelf.doc_ids.len()
                );
            }

            if shelf.doc_ids[0] != expected_doc_ids[0] {
                bail!(
                    "First document ID changed unexpectedly: {:?} != {:?}",
                    shelf.doc_ids[0],
                    expected_doc_ids[0]
                );
            }

            // Check if the document IDs have been updated
            // The second document should have changed its internal ID but still be referenced
            // because the document was replaced with a new one
            if shelf.doc_ids[1] == expected_doc_ids[1] {
                bail!(
                    "Second document ID should have changed after update: {:?}",
                    shelf.doc_ids[1]
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
