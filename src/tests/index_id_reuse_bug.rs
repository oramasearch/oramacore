use serde_json::json;
use std::fs;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::{CreateIndexRequest, DocumentList, IndexId};

#[tokio::test(flavor = "multi_thread")]
async fn test_index_id_reuse_filesystem_dirty_bug() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    // Step 1: Create index with specific ID and insert documents
    let index_id = IndexId::try_new("test-index-reuse".to_string()).unwrap();
    
    // Create index with the specific ID
    test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id,
                embedding: None,
            },
        )
        .await
        .unwrap();

    let index_client = collection_client
        .get_test_index_client(index_id)
        .unwrap();

    // Insert some documents to create filesystem data
    let original_documents: DocumentList = json!([
        {
            "id": "doc1",
            "title": "Original Document 1",
            "content": "Original content that should be deleted"
        },
        {
            "id": "doc2", 
            "title": "Original Document 2",
            "content": "More original content that should be deleted"
        }
    ])
    .try_into()
    .unwrap();

    index_client
        .insert_documents(original_documents)
        .await
        .unwrap();

    // Verify original documents exist
    let search_result = collection_client
        .search(
            json!({
                "term": "Original",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(search_result.count, 2, "Should find 2 original documents");

    // Step 2: Delete the index 
    // This should mark as deleted but NOT immediately clean filesystem
    index_client.delete().await.unwrap();

    // Verify index is deleted
    let search_result = collection_client
        .search(
            json!({
                "term": "Original",
            })
            .try_into()
            .unwrap(),
        )
        .await;
    
    // Search should fail or return no results since index is deleted
    assert!(search_result.is_err() || search_result.unwrap().count == 0);

    // Step 3: Immediately create new index with same ID (before commit cycle runs)
    // This is where the filesystem dirty state issue occurs
    test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id, // Same ID as deleted index!
                embedding: None,
            },
        )
        .await
        .unwrap();

    let new_index_client = collection_client
        .get_test_index_client(index_id)
        .unwrap();

    // Step 4: Insert completely different documents to new index
    let new_documents: DocumentList = json!([
        {
            "id": "new-doc1",
            "title": "New Document 1", 
            "content": "Fresh new content"
        }
    ])
    .try_into()
    .unwrap();

    new_index_client
        .insert_documents(new_documents)
        .await
        .unwrap();

    // Step 5: This should expose the filesystem dirty state issue
    // Expected: Only find 1 new document
    // Actual: Filesystem likely contains mix of old deleted data and new data
    let search_result = collection_client
        .search(
            json!({
                "term": "New",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // This assertion should PASS for the new document
    assert_eq!(search_result.count, 1, "Should find exactly 1 new document");

    // This is the critical test - the old deleted documents should NOT be findable
    // But due to dirty filesystem state, they might still be there
    let old_search_result = collection_client
        .search(
            json!({
                "term": "Original", 
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // THIS IS WHERE THE TEST SHOULD FAIL
    // The filesystem is dirty - old deleted index data is still present
    // Even though we created a "new" index with same ID
    assert_eq!(
        old_search_result.count, 0,
        "FILESYSTEM BUG: Old deleted documents still found! Filesystem is dirty from previous deleted index with same ID. Expected 0 but found {}",
        old_search_result.count
    );

    // Also verify total document count is correct
    let all_search_result = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        all_search_result.count, 1,
        "FILESYSTEM BUG: Total document count wrong! Expected 1 new document but filesystem has dirty state with {} documents", 
        all_search_result.count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_id_reuse_filesystem_inspection() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    // Step 1: Create index with specific ID 
    let index_id = IndexId::try_new("filesystem-inspect-test".to_string()).unwrap();
    
    test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id,
                embedding: None,
            },
        )
        .await
        .unwrap();

    let index_client = collection_client
        .get_test_index_client(index_id)
        .unwrap();

    // Insert documents to create filesystem data
    let documents: DocumentList = json!([
        {
            "id": "fs-doc1",
            "title": "Filesystem Test Doc",
            "content": "This should create filesystem data"
        }
    ])
    .try_into()
    .unwrap();

    index_client.insert_documents(documents).await.unwrap();

    // Force a commit to make sure filesystem data exists
    test_context.commit_all().await.unwrap();

    // Get the filesystem path where the index should be stored
    let data_dir = &test_context.config.reader_side.config.data_dir;
    let collection_dir = data_dir
        .join("collections")
        .join(collection_client.collection_id.as_str());
    let index_dir = collection_dir.join("indexes").join(index_id.as_str());

    // Verify filesystem data exists
    println!("Checking filesystem at: {:?}", index_dir);
    assert!(
        index_dir.exists(),
        "Index directory should exist after commit: {:?}",
        index_dir
    );

    // Check what files are in the index directory
    if let Ok(entries) = fs::read_dir(&index_dir) {
        println!("Files in index directory:");
        for entry in entries {
            if let Ok(entry) = entry {
                println!("  {:?}", entry.file_name());
            }
        }
    }

    // Step 2: Delete the index (this should only mark as deleted)
    index_client.delete().await.unwrap();

    // DON'T commit! This is key - we want to see the filesystem while it's in dirty state
    
    // Step 3: Check if filesystem data is still there (it should be!)
    println!("After delete (no commit) - checking filesystem at: {:?}", index_dir);
    let filesystem_exists_after_delete = index_dir.exists();
    println!("Filesystem still exists after delete: {}", filesystem_exists_after_delete);

    // This demonstrates the issue - filesystem data remains after index deletion
    assert!(
        filesystem_exists_after_delete,
        "EVIDENCE OF BUG: Filesystem should still exist after delete (before commit), but it was cleaned up"
    );

    // Step 4: Try to create new index with same ID
    let create_result = test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id, // Same ID!
                embedding: None,
            },
        )
        .await;

    println!("Create result with same ID: {:?}", create_result);

    // This might fail or succeed depending on how the system handles it
    match create_result {
        Ok(_) => {
            println!("SUCCESS: New index created with same ID");
            
            // If it succeeded, the new index is now using potentially dirty filesystem
            let new_index_client = collection_client
                .get_test_index_client(index_id)
                .unwrap();

            // Try to insert new data
            let new_documents: DocumentList = json!([
                {
                    "id": "new-fs-doc1",
                    "title": "New Filesystem Test Doc",
                    "content": "Fresh data for new index"
                }
            ])
            .try_into()
            .unwrap();

            new_index_client.insert_documents(new_documents).await.unwrap();

            // Now search to see if we get mixed results
            let search_result = collection_client
                .search(
                    json!({
                        "term": "Filesystem",
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();

            println!("Search results after reusing index ID: {:?}", search_result);
            
            // THIS IS THE CRITICAL TEST:
            // If the filesystem was dirty, we might see data from both old and new index
            // Or we might see inconsistent behavior
            if search_result.count > 1 {
                panic!(
                    "FILESYSTEM BUG DETECTED: Found {} documents when searching 'Filesystem'. \
                    Expected only 1 (new document), but found old data still present. \
                    This indicates dirty filesystem state from reused index ID.",
                    search_result.count
                );
            }
        }
        Err(e) => {
            println!("FAILED: Could not create new index with same ID: {:?}", e);
            // This could also be evidence of the bug - the system detecting a conflict
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_directory_cleanup_timing() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_id = IndexId::try_new("cleanup-timing-test".to_string()).unwrap();
    
    // Create and populate index
    test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id,
                embedding: None,
            },
        )
        .await
        .unwrap();

    let index_client = collection_client.get_test_index_client(index_id).unwrap();
    
    let documents: DocumentList = json!([
        {"id": "timing1", "content": "timing test document"}
    ])
    .try_into()
    .unwrap();

    index_client.insert_documents(documents).await.unwrap();
    
    // Commit to create filesystem data
    test_context.commit_all().await.unwrap();

    let data_dir = &test_context.config.reader_side.config.data_dir;
    let index_dir = data_dir
        .join("collections")
        .join(collection_client.collection_id.as_str())
        .join("indexes")
        .join(index_id.as_str());

    assert!(index_dir.exists(), "Index directory should exist after commit");

    // Delete index
    index_client.delete().await.unwrap();

    // Check filesystem immediately after delete (should still exist)
    let exists_after_delete = index_dir.exists();
    println!("Directory exists after delete (before commit): {}", exists_after_delete);

    // Now commit and see when cleanup happens
    test_context.commit_all().await.unwrap();

    let exists_after_commit = index_dir.exists();
    println!("Directory exists after commit: {}", exists_after_commit);

    // This shows the cleanup timing - filesystem is only cleaned during commit
    if exists_after_delete && !exists_after_commit {
        println!("CONFIRMED: Filesystem cleanup happens during commit, not during delete");
    } else if exists_after_delete && exists_after_commit {
        println!("POTENTIAL BUG: Directory still exists even after commit!");
        panic!("Directory should be cleaned up after commit but still exists");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_filesystem_access_bug() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_id = IndexId::try_new("concurrent-access-test".to_string()).unwrap();
    
    // Step 1: Create first index and add data
    test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id,
                embedding: None,
            },
        )
        .await
        .unwrap();

    let first_index = collection_client.get_test_index_client(index_id).unwrap();
    
    // Add data to first index and commit to create filesystem state
    let first_docs: DocumentList = json!([
        {"id": "old1", "content": "old document 1"},
        {"id": "old2", "content": "old document 2"}
    ]).try_into().unwrap();

    first_index.insert_documents(first_docs).await.unwrap();
    test_context.commit_all().await.unwrap();

    // Verify filesystem data exists
    let data_dir = &test_context.config.reader_side.config.data_dir;
    let index_dir = data_dir
        .join("collections")
        .join(collection_client.collection_id.as_str())
        .join("indexes")
        .join(index_id.as_str());

    assert!(index_dir.exists());
    
    // Check filesystem content before deletion
    if let Ok(entries) = fs::read_dir(&index_dir) {
        println!("Files before delete:");
        for entry in entries {
            if let Ok(entry) = entry {
                println!("  {:?}", entry.file_name());
            }
        }
    }

    // Step 2: Delete the index (filesystem should remain dirty)
    first_index.delete().await.unwrap();
    
    // Verify filesystem still exists (dirty state)
    assert!(index_dir.exists(), "Filesystem should still exist after delete");

    // Step 3: Immediately create new index with same ID (before any commit)
    test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id, // SAME ID!
                embedding: None,
            },
        )
        .await
        .unwrap();

    let second_index = collection_client.get_test_index_client(index_id).unwrap();

    // Step 4: Add different data to new index
    let second_docs: DocumentList = json!([
        {"id": "new1", "content": "new document 1"}
    ]).try_into().unwrap();

    second_index.insert_documents(second_docs).await.unwrap();

    // Step 5: Now commit - this should trigger cleanup of old index AND create new index data
    // This is where filesystem conflicts might occur
    test_context.commit_all().await.unwrap();

    // Check filesystem content after commit
    if let Ok(entries) = fs::read_dir(&index_dir) {
        println!("Files after commit with new index:");
        for entry in entries {
            if let Ok(entry) = entry {
                println!("  {:?}", entry.file_name());
            }
        }
    }

    // Step 6: Test search behavior - this is where inconsistencies might show up
    let search_old = collection_client
        .search(
            json!({"term": "old"}).try_into().unwrap(),
        )
        .await
        .unwrap();

    let search_new = collection_client
        .search(
            json!({"term": "new"}).try_into().unwrap(),
        )
        .await
        .unwrap();

    let search_all = collection_client
        .search(
            json!({"term": "*"}).try_into().unwrap(),
        )
        .await
        .unwrap();

    println!("Search results:");
    println!("  Old documents: {}", search_old.count);
    println!("  New documents: {}", search_new.count);
    println!("  All documents: {}", search_all.count);

    // The critical assertions:
    assert_eq!(search_old.count, 0, "Should not find old documents after index recreation");
    assert_eq!(search_new.count, 1, "Should find exactly 1 new document");
    assert_eq!(search_all.count, 1, "Total should be exactly 1 document");

    // If any of these fail, it indicates filesystem state corruption from reusing index ID
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filesystem_pollution_detection() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_id = IndexId::try_new("pollution-test".to_string()).unwrap();
    
    // Create, populate, and delete first index
    test_context.writer.create_index(
        collection_client.write_api_key,
        collection_client.collection_id,
        CreateIndexRequest { index_id, embedding: None },
    ).await.unwrap();

    let first_index = collection_client.get_test_index_client(index_id).unwrap();
    let docs: DocumentList = json!([{"id": "doc1", "data": "first index data"}]).try_into().unwrap();
    first_index.insert_documents(docs).await.unwrap();
    test_context.commit_all().await.unwrap();

    // Get filesystem info
    let data_dir = &test_context.config.reader_side.config.data_dir;
    let index_dir = data_dir
        .join("collections")
        .join(collection_client.collection_id.as_str())
        .join("indexes")
        .join(index_id.as_str());

    // Record filesystem state before deletion
    let files_before_delete: Vec<_> = if let Ok(entries) = fs::read_dir(&index_dir) {
        entries.filter_map(|e| e.ok()).map(|e| e.file_name()).collect()
    } else {
        vec![]
    };

    println!("Files before delete: {:?}", files_before_delete);

    // Delete index
    first_index.delete().await.unwrap();

    // Check files still exist (dirty state)
    let files_after_delete: Vec<_> = if let Ok(entries) = fs::read_dir(&index_dir) {
        entries.filter_map(|e| e.ok()).map(|e| e.file_name()).collect()
    } else {
        vec![]
    };

    println!("Files after delete (before commit): {:?}", files_after_delete);

    // Files should still be there
    assert!(!files_after_delete.is_empty(), "Files should still exist after delete");
    assert_eq!(files_before_delete, files_after_delete, "File list should be unchanged after delete");

    // Create new index with same ID (filesystem is dirty!)
    test_context.writer.create_index(
        collection_client.write_api_key,
        collection_client.collection_id,
        CreateIndexRequest { index_id, embedding: None },
    ).await.unwrap();

    let second_index = collection_client.get_test_index_client(index_id).unwrap();
    
    // Add new data to potentially polluted filesystem
    let new_docs: DocumentList = json!([{"id": "doc2", "data": "second index data"}]).try_into().unwrap();
    second_index.insert_documents(new_docs).await.unwrap();

    // This commit should clean old data AND write new data
    test_context.commit_all().await.unwrap();

    let files_after_commit: Vec<_> = if let Ok(entries) = fs::read_dir(&index_dir) {
        entries.filter_map(|e| e.ok()).map(|e| e.file_name()).collect()
    } else {
        vec![]
    };

    println!("Files after commit with new index: {:?}", files_after_commit);

    // Now test for data corruption/pollution
    let all_results = collection_client
        .search(json!({"term": "*"}).try_into().unwrap())
        .await
        .unwrap();

    let first_results = collection_client
        .search(json!({"term": "first"}).try_into().unwrap())
        .await
        .unwrap();

    let second_results = collection_client
        .search(json!({"term": "second"}).try_into().unwrap())
        .await
        .unwrap();

    println!("Search results after index ID reuse:");
    println!("  Total documents: {}", all_results.count);
    println!("  'first' matches: {}", first_results.count);
    println!("  'second' matches: {}", second_results.count);

    // These assertions will fail if there's filesystem pollution
    assert_eq!(all_results.count, 1, "Should have exactly 1 document (new index only)");
    assert_eq!(first_results.count, 0, "Should have 0 matches for 'first' (old data should be gone)");
    assert_eq!(second_results.count, 1, "Should have 1 match for 'second' (new data)");

    // Additional check: if filesystem pollution occurred, we might see old files mixed with new
    // This would be evidence of the bug even if search results are correct
}