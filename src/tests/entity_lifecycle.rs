use std::time::Duration;

use anyhow::bail;
use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::collection_manager::sides::read::IndexFieldStatsType;
use crate::collection_manager::sides::write::WriteError;
use crate::python::embeddings::Model;
use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use crate::types::CreateCollection;
use crate::types::DocumentList;
use crate::types::ReadApiKey;
use crate::types::WriteApiKey;

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_id_already_exists() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let output = test_context
        .writer
        .create_collection(
            test_context.master_api_key,
            CreateCollection {
                id: collection_client.collection_id,
                read_api_key: match collection_client.read_api_key.clone() {
                    ReadApiKey::ApiKey(k) => k,
                    _ => panic!(),
                },
                write_api_key: match collection_client.write_api_key {
                    WriteApiKey::ApiKey(k) => k,
                    _ => panic!(),
                },
                description: None,
                mcp_description: None,
                embeddings_model: Some(Model::BGESmall),
                language: None,
            },
        )
        .await;

    assert!(matches!(
        output,
        Err(WriteError::CollectionAlreadyExists(_))
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_collection() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.indexes_stats.len(), 1);

    let _ = collection_client.create_index().await.unwrap();
    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.indexes_stats.len(), 2);

    index_1_client.delete().await.unwrap();

    let err = index_1_client.insert_documents(DocumentList(vec![])).await;
    assert!(err.is_err());

    collection_client.delete().await.unwrap();

    let err = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await;
    assert!(err.is_err());

    test_context.commit_all().await.unwrap();

    let err = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await;
    assert!(err.is_err());

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let err = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await;
    assert!(err.is_err());

    drop(test_context);
}

/// A collection deleted and created again with the same ID before the read side commits
/// must keep its documents across the commit and a restart.
#[tokio::test(flavor = "multi_thread")]
async fn test_recreate_collection_before_commit() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_id = TestContext::generate_collection_id();
    let write_api_key = TestContext::generate_api_key();
    let read_api_key = TestContext::generate_api_key();
    let documents = json!([
        { "id": "1", "text": "first document" },
        { "id": "2", "text": "second document" },
        { "id": "3", "text": "third document" },
    ]);
    let doc_ids: Vec<String> = vec!["1".to_string(), "2".to_string(), "3".to_string()];

    let collection_client = test_context
        .create_collection_with_id(collection_id, write_api_key, read_api_key, None)
        .await
        .unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    index_client
        .insert_documents(documents.clone().try_into().unwrap())
        .await
        .unwrap();
    // Embeddings are computed asynchronously: wait for them, so no operation for the
    // deleted index reaches the read side after the collection is created again.
    wait_for(&collection_client, |collection_client| {
        let index_id = index_client.index_id;
        async move {
            let mut index_stats = collection_client
                .reader_stats()
                .await?
                .indexes_stats
                .into_iter()
                .find(|i| i.id == index_id)
                .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
            let IndexFieldStatsType::EmbeddingFieldStorage(stats) =
                index_stats.fields_stats.remove(0).stats
            else {
                bail!("Expected embedding field stats");
            };
            if stats.vector_count < 3 {
                bail!("Expected 3 embeddings, found {}", stats.vector_count);
            }
            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();
    collection_client.delete().await.unwrap();

    // No commit in between: the read side still holds the deleted collection.
    let collection_client = test_context
        .create_collection_with_id(collection_id, write_api_key, read_api_key, None)
        .await
        .unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    // The read side only logs a failed collection commit, so the storage is
    // checked through batch-get after the commit and after a restart.
    test_context.commit_all().await.unwrap();

    let found = collection_client
        .batch_get_documents(doc_ids.clone())
        .await
        .expect("batch-get after commit should succeed");
    assert_eq!(found.len(), 3);

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_id,
            WriteApiKey::from_api_key(write_api_key),
            ReadApiKey::from_api_key(read_api_key),
        )
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "document",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .expect("search after reload should succeed");
    assert_eq!(result.count, 3);

    let found = collection_client
        .batch_get_documents(doc_ids)
        .await
        .expect("batch-get after reload should succeed");
    assert_eq!(found.len(), 3);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    index_1_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Tommaso A" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let index_2_client = collection_client.create_index().await.unwrap();
    index_2_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Tommaso B" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2);

    index_1_client.delete().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_index_committed() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    index_1_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Tommaso A" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let index_2_client = collection_client.create_index().await.unwrap();
    index_2_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Tommaso B" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2);

    index_1_client.delete().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    let config = test_context.config.reader_side.config.data_dir.clone();

    println!("Deleting data dir {config:?}");

    // The index is deleted
    let path = config
        .join("collections")
        .join(collection_client.collection_id.as_str())
        .join("indexes");
    let indexes = std::fs::read_dir(path).unwrap().collect::<Vec<_>>();
    assert_eq!(indexes.len(), 1);

    // The documents are deleted
    let path = config.join("docs");
    let documents = std::fs::read_dir(path).unwrap().collect::<Vec<_>>();
    assert_eq!(documents.len(), 1);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_promotion_with_committed_and_uncommitted_data() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let temp_index_client = collection_client
        .create_temp_index(index_client.index_id)
        .await
        .unwrap();

    let docs = (0..20)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text",
                "metadata": { "is_good": i % 2 == 0 }
            })
        })
        .collect::<Vec<_>>();
    temp_index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();
    test_context.commit_all().await.unwrap();

    let docs = (20..40)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text",
                "metadata": { "is_good": i % 2 == 0 }
            })
        })
        .collect::<Vec<_>>();
    temp_index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();
    collection_client
        .replace_index(index_client.index_id, temp_index_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await; // Wait for the replace to be effective

    test_context.commit_all().await.unwrap();

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 40);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unchanged_field_path_after_reload() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Commit 1: Create field with data
    index_client
        .insert_documents(
            json!([
                {"id": "1", "number_field": 100},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    test_context.commit_all().await.unwrap(); // offset-0

    // Commit 2: Add DIFFERENT field (number_field unchanged)
    index_client
        .insert_documents(
            json!([
                {"id": "2", "string_field": "test"},  // No number_field!
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    test_context.commit_all().await.unwrap(); // offset-1

    // number_field should return MergeResult::Unchanged
    // and keep data_dir = offset-0

    // Commit 3: Again, no changes to number_field
    index_client
        .insert_documents(
            json!([
                {"id": "3", "string_field": "test2"},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    test_context.commit_all().await.unwrap(); // offset-2

    // Read index.json and check paths
    let index_json_path = test_context
        .config
        .reader_side
        .config
        .data_dir
        .join("collections")
        .join(collection_client.collection_id.as_str())
        .join("indexes")
        .join(index_client.index_id.to_string())
        .join("index.json");

    let content = std::fs::read_to_string(&index_json_path).unwrap();
    println!("index.json content:\n{content}");

    // Check for mixed offsets
    let offset_0_count = content.matches("offset-0").count();
    let offset_1_count = content.matches("offset-1").count();
    let offset_2_count = content.matches("offset-2").count();

    println!("offset-0: {offset_0_count}, offset-1: {offset_1_count}, offset-2: {offset_2_count}");

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();
    // Reload
    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    // Try to search - this should work if paths are correct
    let result = collection_client
        .search(
            json!({
                "term": "test"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(result.count, 2);

    drop(test_context);
}
