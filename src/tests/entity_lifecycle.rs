use serde_json::json;

use crate::ai::OramaModel;
use crate::collection_manager::sides::write::OramaModelSerializable;
use crate::collection_manager::sides::write::WriteError;
use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::CreateCollection;
use crate::types::DocumentList;
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
                read_api_key: collection_client.read_api_key,
                write_api_key: match collection_client.write_api_key {
                    WriteApiKey::ApiKey(k) => k,
                },
                description: None,
                embeddings_model: Some(OramaModelSerializable(OramaModel::BgeSmall)),
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

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
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

    let test_context = test_context.reload().await;

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

    println!("Deleting data dir {:?}", config);

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
