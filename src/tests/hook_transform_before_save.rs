use std::collections::HashMap;

use anyhow::Context;
use futures::FutureExt;
use oramacore_lib::hook_storage::HookType;
use oramacore_lib::secrets::local::LocalSecretsConfig;
use oramacore_lib::secrets::SecretsManagerConfig;
use serde_json::json;

use crate::tests::utils::{create_oramacore_config, init_log, wait_for, TestContext};
use crate::types::{
    CollectionStatsRequest, DocumentList, ReadApiKey, UpdateDocumentRequest, WriteApiKey,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_insert_happy_path() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    return documents.map(doc => {
        doc.title = doc.title.toUpperCase();
        doc.transformed = true;
        return doc;
    });
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "hello"},
        {"id": "2", "title": "world"}
    ])
    .try_into()
    .unwrap();

    index_client.insert_documents(documents).await.unwrap();

    let results = collection_client
        .search(
            json!({
                "term": "HELLO",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(results.count, 1);
    let document = results.hits[0].document.as_ref().unwrap();
    assert_eq!(document.get("title").unwrap(), "HELLO");
    assert_eq!(document.get("transformed").unwrap(), true);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_update_happy_path() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([{"id": "1", "title": "hello", "count": 1}])
        .try_into()
        .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    return documents.map(doc => {
        doc.title = doc.title.toUpperCase();
        doc.updated = true;
        return doc;
    });
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let req: UpdateDocumentRequest = serde_json::from_value(json!({
        "strategy": "merge",
        "documents": [{"id": "1", "count": 2}],
    }))
    .unwrap();
    index_client.update_documents(req).await.unwrap();

    let results = collection_client
        .search(
            json!({
                "term": "HELLO",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(results.count, 1);
    let document = results.hits[0].document.as_ref().unwrap();
    assert_eq!(document.get("title").unwrap(), "HELLO");
    assert_eq!(document.get("count").unwrap(), 2);
    assert_eq!(document.get("updated").unwrap(), true);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_insert_cannot_add_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    documents.push({id: "3", title: "extra"});
    return documents;
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([{"id": "1", "title": "test"}]).try_into().unwrap();

    let result = index_client.unchecked_insert_documents(documents).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Documents cannot be added or removed"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_insert_cannot_remove_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    return documents.slice(0, 1);
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "test"},
        {"id": "2", "title": "test2"}
    ])
    .try_into()
    .unwrap();

    let result = index_client.unchecked_insert_documents(documents).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Documents cannot be added or removed"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hooks_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::BeforeRetrieval,
            r#"
const beforeRetrieval = function () { }
export default { beforeRetrieval }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let write_api_key = collection_client.write_api_key;
    let collection_id = collection_client.collection_id;
    let read_api_key_for_reload = read_api_key.clone();
    let stats = wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        let read_api_key = read_api_key.clone();
        async move {
            let stats = reader
                .collection_stats(
                    &read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("")?;

            if stats.hooks.is_empty() {
                return Err(anyhow::anyhow!("hooks not arrived yet"));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    assert_eq!(stats.hooks, vec![HookType::BeforeRetrieval,]);

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key_for_reload)
        .unwrap();

    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.hooks, vec![HookType::BeforeRetrieval,]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_before_search_hook_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::BeforeSearch,
            r#"
const beforeSearch = function (search_params, claims) { return search_params; }
export default { beforeSearch }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let write_api_key = collection_client.write_api_key;
    let collection_id = collection_client.collection_id;
    let read_api_key_for_reload = read_api_key.clone();
    let stats = wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        let read_api_key = read_api_key.clone();
        async move {
            let stats = reader
                .collection_stats(
                    &read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("")?;

            if stats.hooks.is_empty() {
                return Err(anyhow::anyhow!("hooks not arrived yet"));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    assert_eq!(stats.hooks, vec![HookType::BeforeSearch,]);

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key_for_reload)
        .unwrap();

    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.hooks, vec![HookType::BeforeSearch,]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_hook_error() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    throw new Error("Hook intentionally failed");
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([{"id": "1", "title": "test"}]).try_into().unwrap();

    let result = index_client.unchecked_insert_documents(documents).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    println!("Actual error: {err_msg}");
    assert!(
        err_msg.contains("Error in hook") || err_msg.contains("Hook") || err_msg.contains("failed"),
        "Expected error about hook, got: {err_msg}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_hook_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    return documents.map(doc => {
        doc.title = doc.title.toUpperCase();
        return doc;
    });
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let write_api_key = collection_client.write_api_key;
    let collection_id = collection_client.collection_id;
    let read_api_key_for_reload = read_api_key.clone();

    let stats = wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        let read_api_key = read_api_key.clone();
        async move {
            let stats = reader
                .collection_stats(
                    &read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("")?;

            if !stats.hooks.contains(&HookType::TransformDocumentBeforeSave) {
                return Err(anyhow::anyhow!("hook not arrived yet"));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    assert!(stats.hooks.contains(&HookType::TransformDocumentBeforeSave));

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key_for_reload)
        .unwrap();

    let stats = collection_client.reader_stats().await.unwrap();
    assert!(stats.hooks.contains(&HookType::TransformDocumentBeforeSave));

    // Verify hook still works after reload
    let index_client = collection_client.create_index().await.unwrap();
    let documents: DocumentList = json!([{"id": "1", "title": "hello"}]).try_into().unwrap();
    index_client.insert_documents(documents).await.unwrap();

    let results = collection_client
        .search(
            json!({
                "term": "HELLO",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(results.count, 1);
    assert_eq!(
        results.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("title")
            .unwrap(),
        "HELLO"
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_update_cannot_add_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([{"id": "1", "title": "test"}]).try_into().unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    documents.push({id: "2", title: "extra"});
    return documents;
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let req: UpdateDocumentRequest = serde_json::from_value(json!({
        "strategy": "merge",
        "documents": [{"id": "1", "count": 1}],
    }))
    .unwrap();
    let result = index_client.update_documents(req).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Documents cannot be added or removed"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_update_cannot_remove_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "test"},
        {"id": "2", "title": "test2"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    return [];
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let req: UpdateDocumentRequest = serde_json::from_value(json!({
        "strategy": "merge",
        "documents": [{"id": "1", "count": 1}, {"id": "2", "count": 2}],
    }))
    .unwrap();
    let result = index_client.update_documents(req).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Documents cannot be added or removed"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_shuffle_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents) {
    // Attempt to swap documents with id "1" and id "2"
    const doc1_idx = documents.findIndex(doc => doc.id === "1");
    const doc2_idx = documents.findIndex(doc => doc.id === "2");
    
    if (doc1_idx !== -1 && doc2_idx !== -1) {
        const temp = documents[doc1_idx];
        documents[doc1_idx] = documents[doc2_idx];
        documents[doc2_idx] = temp;
    }
    
    return documents;
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    // Attempt to insert documents with id "1" and "2"
    // This should fail because the hook tries to shuffle them,
    // and this can cause a miss match between the indexed documents and the saved one.
    let documents: DocumentList = json!([
        {"id": "1", "title": "original first document"},
        {"id": "2", "title": "original second document"}
    ])
    .try_into()
    .unwrap();

    let result = index_client.unchecked_insert_documents(documents).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Document IDs cannot be changed or reordered"),
        "Expected error about document reordering, got: {err_msg}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_before_save_receives_values_and_secrets() {
    init_log();

    // Generate the collection ID upfront so we can seed matching secrets
    let collection_id = TestContext::generate_collection_id();
    let col_id_str = collection_id.to_string();

    let secrets_config = SecretsManagerConfig {
        aws: None,
        local: Some(LocalSecretsConfig {
            secrets: HashMap::from([
                (format!("{col_id_str}_API_KEY"), "my_secret_key".to_string()),
                (format!("{col_id_str}_DB_HOST"), "db.internal".to_string()),
                // This secret belongs to a different collection and should not be visible
                ("othercol_TOKEN".to_string(), "other_token".to_string()),
            ]),
        }),
    };

    let mut config = create_oramacore_config();
    config.writer_side.master_api_key = TestContext::generate_api_key();
    config.writer_side.secrets_manager = Some(secrets_config.clone());
    config.reader_side.secrets_manager = Some(secrets_config);
    let test_context = TestContext::new_with_config(config).await;

    // Create the collection with the known ID so secrets prefix matches
    let write_api_key = TestContext::generate_api_key();
    let read_api_key_raw = TestContext::generate_api_key();
    let read_api_key = ReadApiKey::from_api_key(read_api_key_raw);

    test_context
        .writer
        .create_collection(
            test_context.master_api_key,
            crate::types::CreateCollection {
                id: collection_id,
                description: None,
                mcp_description: None,
                read_api_key: read_api_key_raw,
                write_api_key,
                language: None,
                embeddings_model: Some(crate::python::embeddings::Model::BGESmall),
            },
        )
        .await
        .unwrap();

    let read_api_key_for_wait = read_api_key.clone();
    wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        let read_api_key = read_api_key_for_wait.clone();
        async move {
            reader
                .collection_stats(
                    &read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("Collection not ready yet")
        }
        .boxed()
    })
    .await
    .unwrap();

    let collection_client = test_context
        .get_test_collection_client(
            collection_id,
            WriteApiKey::from_api_key(write_api_key),
            read_api_key,
        )
        .unwrap();

    // Set collection values that the hook should receive
    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();
    collection
        .set_value("env".to_string(), "testing".to_string())
        .await
        .unwrap();
    drop(collection);

    // Hook uses collectionValues (2nd arg) and secrets (3rd arg) to annotate documents.
    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents, collectionValues, secrets) {
    return documents.map(doc => {
        doc.from_values = collectionValues.env || "missing";
        doc.values_count = Object.keys(collectionValues).length;
        doc.secret_api_key = secrets.API_KEY || "missing";
        doc.secret_db_host = secrets.DB_HOST || "missing";
        doc.secret_other_token = secrets.TOKEN || "not_visible";
        doc.secret_count = Object.keys(secrets).length;
        return doc;
    });
}
export default { transformDocumentBeforeSave }"#
                .to_string(),
        )
        .await
        .unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "hello"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    let results = collection_client
        .search(
            json!({
                "term": "hello",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(results.count, 1);
    let document = results.hits[0].document.as_ref().unwrap();
    assert_eq!(
        document.get("from_values").unwrap(),
        "testing",
        "Hook should receive collection values as second argument"
    );
    assert_eq!(
        document.get("values_count").unwrap(),
        1,
        "Only 1 collection value should be present"
    );
    assert_eq!(
        document.get("secret_api_key").unwrap(),
        "my_secret_key",
        "Hook should receive API_KEY secret"
    );
    assert_eq!(
        document.get("secret_db_host").unwrap(),
        "db.internal",
        "Hook should receive DB_HOST secret"
    );
    assert_eq!(
        document.get("secret_other_token").unwrap(),
        "not_visible",
        "Secrets from other collections should not be visible"
    );
    assert_eq!(
        document.get("secret_count").unwrap(),
        2,
        "Only 2 secrets should match this collection"
    );
}
