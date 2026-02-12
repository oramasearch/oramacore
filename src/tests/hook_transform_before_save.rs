use oramacore_lib::hook_storage::HookType;

use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use crate::types::{CollectionStatsRequest, DocumentList, UpdateDocumentRequest};
use anyhow::Context;
use futures::FutureExt;
use serde_json::json;

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

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    // Set collection values before the hook is installed
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
    // Since no secrets manager is configured in tests, secrets will be an empty object.
    collection_client
        .insert_hook(
            HookType::TransformDocumentBeforeSave,
            r#"
const transformDocumentBeforeSave = function (documents, collectionValues, secrets) {
    return documents.map(doc => {
        doc.from_values = collectionValues.env || "missing";
        doc.secrets_type = typeof secrets;
        doc.secrets_empty = Object.keys(secrets).length === 0;
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
    // Verify the hook received collection values
    assert_eq!(
        document.get("from_values").unwrap(),
        "testing",
        "Hook should receive collection values as second argument"
    );
    // Verify the hook received secrets (empty object since no secrets manager configured)
    assert_eq!(
        document.get("secrets_type").unwrap(),
        "object",
        "Hook should receive secrets as third argument"
    );
    assert_eq!(
        document.get("secrets_empty").unwrap(),
        true,
        "Secrets should be empty when no secrets manager is configured"
    );
}
