use oramacore_lib::hook_storage::HookType;

use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use crate::types::{CollectionStatsRequest, DocumentList};
use anyhow::Context;
use futures::FutureExt;
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_happy_path() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "hello", "secret": "password123"},
        {"id": "2", "title": "world", "secret": "token456"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentAfterSearch,
            r#"
const transformDocumentAfterSearch = function (documents) {
    return documents.map(doc => {
        delete doc.document.secret;
        doc.document.redacted = true;
        return doc;
    });
}
export default { transformDocumentAfterSearch }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let collection_id = collection_client.collection_id;
    wait_for(&test_context, |t| {
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

            if !stats
                .hooks
                .contains(&HookType::TransformDocumentAfterSearch)
            {
                return Err(anyhow::anyhow!("hook not arrived yet"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

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
    assert_eq!(document.get("title").unwrap(), "hello");
    assert!(document.get("secret").is_none());
    assert_eq!(document.get("redacted").unwrap(), true);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_document_removed() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "hello"},
        {"id": "2", "title": "world"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentAfterSearch,
            r#"
const transformDocumentAfterSearch = function (documents) {
    return [];
}
export default { transformDocumentAfterSearch }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let collection_id = collection_client.collection_id;
    wait_for(&test_context, |t| {
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

            if !stats
                .hooks
                .contains(&HookType::TransformDocumentAfterSearch)
            {
                return Err(anyhow::anyhow!("hook not arrived yet"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

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

    assert_eq!(results.count, 0);
    assert_eq!(results.hits.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_hook_returns_none() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "hello"},
        {"id": "2", "title": "world"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentAfterSearch,
            r#"
const transformDocumentAfterSearch = function (documents) {
    // Hook returns null/undefined - original results should be preserved
    return null;
}
export default { transformDocumentAfterSearch }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let collection_id = collection_client.collection_id;
    wait_for(&test_context, |t| {
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

            if !stats
                .hooks
                .contains(&HookType::TransformDocumentAfterSearch)
            {
                return Err(anyhow::anyhow!("hook not arrived yet"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

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

    // When hook returns None, original search results should be preserved
    assert_eq!(results.count, 1);
    assert_eq!(results.hits.len(), 1);
    let document = results.hits[0].document.as_ref().unwrap();
    assert_eq!(document.get("title").unwrap(), "hello");
    assert_eq!(document.get("id").unwrap(), "1");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_hook_after_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([{"id": "1", "title": "hello", "secret": "password"}])
        .try_into()
        .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentAfterSearch,
            r#"
const transformDocumentAfterSearch = function (documents) {
    return documents.map(doc => {
        delete doc.document.secret;
        return doc;
    });
}
export default { transformDocumentAfterSearch }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let write_api_key = collection_client.write_api_key;
    let collection_id = collection_client.collection_id;
    let read_api_key_for_reload = read_api_key.clone();

    wait_for(&test_context, |t| {
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

            if !stats
                .hooks
                .contains(&HookType::TransformDocumentAfterSearch)
            {
                return Err(anyhow::anyhow!("hook not arrived yet"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key_for_reload)
        .unwrap();

    let stats = collection_client.reader_stats().await.unwrap();
    assert!(stats
        .hooks
        .contains(&HookType::TransformDocumentAfterSearch));

    // Verify hook still works after reload
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
    assert!(results.hits[0]
        .document
        .as_ref()
        .unwrap()
        .get("secret")
        .is_none());

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_conditional_filter() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "public doc", "visibility": "public"},
        {"id": "2", "title": "private doc", "visibility": "private"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentAfterSearch,
            r#"
const transformDocumentAfterSearch = function (documents) {
    return documents.filter(doc => doc.document.visibility === "public");
}
export default { transformDocumentAfterSearch }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let collection_id = collection_client.collection_id;
    wait_for(&test_context, |t| {
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

            if !stats
                .hooks
                .contains(&HookType::TransformDocumentAfterSearch)
            {
                return Err(anyhow::anyhow!("hook not arrived yet"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

    let results = collection_client
        .search(
            json!({
                "term": "doc",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Only public document should be returned
    assert_eq!(results.count, 1);
    assert_eq!(
        results.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("title")
            .unwrap(),
        "public doc"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_hook_error() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([{"id": "1", "title": "hello"}]).try_into().unwrap();
    index_client.insert_documents(documents).await.unwrap();

    collection_client
        .insert_hook(
            HookType::TransformDocumentAfterSearch,
            r#"
const transformDocumentAfterSearch = function (documents) {
    throw new Error("Hook intentionally failed");
}
export default { transformDocumentAfterSearch }"#
                .to_string(),
        )
        .await
        .unwrap();

    let read_api_key = collection_client.read_api_key.clone();
    let collection_id = collection_client.collection_id;
    wait_for(&test_context, |t| {
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

            if !stats
                .hooks
                .contains(&HookType::TransformDocumentAfterSearch)
            {
                return Err(anyhow::anyhow!("hook not arrived yet"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "hello",
            })
            .try_into()
            .unwrap(),
        )
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Hook TransformDocumentAfterSearch execution failed"),
        "Got: {err}",
    );
}
