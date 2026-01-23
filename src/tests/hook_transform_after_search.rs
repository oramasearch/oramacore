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
