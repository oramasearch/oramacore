use std::collections::HashMap;

use anyhow::Context;
use futures::FutureExt;
use oramacore_lib::hook_storage::HookType;
use oramacore_lib::secrets::local::LocalSecretsConfig;
use oramacore_lib::secrets::SecretsManagerConfig;
use serde_json::json;

use crate::tests::utils::{create_oramacore_config, init_log, wait_for, TestContext};
use crate::types::{CollectionStatsRequest, DocumentList, ReadApiKey, WriteApiKey};

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
        delete doc[1].secret;
        doc[1].redacted = true;
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
        delete doc[1].secret;
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
    return documents.filter(doc => doc[1].visibility === "public");
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

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_receives_values_and_secrets() {
    init_log();

    // Generate the collection ID upfront so we can seed matching secrets
    let collection_id = TestContext::generate_collection_id();
    let col_id_str = collection_id.to_string();

    let secrets_config = SecretsManagerConfig {
        aws: None,
        local: Some(LocalSecretsConfig {
            secrets: HashMap::from([(
                format!("{col_id_str}_TOKEN"),
                "search_token_789".to_string(),
            )]),
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

    let index_client = collection_client.create_index().await.unwrap();

    let documents: DocumentList = json!([
        {"id": "1", "title": "hello world"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

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
        .set_value("visibility".to_string(), "public".to_string())
        .await
        .unwrap();
    drop(collection);

    // Hook uses collectionValues (2nd arg) and secrets (3rd arg) to annotate documents.
    collection_client
        .insert_hook(
            HookType::TransformDocumentAfterSearch,
            r#"
const transformDocumentAfterSearch = function (documents, collectionValues, secrets) {
    return documents.map(doc => {
        doc[1].from_values = collectionValues.visibility || "missing";
        doc[1].values_count = Object.keys(collectionValues).length;
        doc[1].secret_token = secrets.TOKEN || "missing";
        doc[1].secret_count = Object.keys(secrets).length;
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

    // Wait for collection value to propagate to the reader
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key.clone();
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, &read_api_key).await?;
            let value = collection.get_value("visibility").await;
            if value.is_none() {
                return Err(anyhow::anyhow!("collection value not yet available"));
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
    // Verify the hook received collection values
    assert_eq!(
        document.get("from_values").unwrap(),
        "public",
        "Hook should receive collection values as second argument"
    );
    assert_eq!(
        document.get("values_count").unwrap(),
        1,
        "Only 1 collection value should be present"
    );
    // Verify the hook received secrets
    assert_eq!(
        document.get("secret_token").unwrap(),
        "search_token_789",
        "After-search hook should receive TOKEN secret"
    );
    assert_eq!(
        document.get("secret_count").unwrap(),
        1,
        "Only 1 secret should match this collection"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transform_after_search_hook_invalid_return_type() {
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
    // Return invalid type - should cause deserialization error
    return 1;
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
