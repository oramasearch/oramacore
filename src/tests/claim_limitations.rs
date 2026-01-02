use crate::collection_manager::sides::write::WriteError;
use crate::collection_manager::sides::ReplaceIndexReason;
use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::ReplaceIndexRequest;
use crate::types::{ClaimLimits, Claims, StackString, WriteApiKey};
use serde_json::json;
use std::convert::TryInto;

/// Test that inserting documents into a temp index enforces JWT limits.
///
/// The formula for temp index limit check is:
/// temp_docs + all_runtime_indexes - linked_runtime_index <= max_doc_count
///
/// This means: if the temp index will replace a runtime index with 10 docs,
/// and the limit is 15, we can only insert up to 15 docs into the temp index.
#[tokio::test(flavor = "multi_thread")]
async fn test_temp_index_doc_limitation() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let api_key = WriteApiKey::Claims(Claims {
        sub: collection_client.collection_id,
        aud: StackString::try_new("aa").unwrap(),
        iss: StackString::try_new("aa").unwrap(),
        scope: StackString::try_new("write").unwrap(),
        limits: ClaimLimits { max_doc_count: 15 },
    });

    // Create runtime index with 10 docs
    let runtime_index = collection_client.create_index().await.unwrap();
    let docs_10: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    runtime_index
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            runtime_index.index_id,
            docs_10.clone().try_into().unwrap(),
        )
        .await
        .unwrap();

    // Create temp index linked to runtime index
    let temp_index = collection_client
        .create_temp_index(runtime_index.index_id)
        .await
        .unwrap();

    // Insert 10 docs into temp index - should succeed
    // (10 docs in temp + 0 other runtime - 10 linked runtime docs = 10 <= 15)
    // Actually: formula counts all runtime EXCEPT linked, so:
    // 0 (other runtime) + 0 (current temp) + 10 (new) = 10 <= 15
    temp_index
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            temp_index.index_id,
            docs_10.clone().try_into().unwrap(),
        )
        .await
        .unwrap();

    // Insert 6 more docs into temp index - should FAIL
    // Formula: 0 (other runtime) + 10 (current temp) + 6 (new) = 16 > 15
    let docs_6: Vec<_> = (10_u8..16_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();

    let err = temp_index
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            temp_index.index_id,
            docs_6.try_into().unwrap(),
        )
        .await
        .expect_err("Should fail: exceeds limit");

    assert!(matches!(err, WriteError::DocumentLimitExceeded(_, 15)));

    drop(test_context);
}

/// Test that temp index limit check correctly excludes the linked runtime index.
///
/// Scenario:
/// - Limit: 25 docs
/// - Runtime index 1: 10 docs
/// - Runtime index 2: 10 docs
/// - Temp index (linked to index 1): should be able to hold up to 15 docs
///   (because index 1's 10 docs are excluded, but index 2's 10 docs count)
#[tokio::test(flavor = "multi_thread")]
async fn test_temp_index_excludes_linked_runtime_index() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let api_key = WriteApiKey::Claims(Claims {
        sub: collection_client.collection_id,
        aud: StackString::try_new("aa").unwrap(),
        iss: StackString::try_new("aa").unwrap(),
        scope: StackString::try_new("write").unwrap(),
        limits: ClaimLimits { max_doc_count: 25 },
    });

    // Create runtime index 1 with 10 docs
    let runtime_index_1 = collection_client.create_index().await.unwrap();
    let docs_10: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    runtime_index_1
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            runtime_index_1.index_id,
            docs_10.clone().try_into().unwrap(),
        )
        .await
        .unwrap();

    // Create runtime index 2 with 10 docs
    let runtime_index_2 = collection_client.create_index().await.unwrap();
    runtime_index_2
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            runtime_index_2.index_id,
            docs_10.clone().try_into().unwrap(),
        )
        .await
        .unwrap();

    // Create temp index linked to runtime index 1
    let temp_index = collection_client
        .create_temp_index(runtime_index_1.index_id)
        .await
        .unwrap();

    // Insert 15 docs - should succeed
    // Formula: 10 (index 2) + 0 (current temp) + 15 (new) = 25 <= 25
    let docs_15: Vec<_> = (0_u8..15_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    temp_index
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            temp_index.index_id,
            docs_15.try_into().unwrap(),
        )
        .await
        .unwrap();

    // Insert 1 more doc - should FAIL
    // Formula: 10 (index 2) + 15 (current temp) + 1 (new) = 26 > 25
    let docs_1: Vec<_> = (15_u8..16_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();

    let err = temp_index
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            temp_index.index_id,
            docs_1.try_into().unwrap(),
        )
        .await
        .expect_err("Should fail: exceeds limit");

    assert!(matches!(err, WriteError::DocumentLimitExceeded(_, 25)));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_doc_limitation_zero() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();

    let res = index_client
        .writer
        .insert_documents(
            WriteApiKey::Claims(Claims {
                sub: collection_client.collection_id,
                aud: StackString::try_new("aa").unwrap(),
                iss: StackString::try_new("aa").unwrap(),
                scope: StackString::try_new("write").unwrap(),
                limits: ClaimLimits { max_doc_count: 0 },
            }),
            collection_client.collection_id,
            index_client.index_id,
            docs.try_into().unwrap(),
        )
        .await;
    let err = res.expect_err("Should return error");
    assert!(matches!(err, WriteError::DocumentLimitExceeded(_, 0)));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_doc_limitation_some() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let api_key = WriteApiKey::Claims(Claims {
        sub: collection_client.collection_id,
        aud: StackString::try_new("aa").unwrap(),
        iss: StackString::try_new("aa").unwrap(),
        scope: StackString::try_new("write").unwrap(),
        limits: ClaimLimits { max_doc_count: 15 },
    });

    let docs: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();

    index_client
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            index_client.index_id,
            docs.clone().try_into().unwrap(),
        )
        .await
        .unwrap();

    let res = index_client
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            index_client.index_id,
            docs.try_into().unwrap(),
        )
        .await;

    let err = res.expect_err("Should return error");
    assert!(matches!(err, WriteError::DocumentLimitExceeded(_, 15)));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_replace_doc_limitation() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let api_key = WriteApiKey::Claims(Claims {
        sub: collection_client.collection_id,
        aud: StackString::try_new("aa").unwrap(),
        iss: StackString::try_new("aa").unwrap(),
        scope: StackString::try_new("write").unwrap(),
        limits: ClaimLimits { max_doc_count: 15 },
    });

    let docs: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();

    let index_client_1 = collection_client.create_index().await.unwrap();
    index_client_1
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            index_client_1.index_id,
            docs.clone().try_into().unwrap(),
        )
        .await
        .unwrap();

    let index_client_2 = collection_client
        .create_temp_index(index_client_1.index_id)
        .await
        .unwrap();

    index_client_2
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            index_client_2.index_id,
            docs.clone().try_into().unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .writer
        .replace_index(
            api_key,
            collection_client.collection_id,
            ReplaceIndexRequest {
                reference: None,
                runtime_index_id: index_client_1.index_id,
                temp_index_id: index_client_2.index_id,
            },
            ReplaceIndexReason::CollectionReindexed,
        )
        .await
        .unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();

    let index_client_3 = collection_client
        .create_temp_index(index_client_1.index_id)
        .await
        .unwrap();

    // With temp index limit checking, inserting 20 docs into temp index 3
    // should now fail at insert time, not at replace time.
    // Formula: 0 (other runtime - index_1 is excluded as linked) + 0 (current temp) + 20 (new) = 20 > 15
    let err = index_client_3
        .writer
        .insert_documents(
            api_key,
            collection_client.collection_id,
            index_client_3.index_id,
            docs.clone().try_into().unwrap(),
        )
        .await
        .expect_err("Should fail: 20 docs exceeds limit of 15");

    assert!(matches!(err, WriteError::DocumentLimitExceeded(_, 15)));

    drop(test_context);
}
