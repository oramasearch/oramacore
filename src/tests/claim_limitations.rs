use crate::collection_manager::sides::write::WriteError;
use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::{ClaimLimits, Claims, StackString, WriteApiKey};
use serde_json::json;
use std::convert::TryInto;

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
