use std::{sync::Arc, time::Duration};

use anyhow::{bail, Result};
use axum::extract::State;
use futures::FutureExt;
use http::StatusCode;
use serde_json::{json, Value};

use crate::collection_manager::sides::read::ReadSide;
use crate::tests::utils::{create_oramacore_config, init_log, wait_for, TestContext};
use crate::web_server::api::health_ready;

/// `/health/ready` turns 503 while an operation received from the stream cannot be applied,
/// and back to 200 once it is.
#[tokio::test(flavor = "multi_thread")]
async fn test_health_ready_reports_apply_stall() {
    init_log();

    let mut config = create_oramacore_config();
    config.writer_side.master_api_key = TestContext::generate_api_key();
    config.reader_side.replication.apply_stall_threshold = Duration::from_millis(500);
    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    index_client
        .insert_documents(
            json!([
                { "id": "1", "text": "the quick brown fox" },
                { "id": "2", "text": "jumps over the lazy dog" },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let (status, body) = readiness(&test_context.reader).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["ready"], json!(true));
    assert!(body["applied_offset"].as_u64().unwrap() > 0);
    assert!(body["last_applied_at"].is_string());
    assert!(body["last_received_at"].is_string());
    assert!(body.get("last_commit_at").is_some());
    let applied_before_stall = body["applied_offset"].as_u64().unwrap();

    // Hold the index read locks, as a long search would. Deleting a document needs the
    // index write lock, so the receive loop stalls on that operation.
    let collection = test_context
        .reader
        .get_collection(
            collection_client.collection_id,
            &collection_client.read_api_key,
        )
        .await
        .unwrap();
    let index_locks = collection.get_index_read_locks(None).await.unwrap();
    index_client
        .uncheck_delete_documents(vec!["1".to_string()])
        .await
        .unwrap();

    let body = wait_for_readiness(&test_context.reader, StatusCode::SERVICE_UNAVAILABLE)
        .await
        .unwrap();
    assert_eq!(body["ready"], json!(false));

    drop(index_locks);
    drop(collection);

    let body = wait_for_readiness(&test_context.reader, StatusCode::OK)
        .await
        .unwrap();
    assert_eq!(body["ready"], json!(true));
    assert!(body["applied_offset"].as_u64().unwrap() > applied_before_stall);
}

async fn readiness(read_side: &Arc<ReadSide>) -> (StatusCode, Value) {
    let response = health_ready(State(Some(read_side.clone()))).await;
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

async fn wait_for_readiness(read_side: &Arc<ReadSide>, expected: StatusCode) -> Result<Value> {
    wait_for(read_side, |read_side| {
        async move {
            let (status, body) = readiness(read_side).await;
            if status != expected {
                bail!("Readiness is {status}, expected {expected}: {body}");
            }
            Ok(body)
        }
        .boxed()
    })
    .await
}
