use std::time::Duration;

use anyhow::Result;
use axum::{Json, Router};
use axum_extra::{headers, TypedHeader};
use redact::Secret;
use serde_json::{json, Value};
use tokio::time::sleep;

use crate::{
    collection_manager::{
        dto::{ApiKey, CreateCollectionFrom, SwapCollections},
        sides::notify::NotifierConfig,
    },
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_temp_insert_swap() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    let write_api_key = ApiKey(Secret::new("my-write-api-key".to_string()));
    let read_api_key = ApiKey(Secret::new("my-read-api-key".to_string()));

    let temp_coll_id = write_side
        .create_collection_from(
            write_api_key.clone(),
            CreateCollectionFrom {
                from: collection_id,
                embeddings: None,
                language: None,
            },
        )
        .await?;

    insert_docs(
        write_side.clone(),
        write_api_key.clone(),
        temp_coll_id,
        vec![json!({
            "title": "avvocata",
        })],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            temp_coll_id,
            json!({
                "term": "avvocata",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    write_side
        .swap_collections(
            write_api_key.clone(),
            SwapCollections {
                from: temp_coll_id,
                to: collection_id,
                reference: None,
            },
        )
        .await?;

    sleep(Duration::from_millis(100)).await;

    let output = read_side
        .search(
            read_api_key.clone(),
            collection_id,
            json!({
                "term": "avvocata",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(output.count, 1);

    let stats = read_side
        .collection_stats(read_api_key.clone(), collection_id)
        .await?;
    assert_eq!(stats.document_count, 1);

    let output = write_side
        .list_collections(ApiKey(Secret::new("my-master-api-key".to_string())))
        .await?;

    assert!(!output.iter().any(|c| c.id == temp_coll_id));

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_external_notification_on_swap() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    type AuthorizationBearerHeader =
        TypedHeader<headers::Authorization<headers::authorization::Bearer>>;

    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);

    let notify_handler = move |TypedHeader(auth): AuthorizationBearerHeader,
                               Json(body): Json<Value>| async move {
        sender.send((auth, body)).await.unwrap();
        (axum::http::StatusCode::OK, "ok")
    };

    let router: Router<()> = Router::new().route("/notify", axum::routing::post(notify_handler));
    let addr = "localhost:0";
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    let add = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let mut config = create_oramacore_config();
    config.reader_side.config.notifier = Some(NotifierConfig {
        url: format!("http://{}/notify", add).parse().unwrap(),
        authorization_token: Some("Bearer my-secret-token".to_string()),
        retry_count: 3,
        timeout: Duration::from_secs(5),
    });
    let (write_side, reader_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;

    let write_api_key = ApiKey(Secret::new("my-write-api-key".to_string()));

    let temp_coll_id = write_side
        .create_collection_from(
            write_api_key.clone(),
            CreateCollectionFrom {
                from: collection_id,
                embeddings: None,
                language: None,
            },
        )
        .await?;

    insert_docs(
        write_side.clone(),
        write_api_key.clone(),
        temp_coll_id,
        vec![json!({
            "title": "avvocata",
        })],
    )
    .await?;

    write_side
        .swap_collections(
            write_api_key.clone(),
            SwapCollections {
                from: temp_coll_id,
                to: collection_id,
                reference: Some("test".to_string()),
            },
        )
        .await?;

    let (auth, body) = receiver.recv().await.unwrap();
    assert_eq!(auth.0.token(), "my-secret-token");
    assert_eq!(
        body,
        json!({
            "type": "CollectionSubstituted",
            "data": {
                "target_collection": collection_id.to_string(),
                "source_collection": temp_coll_id.to_string(),
                "reference": "test".to_string()
            }
        })
    );

    drop(reader_side);

    Ok(())
}
