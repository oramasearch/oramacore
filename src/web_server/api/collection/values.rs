use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::sides::write::{WriteError, WriteSide},
    types::{CollectionId, WriteApiKey},
};

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/values/set",
            post(set_value),
        )
        .route(
            "/v1/collections/{collection_id}/values/delete",
            post(delete_value),
        )
        .route(
            "/v1/collections/{collection_id}/values/get/{key}",
            get(get_value),
        )
        .route(
            "/v1/collections/{collection_id}/values/list",
            get(list_values),
        )
        .with_state(write_side)
}

#[derive(Deserialize)]
struct SetValueParams {
    key: String,
    value: String,
}

async fn set_value(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<SetValueParams>,
) -> Result<impl IntoResponse, WriteError> {
    let SetValueParams { key, value } = params;

    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.set_value(key, value).await?;

    Ok(Json(json!({ "success": true })))
}

#[derive(Deserialize)]
struct DeleteValueParams {
    key: String,
}

async fn delete_value(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<DeleteValueParams>,
) -> Result<impl IntoResponse, WriteError> {
    let DeleteValueParams { key } = params;

    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    let removed = collection.delete_value(&key).await?;

    Ok(Json(json!({ "success": true, "removed": removed })))
}

async fn get_value(
    Path((collection_id, key)): Path<(CollectionId, String)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> Result<impl IntoResponse, WriteError> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    let value = collection.get_value(&key).await;

    Ok(Json(json!({ "key": key, "value": value })))
}

async fn list_values(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> Result<impl IntoResponse, WriteError> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    let values: HashMap<String, String> = collection.list_values().await;

    Ok(Json(json!({ "values": values })))
}
