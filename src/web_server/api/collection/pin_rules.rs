use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use http::StatusCode;
use crate::{pin_rules::PinRule, types::ApiKey};
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::sides::{read::ReadSide, write::WriteSide},
    types::{
        CollectionId, WriteApiKey
    },
};
use crate::collection_manager::sides::read::ReadError;

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/pin_rules/inner_list",
            get(list_pin_rules_ids_v1),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/pin_rules/insert",
            post(insert_pin_rule_v1),
        )
        .route(
            "/v1/collections/{collection_id}/pin_rules/delete",
            get(delete_pin_rule_v1),
        )
        .route(
            "/v1/collections/{collection_id}/pin_rules/list",
            get(list_pin_rule_v1),
        )
        .with_state(write_side)
}

async fn insert_pin_rule_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(pin_rule): Json<PinRule<String>>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await.unwrap();

    collection
        .insert_pin_rule(pin_rule)
        .await
        .map(|_| {
            Json(json!({ "success": true }))
        })
        .map_err(|e| {
            (StatusCode::BAD_REQUEST, Json(json!({ "success": false, "error": e.to_string() })))
        })
}

async fn delete_pin_rule_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(DeletePinRuleParams { id }): Json<DeletePinRuleParams>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await.unwrap();

    collection
        .delete_pin_rule(id)
        .await
        .map(|_| {
            Json(json!({ "success": true }))
        })
        .map_err(|e| {
            (StatusCode::BAD_REQUEST, Json(json!({ "success": false, "error": e.to_string() })))
        })
}

async fn list_pin_rule_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await.unwrap();

    collection
        .list_pin_rules()
        .await
        .map(|_| {
            Json(json!({ "success": true }))
        })
        .map_err(|e| {
            (StatusCode::BAD_REQUEST, Json(json!({ "success": false, "error": e.to_string() })))
        })
}


#[derive(Deserialize)]
struct ApiKeyQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn list_pin_rules_ids_v1(
    collection_id: CollectionId,
    Query(ApiKeyQueryParams { api_key }): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> Result<impl IntoResponse, impl IntoResponse> {

    let ids = read_side
        .list_pin_rule_ids(collection_id, api_key)
        .await?;

    Ok::<_, ReadError>(Json(json!({ "success": true, "data": ids })))
}

#[derive(Debug, Clone, Deserialize)]
struct DeletePinRuleParams {
    #[serde(rename = "pin_rule_id_to_delete")]
    id: String,
}
