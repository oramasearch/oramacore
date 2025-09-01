use std::sync::Arc;

use crate::{pin_rules::PinRule, types::ApiKey};
use axum::{
    extract::{Query, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::json;

use crate::collection_manager::sides::read::ReadError;
use crate::collection_manager::sides::write::WriteError;
use crate::types::IndexId;
use crate::{
    collection_manager::sides::{read::ReadSide, write::WriteSide},
    types::{CollectionId, WriteApiKey},
};

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/pin_rules/ids",
            get(list_pin_rules_ids_v1),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/pin_rules/insert",
            post(insert_pin_rule_v1),
        )
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/pin_rules/delete",
            post(delete_pin_rule_v1),
        )
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/pin_rules/list",
            get(list_pin_rule_v1),
        )
        .with_state(write_side)
}

async fn insert_pin_rule_v1(
    collection_id: CollectionId,
    index_id: IndexId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(pin_rule): Json<PinRule<String>>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.insert_pin_rule(index_id, pin_rule).await?;

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "success": true })))
}

async fn delete_pin_rule_v1(
    collection_id: CollectionId,
    index_id: IndexId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(DeletePinRuleParams { id }): Json<DeletePinRuleParams>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.delete_pin_rule(index_id, id).await?;

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "success": true })))
}

async fn list_pin_rule_v1(
    collection_id: CollectionId,
    index_id: IndexId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    let Some(index) = collection.get_index(index_id).await else {
        return Err(WriteError::IndexNotFound(collection_id, index_id));
    };

    let writer = index.get_pin_rule_writer().await;
    let list = writer.list_pin_rules();

    Ok(Json(json!({ "data": list })))
}

#[derive(Deserialize)]
struct ApiKeyQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn list_pin_rules_ids_v1(
    collection_id: CollectionId,
    index_id: IndexId,
    Query(ApiKeyQueryParams { api_key }): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let ids = read_side
        .list_pin_rule_ids(collection_id, index_id, api_key)
        .await?;

    Ok::<_, ReadError>(Json(json!({ "success": true, "data": ids })))
}

#[derive(Debug, Clone, Deserialize)]
struct DeletePinRuleParams {
    #[serde(rename = "pin_rule_id_to_delete")]
    id: String,
}
