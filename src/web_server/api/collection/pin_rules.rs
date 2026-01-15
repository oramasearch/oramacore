use std::sync::Arc;

use crate::types::ReadApiKey;
use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use oramacore_lib::pin_rules::PinRule;
use serde::Deserialize;
use serde_json::json;

use crate::collection_manager::sides::read::ReadError;
use crate::collection_manager::sides::write::WriteError;
use crate::{
    collection_manager::sides::{read::ReadSide, write::WriteSide},
    types::{CollectionId, WriteApiKey},
};

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/merchandising/pin_rules/ids",
            get(list_merchandising_pin_rules_ids_v1),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/merchandising/pin_rules/insert",
            post(insert_merchandising_pin_rules_v1),
        )
        .route(
            "/v1/collections/{collection_id}/merchandising/pin_rules/delete",
            post(delete_merchandising_pin_rules_v1),
        )
        .route(
            "/v1/collections/{collection_id}/merchandising/pin_rules/list",
            get(list_merchandising_pin_rules_v1),
        )
        .with_state(write_side)
}

async fn list_merchandising_pin_rules_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    let pin_rules_writer = collection.get_pin_rule_writer("list").await;

    let list = pin_rules_writer.list_pin_rules();

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "data": list })))
}

async fn insert_merchandising_pin_rules_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(pin_rule): Json<PinRule<String>>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.insert_merchandising_pin_rule(pin_rule).await?;

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "success": true })))
}

async fn delete_merchandising_pin_rules_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(DeletePinRuleParams { id }): Json<DeletePinRuleParams>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.delete_merchandising_pin_rule(id).await?;

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "success": true })))
}

async fn list_merchandising_pin_rules_ids_v1(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let collection = read_side
        .get_collection(collection_id, &read_api_key)
        .await?;

    let rules = collection.get_pin_rules_reader("list").await;
    let ids = rules.get_rule_ids();

    Ok::<_, ReadError>(Json(json!({ "success": true, "data": ids })))
}

#[derive(Debug, Clone, Deserialize)]
struct DeletePinRuleParams {
    #[serde(rename = "pin_rule_id_to_delete")]
    id: String,
}
