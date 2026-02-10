use std::{collections::BTreeMap, sync::Arc};

use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use oramacore_lib::hook_storage::HookType;
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::sides::write::{WriteError, WriteSide},
    types::{CollectionId, WriteApiKey},
};

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/hooks/set",
            post(set_hook_v0),
        )
        .route(
            "/v1/collections/{collection_id}/hooks/delete",
            post(delete_hook_v0),
        )
        .route(
            "/v1/collections/{collection_id}/hooks/list",
            get(list_hook_v0),
        )
        .with_state(write_side)
}

#[derive(Deserialize)]
pub struct NewHookPostParams {
    name: HookTypeWrapper,
    code: String,
}

async fn set_hook_v0(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<NewHookPostParams>,
) -> Result<impl IntoResponse, WriteError> {
    let NewHookPostParams { name, code } = params;

    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.set_hook(name.0, code).await?;

    Ok(Json(json!({ "success": true })))
}

#[derive(Deserialize)]
struct DeleteHookPostParams {
    name_to_delete: HookTypeWrapper,
}

async fn delete_hook_v0(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<DeleteHookPostParams>,
) -> Result<impl IntoResponse, WriteError> {
    let DeleteHookPostParams { name_to_delete } = params;

    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.delete_hook(name_to_delete.0).await?;

    Ok(Json(json!({ "success": true })))
}

async fn list_hook_v0(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> Result<impl IntoResponse, WriteError> {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    let hooks = collection.list_hooks()?;

    let output: BTreeMap<_, _> = hooks.into_iter().collect();

    Ok(Json(json!({ "hooks": output })))
}

#[derive(Deserialize)]
struct HookTypeWrapper(HookType);
