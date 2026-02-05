use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use orama_js_pool::{DomainPermission, Worker};
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

    // Validate the hook code before saving
    let hooks_config = write_side.get_hooks_config();
    let validation_result = Worker::builder()
        .with_domain_permission(DomainPermission::Allow(vec![]))
        .with_evaluation_timeout(Duration::from_millis(hooks_config.builder_timeout_ms))
        .add_module(name.0.get_function_name(), code.clone())
        .build()
        .await;

    if let Err(e) = validation_result {
        return Err(WriteError::Generic(anyhow::anyhow!(
            "Hook validation failed: {e}"
        )));
    }

    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    // Insert hook into storage and update the JS pool
    collection
        .get_hook_storage()
        .insert_hook(name.0, code.clone())
        .await?;
    collection
        .update_hook_in_pool(name.0, code)
        .await
        .map_err(|e| WriteError::Generic(e))?;

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

    // Delete hook from storage and remove from JS pool
    collection
        .get_hook_storage()
        .delete_hook(name_to_delete.0)
        .await?;
    collection
        .remove_hook_from_pool(name_to_delete.0)
        .await
        .map_err(|e| WriteError::Generic(e))?;

    Ok(Json(json!({ "success": true })))
}

async fn list_hook_v0(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> Result<impl IntoResponse, WriteError> {
    let hook = write_side
        .get_hooks_storage(write_api_key, collection_id)
        .await?;
    let hooks = hook.list_hooks()?;

    let output: HashMap<_, _> = hooks.into_iter().collect();

    Ok(Json(json!({ "hooks": output })))
}

#[derive(Deserialize)]
struct HookTypeWrapper(HookType);
