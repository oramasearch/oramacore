use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::*;
use http::StatusCode;
use serde_json::json;

use crate::{
    collection_manager::sides::write::WriteSide,
    types::{
        CollectionId, DeleteHookParams, GetHookQueryParams, IndexId, NewHookPostParams, WriteApiKey,
    },
};

pub fn apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(add_hook_v0())
        .add(get_hook_v0())
        .add(delete_hook_v0())
        .add(list_hooks_v0())
        .with_state(write_side)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/{index_id}/hooks/create",
    description = "Add a new JavaScript hook"
)]
async fn add_hook_v0(
    collection_id: CollectionId,
    index_id: IndexId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<NewHookPostParams>,
) -> impl IntoResponse {
    let NewHookPostParams { name, code } = params;

    let hooks_runtime = write_side
        .get_hooks_runtime(write_api_key, collection_id)
        .await?;

    hooks_runtime
        .insert_javascript_hook(index_id, name, code)
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}

#[endpoint(
    method = "GET",
    path = "/v1/{collection_id}/indexes/{index_id}/hooks/get",
    description = "Get an existing JavaScript hook"
)]
async fn get_hook_v0(
    collection_id: CollectionId,
    index_id: IndexId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    params: Query<GetHookQueryParams>,
) -> impl IntoResponse {
    let GetHookQueryParams { name } = params.0;

    let hooks_runtime = write_side
        .get_hooks_runtime(write_api_key, collection_id)
        .await?;

    hooks_runtime
        .get_javascript_hook(index_id, name)
        .await
        .map(|r| match r {
            Some(full_hook) => Json(json!({ "hook": full_hook.to_string() })),
            None => Json(json!({ "hook": null })),
        })
}

#[endpoint(
    method = "DELETE",
    path = "/v1/{collection_id}/hooks/remove",
    description = "Delete an existing JavaScript hook"
)]
async fn delete_hook_v0(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<DeleteHookParams>,
) -> impl IntoResponse {
    let name = params.name;

    let hooks_runtime = write_side
        .get_hooks_runtime(write_api_key, collection_id)
        .await?;

    hooks_runtime
        .delete_javascript_hook(name)
        .await
        .map(|_| Json(json!({ "success": true })))
}

#[endpoint(
    method = "GET",
    path = "/v1/{collection_id}/hooks/list",
    description = "Get an existing JavaScript hook"
)]
async fn list_hooks_v0(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    let hooks_runtime = write_side
        .get_hooks_runtime(write_api_key, collection_id)
        .await?;

    hooks_runtime
        .list_javascript_hooks()
        .await
        .map(|hooks| Json(json!(hooks)))
}
