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
        ApiKey, CollectionId, DeleteHookParams, GetHookQueryParams, IndexId, NewHookPostParams,
    },
    web_server::api::util::print_error,
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
    write_api_key: ApiKey,
    Json(params): Json<NewHookPostParams>,
) -> impl IntoResponse {
    let NewHookPostParams { name, code } = params;
    match write_side
        .insert_javascript_hook(write_api_key, collection_id, index_id, name, code)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            print_error(&e, "Error adding hook");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
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
    write_api_key: ApiKey,
    params: Query<GetHookQueryParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, impl IntoResponse)> {
    let GetHookQueryParams { name } = params.0;
    match write_side
        .get_javascript_hook(write_api_key, collection_id, index_id, name)
        .await
    {
        Ok(Some(full_hook)) => Ok(Json(json!({ "hook": full_hook.to_string() }))),
        Ok(None) => Ok(Json(json!({ "hook": null }))),
        Err(e) => {
            print_error(&e, "Error getting hook");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "DELETE",
    path = "/v1/{collection_id}/hooks/remove",
    description = "Delete an existing JavaScript hook"
)]
async fn delete_hook_v0(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
    Json(params): Json<DeleteHookParams>,
) -> impl IntoResponse {
    let name = params.name;

    match write_side
        .delete_javascript_hook(write_api_key, collection_id, name)
        .await
    {
        Ok(Some(_)) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Ok(None) => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "Unable to find hook to delete" })),
        )),
        Err(e) => {
            print_error(&e, "Error deleting hook");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/{collection_id}/hooks/list",
    description = "Get an existing JavaScript hook"
)]
async fn list_hooks_v0(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
) -> impl IntoResponse {
    match write_side
        .list_javascript_hooks(write_api_key, collection_id)
        .await
    {
        Ok(hooks) => Ok((StatusCode::OK, Json(json!(hooks)))),
        Err(e) => {
            print_error(&e, "Error listing hooks");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}
