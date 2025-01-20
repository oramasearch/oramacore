use std::{str::FromStr, sync::Arc};

use crate::collection_manager::sides::hooks::{Hook, HookValue};
use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::*;
use http::StatusCode;
use serde_json::json;
use std::collections::HashMap;

use crate::{
    collection_manager::{
        dto::DeleteHookParams, dto::GetHookQueryParams, dto::NewHookPostParams, sides::WriteSide,
    },
    types::CollectionId,
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
    path = "/v0/{collection_id}/hooks/add",
    description = "Add a new JavaScript hook"
)]
async fn add_hook_v0(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<NewHookPostParams>,
) -> impl IntoResponse {
    let name = params.name;
    let code: String = params.code;
    let collection_id = CollectionId(id);
    let hook = match Hook::from_str(&name) {
        Ok(hook_name) => hook_name,
        Err(e) => {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    };

    match write_side
        .insert_javascript_hook(collection_id, hook, code)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[endpoint(
    method = "GET",
    path = "/v0/{collection_id}/hooks/get",
    description = "Get an existing JavaScript hook"
)]
async fn get_hook_v0(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    params: Query<GetHookQueryParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let name = params.name.clone();
    let hook = match Hook::from_str(&name) {
        Ok(hook_name) => hook_name,
        Err(e) => {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    };

    match write_side.get_javascript_hook(collection_id, hook).await {
        Some(full_hook) => Ok((
            StatusCode::OK,
            // @todo: handle unwrap
            Json(json!({ "hook": full_hook.to_string().unwrap() })),
        )),
        None => Ok((StatusCode::OK, Json(json!({ "hook": null })))),
    }
}

#[endpoint(
    method = "DELETE",
    path = "/v0/{collection_id}/hooks/remove",
    description = "Delete an existing JavaScript hook"
)]
async fn delete_hook_v0(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteHookParams>,
) -> impl IntoResponse {
    let name = params.name;
    let collection_id = CollectionId(id);

    let hook = match Hook::from_str(&name) {
        Ok(hook_name) => hook_name,
        Err(e) => {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    };

    match write_side.delete_javascript_hook(collection_id, hook).await {
        Some(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        None => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "Unable to find hook to delete" })),
        )),
    }
}

#[endpoint(
    method = "GET",
    path = "/v0/{collection_id}/hooks/list",
    description = "Get an existing JavaScript hook"
)]
async fn list_hooks_v0(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let hooks = write_side.list_javascript_hooks(collection_id).await;

    match serde_json::to_string(&serialize_hook_values(hooks)) {
        Ok(hooks_json) => Ok((StatusCode::OK, Json(json!({ "hooks": hooks_json })))),
        Err(e) => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

fn serialize_hook_values(values: Vec<(String, HookValue)>) -> Vec<HashMap<String, HookValue>> {
    values
        .into_iter()
        .map(|(key, value)| {
            let mut map = HashMap::new();
            map.insert(key, value);
            map
        })
        .collect()
}
