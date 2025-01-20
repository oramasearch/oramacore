use std::{str::FromStr, sync::Arc};

use crate::collection_manager::sides::hooks::Hook;
use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::*;
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;
use utoipa::{IntoParams, ToSchema};

use crate::{
    collection_manager::{dto::NewHook, sides::WriteSide},
    types::CollectionId,
};

pub fn apis(write_side: Arc<WriteSide>) -> Router {
    Router::new().add(add_hook_v0()).with_state(write_side)
}

#[endpoint(
    method = "POST",
    path = "/v0/{collection_id}/hooks/add",
    description = "Add a new JavaScript hook"
)]
async fn add_hook_v0(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<NewHook>,
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

#[derive(Deserialize, IntoParams)]
struct GetHookQueryParams {
    pub name: String,
}

#[endpoint(
    method = "GET",
    path = "/v0/{collection_id}/hooks",
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
