use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_extra::{headers, TypedHeader};
use axum_openapi3::*;
use http::StatusCode;
use redact::Secret;
use serde_json::json;

use crate::{
    collection_manager::{
        dto::{ApiKey, DeleteHookParams, GetHookQueryParams, NewHookPostParams},
        sides::WriteSide,
    },
    types::CollectionId,
};

type AuthorizationBearerHeader =
    TypedHeader<headers::Authorization<headers::authorization::Bearer>>;

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
    path = "/v1/collections/{id}/hooks/create",
    description = "Add a new JavaScript hook"
)]
async fn add_hook_v0(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(params): Json<NewHookPostParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    let NewHookPostParams { name, code } = params;
    match write_side
        .insert_javascript_hook(write_api_key, collection_id, name, code)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            e.chain()
                .skip(1)
                .for_each(|cause| println!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/{collection_id}/hooks/get",
    description = "Get an existing JavaScript hook"
)]
async fn get_hook_v0(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    params: Query<GetHookQueryParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    let GetHookQueryParams { name } = params.0;
    match write_side
        .get_javascript_hook(write_api_key, collection_id, name)
        .await
    {
        Ok(Some(full_hook)) => Ok(Json(json!({ "hook": full_hook.to_string() }))),
        Ok(None) => Ok(Json(json!({ "hook": null }))),
        Err(e) => {
            e.chain()
                .skip(1)
                .for_each(|cause| println!("because: {}", cause));
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
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(params): Json<DeleteHookParams>,
) -> impl IntoResponse {
    let name = params.name;
    let collection_id = CollectionId(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

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
            e.chain()
                .skip(1)
                .for_each(|cause| println!("because: {}", cause));
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
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side
        .list_javascript_hooks(write_api_key, collection_id)
        .await
    {
        Ok(hooks) => Ok((StatusCode::OK, Json(json!(hooks)))),
        Err(e) => {
            e.chain()
                .skip(1)
                .for_each(|cause| println!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}
