use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_extra::{headers, TypedHeader};
use axum_openapi3::{utoipa::IntoParams, *};
use http::StatusCode;
use redact::Secret;
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::{
        dto::{
            ApiKey, DeleteSystemPromptParams, InsertSystemPromptParams, UpdateSystemPromptParams,
        },
        sides::{system_prompts::SystemPrompt, ReadSide, WriteSide},
    },
    types::CollectionId,
};

type AuthorizationBearerHeader =
    TypedHeader<headers::Authorization<headers::authorization::Bearer>>;

#[derive(Deserialize, IntoParams)]
struct ApiKeyQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize, IntoParams)]
struct GetSystemPromptQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
    #[serde(rename = "system_prompt_id")]
    system_prompt_id: String,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(get_system_prompt_v1())
        .add(list_system_prompts_v1())
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(insert_system_prompt_v1())
        .add(delete_system_prompt_v1())
        .add(update_system_prompt_v1())
        .add(validate_system_prompt_v1())
        .with_state(write_side)
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{id}/system_prompts/get",
    description = "Get a single system prompt by ID"
)]
async fn get_system_prompt_v1(
    Path(id): Path<String>,
    Query(query): Query<GetSystemPromptQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let collection_id = CollectionId::from(id);
    let system_prompt_id = query.system_prompt_id;
    let read_api_key = query.api_key;

    match read_side
        .get_system_prompt(read_api_key, collection_id, system_prompt_id)
        .await
    {
        Ok(Some(system_prompt)) => Ok((
            StatusCode::OK,
            Json(json!({ "system_prompt": system_prompt })),
        )),
        Ok(None) => Ok((StatusCode::OK, Json(json!({ "system_prompt": null })))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{id}/system_prompts/all",
    description = "Get all system prompts in a collection"
)]
async fn list_system_prompts_v1(
    Path(id): Path<String>,
    Query(query): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let collection_id = CollectionId::from(id);
    let read_api_key = query.api_key;

    match read_side
        .get_all_system_prompts_by_collection(read_api_key, collection_id)
        .await
    {
        Ok(system_prompts) => Ok((
            StatusCode::OK,
            Json(json!({ "system_prompts": system_prompts })),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/system_prompts/validate",
    description = "Validate a system prompt"
)]
async fn validate_system_prompt_v1(
    Path(id): Path<String>,
    TypedHeader(auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertSystemPromptParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId::from(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    let system_prompt = SystemPrompt {
        id: params.id.clone().unwrap_or_else(cuid2::create_id),
        name: params.name.clone(),
        prompt: params.prompt.clone(),
        usage_mode: params.usage_mode.clone(),
    };

    match write_side
        .validate_system_prompt(write_api_key, collection_id, system_prompt)
        .await
    {
        Ok(result) => Ok((StatusCode::OK, Json(json!({ "result": result })))),
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
    method = "POST",
    path = "/v1/collections/{id}/system_prompts/insert",
    description = "Insert a new system prompt"
)]
async fn insert_system_prompt_v1(
    Path(id): Path<String>,
    TypedHeader(auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertSystemPromptParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId::from(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    let system_prompt_id = params.id.unwrap_or(cuid2::create_id());

    let system_prompt = SystemPrompt {
        id: system_prompt_id,
        name: params.name.clone(),
        prompt: params.prompt.clone(),
        usage_mode: params.usage_mode.clone(),
    };

    match write_side
        .insert_system_prompt(write_api_key, collection_id, system_prompt.clone())
        .await
    {
        Ok(_) => Ok((
            StatusCode::OK,
            Json(
                json!({ "success": true, "id": system_prompt.id, "system_prompt": system_prompt }),
            ),
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
    method = "POST",
    path = "/v1/collections/{id}/system_prompts/delete",
    description = "Deletes an existing system prompt"
)]
async fn delete_system_prompt_v1(
    Path(id): Path<String>,
    TypedHeader(auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteSystemPromptParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId::from(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side
        .delete_system_prompt(write_api_key, collection_id, params.id)
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
    method = "POST",
    path = "/v1/collections/{id}/system_prompts/update",
    description = "Updates an existing system prompt"
)]
async fn update_system_prompt_v1(
    Path(id): Path<String>,
    TypedHeader(auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateSystemPromptParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId::from(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    let system_prompt = SystemPrompt {
        id: params.id.clone(),
        name: params.name.clone(),
        prompt: params.prompt.clone(),
        usage_mode: params.usage_mode.clone(),
    };

    match write_side
        .update_system_prompt(write_api_key, collection_id, system_prompt)
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
