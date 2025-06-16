use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{utoipa::IntoParams, *};
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;
use tracing::{info, warn};

use crate::{
    collection_manager::sides::{read::ReadSide, system_prompts::SystemPrompt, write::WriteSide},
    types::{
        ApiKey, CollectionId, DeleteSystemPromptParams, InsertSystemPromptParams,
        InteractionLLMConfig, UpdateSystemPromptParams,
    },
    web_server::api::util::print_error,
};

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
    path = "/v1/collections/{collection_id}/system_prompts/get",
    description = "Get a single system prompt by ID"
)]
async fn get_system_prompt_v1(
    collection_id: CollectionId,
    Query(query): Query<GetSystemPromptQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
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
        Err(e) => {
            print_error(&e, "Error getting system prompt");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/system_prompts/all",
    description = "Get all system prompts in a collection"
)]
async fn list_system_prompts_v1(
    collection_id: CollectionId,
    Query(query): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    match read_side
        .get_all_system_prompts_by_collection(read_api_key, collection_id)
        .await
    {
        Ok(system_prompts) => Ok((
            StatusCode::OK,
            Json(json!({ "system_prompts": system_prompts })),
        )),
        Err(e) => {
            print_error(&e, "Error getting all system prompts");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/system_prompts/validate",
    description = "Validate a system prompt"
)]
async fn validate_system_prompt_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(mut params): Json<InsertSystemPromptParams>,
) -> impl IntoResponse {
    if write_side.is_gpu_overloaded() {
        match write_side.select_random_remote_llm_service() {
            Some((provider, model)) => {
                info!("GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.", provider);
                params.llm_config = Some(InteractionLLMConfig { model, provider });
            }
            None => {
                warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
            }
        }
    }

    let system_prompt = SystemPrompt {
        id: params.id.clone().unwrap_or_else(cuid2::create_id),
        name: params.name.clone(),
        prompt: params.prompt.clone(),
        usage_mode: params.usage_mode.clone(),
    };

    write_side
        .validate_system_prompt(
            write_api_key,
            collection_id,
            system_prompt,
            params.llm_config,
        )
        .await
        .map(|result| Json(json!({ "result": result })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/system_prompts/insert",
    description = "Insert a new system prompt"
)]
async fn insert_system_prompt_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertSystemPromptParams>,
) -> impl IntoResponse {
    let system_prompt_id = params.id.unwrap_or(cuid2::create_id());

    let system_prompt = SystemPrompt {
        id: system_prompt_id,
        name: params.name.clone(),
        prompt: params.prompt.clone(),
        usage_mode: params.usage_mode.clone(),
    };

    write_side
        .insert_system_prompt(write_api_key, collection_id, system_prompt.clone())
        .await
        .map(|_| {
            Json(json!({ "success": true, "id": system_prompt.id, "system_prompt": system_prompt }))
        })
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/system_prompts/delete",
    description = "Deletes an existing system prompt"
)]
async fn delete_system_prompt_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteSystemPromptParams>,
) -> impl IntoResponse {
    write_side
        .delete_system_prompt(write_api_key, collection_id, params.id)
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/system_prompts/update",
    description = "Updates an existing system prompt"
)]
async fn update_system_prompt_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateSystemPromptParams>,
) -> impl IntoResponse {
    let system_prompt = SystemPrompt {
        id: params.id.clone(),
        name: params.name.clone(),
        prompt: params.prompt.clone(),
        usage_mode: params.usage_mode.clone(),
    };

    write_side
        .update_system_prompt(write_api_key, collection_id, system_prompt)
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}
