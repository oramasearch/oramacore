use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;
use tracing::{info, warn};

use crate::{
    ai::llms::LLMService,
    collection_manager::sides::{
        read::ReadSide,
        system_prompts::{CollectionSystemPromptsInterface, SystemPrompt},
        write::{WriteError, WriteSide},
    },
    types::{
        CollectionId, DeleteSystemPromptParams, InsertSystemPromptParams, InteractionLLMConfig,
        ReadApiKey, UpdateSystemPromptParams, WriteApiKey,
    },
    web_server::api::util::print_error,
};

#[derive(Deserialize)]
struct GetSystemPromptQueryParams {
    #[serde(rename = "system_prompt_id")]
    system_prompt_id: String,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/system_prompts/get",
            get(get_system_prompt_v1),
        )
        .route(
            "/v1/collections/{collection_id}/system_prompts/all",
            get(list_system_prompts_v1),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/system_prompts/insert",
            post(insert_system_prompt_v1),
        )
        .route(
            "/v1/collections/{collection_id}/system_prompts/delete",
            post(delete_system_prompt_v1),
        )
        .route(
            "/v1/collections/{collection_id}/system_prompts/update",
            post(update_system_prompt_v1),
        )
        .route(
            "/v1/collections/{collection_id}/system_prompts/validate",
            post(validate_system_prompt_v1),
        )
        .with_state(write_side)
}

async fn get_system_prompt_v1(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
    Query(query): Query<GetSystemPromptQueryParams>,
) -> impl IntoResponse {
    let system_prompt_id = query.system_prompt_id;

    match read_side
        .get_system_prompt(&read_api_key, collection_id, system_prompt_id)
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

async fn list_system_prompts_v1(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
) -> impl IntoResponse {
    match read_side
        .get_all_system_prompts_by_collection(&read_api_key, collection_id)
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

async fn validate_system_prompt_v1(
    Path(collection_id): Path<CollectionId>,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(mut params): Json<InsertSystemPromptParams>,
) -> impl IntoResponse {
    let llm_service = write_side.llm_service();
    if llm_service.is_gpu_overloaded() {
        match llm_service.select_random_remote_llm_service() {
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

    let system_prompts_manager = write_side
        .get_system_prompts_manager(write_api_key, collection_id)
        .await?;
    system_prompts_manager
        .validate_system_prompt(system_prompt, params.llm_config)
        .await
        .map(|result| Json(json!({ "result": result })))
}

async fn insert_system_prompt_v1(
    Path(collection_id): Path<CollectionId>,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertSystemPromptParams>,
) -> impl IntoResponse {
    let llm_config = handle_llm_config(write_side.llm_service(), params.llm_config.clone()).await;

    let system_prompts_manager = write_side
        .get_system_prompts_manager(write_api_key, collection_id)
        .await?;

    let system_prompt_id = params.id.unwrap_or(cuid2::create_id());

    let system_prompt = SystemPrompt {
        id: system_prompt_id,
        name: params.name,
        prompt: params.prompt,
        usage_mode: params.usage_mode,
    };

    raise_on_invalid_prompt(&system_prompts_manager, &system_prompt, llm_config).await?;

    system_prompts_manager
        .insert_system_prompt(system_prompt.clone())
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}

async fn delete_system_prompt_v1(
    Path(collection_id): Path<CollectionId>,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteSystemPromptParams>,
) -> impl IntoResponse {
    let system_prompts_manager = write_side
        .get_system_prompts_manager(write_api_key, collection_id)
        .await?;

    system_prompts_manager
        .delete_system_prompt(params.id)
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}

async fn update_system_prompt_v1(
    Path(collection_id): Path<CollectionId>,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateSystemPromptParams>,
) -> impl IntoResponse {
    let llm_config = handle_llm_config(write_side.llm_service(), params.llm_config.clone()).await;

    let system_prompt = SystemPrompt {
        id: params.id,
        name: params.name,
        prompt: params.prompt,
        usage_mode: params.usage_mode,
    };

    let system_prompts_manager = write_side
        .get_system_prompts_manager(write_api_key, collection_id)
        .await?;

    raise_on_invalid_prompt(&system_prompts_manager, &system_prompt, llm_config).await?;

    system_prompts_manager
        .update_system_prompt(system_prompt)
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}

async fn handle_llm_config(
    llm_service: &LLMService,
    llm_config: Option<InteractionLLMConfig>,
) -> Option<InteractionLLMConfig> {
    if !llm_service.is_gpu_overloaded() {
        return llm_config;
    }

    match llm_service.select_random_remote_llm_service() {
        Some((provider, model)) => {
            info!(
                "GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.",
                provider
            );
            Some(InteractionLLMConfig { model, provider })
        }
        None => {
            warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
            llm_config
        }
    }
}

async fn raise_on_invalid_prompt(
    system_prompts_manager: &CollectionSystemPromptsInterface,
    system_prompt: &SystemPrompt,
    llm_config: Option<InteractionLLMConfig>,
) -> Result<(), WriteError> {
    let validation_result = system_prompts_manager
        .validate_system_prompt(system_prompt.clone(), llm_config)
        .await?;

    let is_valid = validation_result.overall_assessment.valid
        && validation_result.security.valid
        && validation_result.technical.valid;

    if !is_valid {
        Err(WriteError::Generic(anyhow::anyhow!(
            "System prompt is invalid"
        )))
    } else {
        Ok(())
    }
}
