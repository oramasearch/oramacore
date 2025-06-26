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

use crate::{
    ai::tools::{Tool, ToolError},
    collection_manager::sides::{read::ReadSide, write::WriteSide},
    types::{
        ApiKey, CollectionId, DeleteToolParams, InsertToolsParams, RunToolsParams,
        UpdateToolParams, WriteApiKey,
    },
};

#[derive(Deserialize, IntoParams)]
struct ApiKeyQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize, IntoParams)]
struct GetToolQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
    #[serde(rename = "tool_id")]
    tool_id: String,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(get_tool_v1())
        .add(get_all_tools_v1())
        .add(run_tools_v1())
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(insert_tool_v1())
        .add(delete_tool_v1())
        .add(update_tool_v1())
        .with_state(write_side)
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/tools/get",
    description = "Get a single tool by ID"
)]
async fn get_tool_v1(
    collection_id: CollectionId,
    Query(query): Query<GetToolQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> Result<impl IntoResponse, ToolError> {
    let tool_id = query.tool_id;
    let read_api_key = query.api_key;

    let tool_interface = read_side
        .get_tools_interface(read_api_key, collection_id)
        .await?;

    let j = match tool_interface.get_tool(tool_id).await? {
        None => Json(json!({ "tool": null })),
        Some(tool) => Json(json!({ "tool": tool })),
    };

    Ok(j)
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/tools/all",
    description = "Get all tools in a collection"
)]
async fn get_all_tools_v1(
    collection_id: CollectionId,
    Query(query): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> Result<impl IntoResponse, ToolError> {
    let read_api_key = query.api_key;

    let tool_interface = read_side
        .get_tools_interface(read_api_key, collection_id)
        .await?;

    let tools = tool_interface.get_all_tools_by_collection().await?;
    Ok(Json(json!({ "tools": tools })))
}

// #[axum::debug_handler]
#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/tools/run",
    description = "Run one or more tools"
)]
async fn run_tools_v1(
    collection_id: CollectionId,
    read_api_key: ApiKey,
    read_side: State<Arc<ReadSide>>,
    Json(params): Json<RunToolsParams>,
) -> Result<impl IntoResponse, ToolError> {
    let tool_interface = read_side
        .get_tools_interface(read_api_key, collection_id)
        .await?;

    let tools_result = tool_interface
        .execute_tools(params.messages, params.tool_ids, params.llm_config)
        .await?;

    Ok(Json(json!({ "results": tools_result })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/tools/insert",
    description = "Insert a new tool"
)]
async fn insert_tool_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertToolsParams>,
) -> impl IntoResponse {
    let tool = Tool {
        id: params.id,
        description: params.description,
        parameters: params.parameters,
        code: params.code,
    };

    let tool_interface = write_side
        .get_tools_manager(write_api_key, collection_id)
        .await?;

    tool_interface.insert_tool(tool.clone()).await.map(|_| {
        (
            StatusCode::CREATED,
            Json(json!({ "success": true, "id": tool.id, "tool": tool })),
        )
    })
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/tools/delete",
    description = "Deletes an existing tool"
)]
async fn delete_tool_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteToolParams>,
) -> impl IntoResponse {
    let tool_interface = write_side
        .get_tools_manager(write_api_key, collection_id)
        .await?;

    tool_interface
        .delete_tool(params.id)
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/tools/update",
    description = "Updates an existing tool"
)]
async fn update_tool_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateToolParams>,
) -> impl IntoResponse {
    let tool = Tool {
        id: params.id,
        description: params.description,
        parameters: params.parameters,
        code: params.code,
    };

    let tool_interface = write_side
        .get_tools_manager(write_api_key, collection_id)
        .await?;

    tool_interface
        .update_tool(tool)
        .await
        .map(|_| (StatusCode::OK, Json(json!({ "success": true }))))
}
