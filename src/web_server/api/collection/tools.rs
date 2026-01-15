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

use crate::{
    ai::tools::{Tool, ToolError},
    collection_manager::sides::{read::ReadSide, write::WriteSide},
    types::{
        CollectionId, DeleteToolParams, InsertToolsParams, ReadApiKey, RunToolsParams,
        UpdateToolParams, WriteApiKey,
    },
};

#[derive(Deserialize)]
struct GetToolQueryParams {
    #[serde(rename = "tool_id")]
    tool_id: String,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/tools/get",
            get(get_tool_v1),
        )
        .route(
            "/v1/collections/{collection_id}/tools/all",
            get(get_all_tools_v1),
        )
        .route(
            "/v1/collections/{collection_id}/tools/run",
            post(run_tools_v1),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/tools/insert",
            post(insert_tool_v1),
        )
        .route(
            "/v1/collections/{collection_id}/tools/delete",
            post(delete_tool_v1),
        )
        .route(
            "/v1/collections/{collection_id}/tools/update",
            post(update_tool_v1),
        )
        .with_state(write_side)
}

async fn get_tool_v1(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
    Query(query): Query<GetToolQueryParams>,
) -> Result<impl IntoResponse, ToolError> {
    let tool_id = query.tool_id;

    let tool_interface = read_side
        .get_tools_interface(read_api_key, collection_id)
        .await?;

    let j = match tool_interface.get_tool(tool_id).await? {
        None => Json(json!({ "tool": null })),
        Some(tool) => Json(json!({ "tool": tool })),
    };

    Ok(j)
}

async fn get_all_tools_v1(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
) -> Result<impl IntoResponse, ToolError> {
    let tool_interface = read_side
        .get_tools_interface(read_api_key, collection_id)
        .await?;

    let tools = tool_interface.get_all_tools_by_collection().await?;
    Ok(Json(json!({ "tools": tools })))
}

// #[axum::debug_handler]
async fn run_tools_v1(
    Path(collection_id): Path<CollectionId>,
    read_api_key: ReadApiKey,
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

async fn insert_tool_v1(
    Path(collection_id): Path<CollectionId>,
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

async fn delete_tool_v1(
    Path(collection_id): Path<CollectionId>,
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

async fn update_tool_v1(
    Path(collection_id): Path<CollectionId>,
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
