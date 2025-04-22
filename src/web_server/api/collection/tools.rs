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
    ai::tools::Tool,
    collection_manager::sides::{ReadSide, WriteSide},
    types::{
        ApiKey, CollectionId, DeleteToolParams, InsertToolsParams, RunToolsParams, UpdateToolParams,
    },
    web_server::api::collection::admin::print_error,
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
) -> impl IntoResponse {
    let tool_id = query.tool_id;
    let read_api_key = query.api_key;

    match read_side
        .get_tool(read_api_key, collection_id, tool_id)
        .await
    {
        Ok(Some(tool)) => Ok((StatusCode::OK, Json(json!({ "tool": tool })))),
        Ok(None) => Ok((StatusCode::OK, Json(json!({ "tool": null })))),
        Err(e) => {
            print_error(&e, "Error getting tool");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
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
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    match read_side
        .get_all_tools_by_collection(read_api_key, collection_id)
        .await
    {
        Ok(tools) => Ok((StatusCode::OK, Json(json!({ "tools": tools })))),
        Err(e) => {
            print_error(&e, "Error getting all tools");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/tools/insert",
    description = "Insert a new tool"
)]
async fn insert_tool_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertToolsParams>,
) -> impl IntoResponse {
    let tool = Tool {
        id: params.id,
        description: params.description,
        parameters: params.parameters,
        code: params.code,
    };

    match write_side
        .insert_tool(write_api_key, collection_id, tool.clone())
        .await
    {
        Ok(_) => Ok((
            StatusCode::CREATED,
            Json(json!({ "success": true, "id": tool.id, "tool": tool })),
        )),
        Err(e) => {
            print_error(&e, "Error inserting tool");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/tools/delete",
    description = "Deletes an existing tool"
)]
async fn delete_tool_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteToolParams>,
) -> impl IntoResponse {
    match write_side
        .delete_tool(write_api_key, collection_id, params.id)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            print_error(&e, "Error deleting tool");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/tools/update",
    description = "Updates an existing tool"
)]
async fn update_tool_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateToolParams>,
) -> impl IntoResponse {
    let tool = Tool {
        id: params.id,
        description: params.description,
        parameters: params.parameters,
        code: params.code,
    };

    match write_side
        .update_tool(write_api_key, collection_id, tool)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            print_error(&e, "Error updating tool");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

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
) -> impl IntoResponse {
    let tools_result = read_side
        .execute_tools(
            read_api_key,
            collection_id,
            params.messages,
            params.tool_ids,
            params.llm_config,
        )
        .await;

    match tools_result {
        Ok(results) => Ok((StatusCode::OK, Json(json!({ "results": results })))),
        Err(e) => {
            print_error(&e, "Error running tools");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}
