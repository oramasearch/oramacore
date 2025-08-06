use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{utopia::IntoParams, *};
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::sides::{read::ReadSide, write::WriteSide},
    types::{ApiKey, CollectionId},
    web_server::api::util::print_error,
};

#[derive(Deserialize, IntoParams)]
struct ApiKeyQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new().add(execute_mcp_v1()).with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new().add(configure_mcp_v1()).with_state(write_side)
}

#[derive(Deserialize, ToSchema)]
struct MCPExecuteRequest {
    operation: String,
    parameters: serde_json::Value,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/mcp/execute",
    description = "Execute an MCP operation for a specific collection"
)]
async fn execute_mcp_v1(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<ApiKeyQueryParams>,
    Json(request): Json<MCPExecuteRequest>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let read_api_key = query.api_key;

    match read_side.get_collection(read_api_key, collection_id).await {
        Ok(Some(_)) => {
            // Execute MCP operation
            match execute_mcp_operation(&request.operation, &request.parameters).await {
                Ok(result) => Ok((StatusCode::OK, Json(result))),
                Err(e) => {
                    print_error(&e, "Error executing MCP operation");
                    Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": e.to_string() })),
                    ))
                }
            }
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Collection not found" })),
        )),
        Err(e) => {
            print_error(&e, "Error accessing collection");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[derive(Deserialize, ToSchema)]
struct MCPConfigRequest {
    config: serde_json::Value,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/mcp/configure",
    description = "Configure MCP settings for a specific collection"
)]
async fn configure_mcp_v1(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
    Json(request): Json<MCPConfigRequest>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    match write_side
        .configure_mcp(write_api_key, collection_id, request.config)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            print_error(&e, "Error configuring MCP");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

// @todo: move this to the read side
async fn execute_mcp_operation(
    operation: &str,
    parameters: &serde_json::Value,
) -> Result<serde_json::Value, anyhow::Error> {
    match operation {
        "search" => {
            // @todo: implement search operation
            Ok(json!({ "results": [] }))
        }
        "nlp_search" => {
            // @todo: implement nlp search operation
            Ok(json!({ "analysis": {} }))
        }
        _ => Err(anyhow::anyhow!("Unknown operation: {}", operation)),
    }
}
