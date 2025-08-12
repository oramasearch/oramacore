use std::{future::Future, sync::Arc};

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{utoipa::ToSchema, *};
use rmcp::{
    handler::server::tool::{Parameters, ToolRouter},
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use utoipa::IntoParams;

use crate::{
    collection_manager::sides::read::ReadSide,
    types::{ApiKey, CollectionId, SearchParams},
};

#[derive(Clone)]
pub struct StructuredOutputServer {
    tool_router: ToolRouter<Self>,
}

#[tool_handler(router = self.tool_router)]
impl rmcp::ServerHandler for StructuredOutputServer {}

impl Default for StructuredOutputServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router(router = tool_router)]
impl StructuredOutputServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "search",
        description = "Perform a search operation on all the indexes"
    )]
    pub async fn search(&self, _params: Parameters<SearchParams>) -> String {
        "Search functionality not yet implemented".to_string()
    }
}

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new().add(mcp_endpoint()).with_state(read_side)
}

#[derive(Deserialize, IntoParams, ToSchema)]
struct McpQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/mcp",
    description = "MCP (Model Context Protocol) Endpoint"
)]
async fn mcp_endpoint(
    Path(_collection_id): Path<CollectionId>,
    Query(_query): Query<McpQueryParams>,
    _read_side: State<Arc<ReadSide>>,
    _headers: HeaderMap,
    body: Body,
) -> impl IntoResponse {
    // Parse request body
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            let error_response = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: "Parse error".to_string(),
                }),
            };
            return (StatusCode::BAD_REQUEST, Json(error_response));
        }
    };

    let request: JsonRpcRequest = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(_) => {
            let error_response = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: "Parse error".to_string(),
                }),
            };
            return (StatusCode::BAD_REQUEST, Json(error_response));
        }
    };

    // Handle different MCP methods
    let result = match request.method.as_str() {
        "initialize" => {
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "oramacore-mcp",
                    "version": "1.0.0"
                }
            })
        }
        "tools/list" => {
            serde_json::json!({
                "tools": [
                    {
                        "name": "search",
                        "description": "Perform a search operation on all the indexes",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                ]
            })
        }
        "tools/call" => {
            // Create MCP server instance and handle tool call
            let _server = StructuredOutputServer::new();
            serde_json::json!({
                "content": [
                    {
                        "type": "text",
                        "text": "Search functionality not yet implemented"
                    }
                ]
            })
        }
        _ => {
            let error_response = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: "Method not found".to_string(),
                }),
            };
            return (StatusCode::OK, Json(error_response));
        }
    };

    let response = JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id,
        result: Some(result),
        error: None,
    };

    (StatusCode::OK, Json(response))
}
