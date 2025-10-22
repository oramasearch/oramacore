use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{post, put},
    Json, Router,
};

use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::{
    collection_manager::sides::{
        read::ReadSide,
        write::{WriteError, WriteSide},
    },
    python::mcp::McpService,
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, UpdateCollectionMcpRequest, WriteApiKey,
    },
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/mcp", post(mcp_endpoint))
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/mcp/update",
            put(update_mcp_endpoint),
        )
        .with_state(write_side)
}

#[derive(Deserialize)]
struct McpQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
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

async fn mcp_endpoint(
    Path(collection_id): Path<CollectionId>,
    Query(query): Query<McpQueryParams>,
    State(read_side): State<Arc<ReadSide>>,
    _headers: HeaderMap,
    body: Body,
) -> impl IntoResponse {
    let api_key = query.api_key;

    match read_side.check_read_api_key(collection_id, api_key).await {
        Ok(_) => {}
        Err(_err) => {
            let error_response = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code: 401,
                    message: "unauthorized".to_string(),
                }),
            };
            return (StatusCode::UNAUTHORIZED, Json(error_response));
        }
    }

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

    if request.jsonrpc != "2.0" {
        let error_response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "Invalid Request. JSON-RPC version must be 2.0".to_string(),
            }),
        };
        return (StatusCode::BAD_REQUEST, Json(error_response));
    }

    let collection_info = read_side
        .collection_stats(
            api_key,
            collection_id,
            CollectionStatsRequest { with_keys: false },
        )
        .await
        .ok();

    let collection_description = collection_info
        .as_ref()
        .and_then(|stats| stats.mcp_description.as_ref())
        .map(String::as_str)
        .unwrap_or("the collection")
        .to_string();

    let mcp_service = match McpService::new(
        read_side.clone(),
        api_key,
        collection_id,
        collection_description,
    ) {
        Ok(service) => service,
        Err(e) => {
            error!("Failed to create MCP service: {}", e);
            let error_response = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: format!("Internal error: Failed to initialize MCP service: {}", e),
                }),
            };
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response));
        }
    };

    let result = match request.method.as_str() {
        "initialize" => {
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "orama-mcp",
                    "version": "1.0.0"
                }
            })
        }
        "tools/list" => {
            // Delegate to Python MCP to get the list of tools
            match mcp_service.list_tools() {
                Ok(tools_list) => {
                    debug!("Successfully retrieved tools list from Python MCP");
                    serde_json::json!({
                        "tools": tools_list
                    })
                }
                Err(e) => {
                    error!("Failed to list tools from Python MCP: {}", e);
                    let error_response = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32603,
                            message: format!("Failed to list tools: {}", e),
                        }),
                    };
                    return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response));
                }
            }
        }
        "tools/call" => {
            if let Some(params) = request.params.as_ref() {
                if let Some(tool_name) = params.get("name").and_then(|v| v.as_str()) {
                    let arguments = params
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::json!({}));

                    debug!(
                        "Calling Python MCP tool '{}' with arguments: {:?}",
                        tool_name, arguments
                    );

                    match mcp_service.call_tool(tool_name, arguments) {
                        Ok(result) => {
                            debug!("Successfully executed tool '{}' via Python MCP", tool_name);
                            serde_json::json!({
                                "content": [
                                    {
                                        "type": "text",
                                        "text": serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
                                    }
                                ]
                            })
                        }
                        Err(e) => {
                            error!("Failed to execute tool '{}': {}", tool_name, e);
                            serde_json::json!({
                                "content": [
                                    {
                                        "type": "text",
                                        "text": format!("Tool execution error: {}", e)
                                    }
                                ]
                            })
                        }
                    }
                } else {
                    let error_response = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32602,
                            message: "Missing tool name in parameters".to_string(),
                        }),
                    };
                    return (StatusCode::BAD_REQUEST, Json(error_response));
                }
            } else {
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Missing parameters for tool call".to_string(),
                    }),
                };
                return (StatusCode::BAD_REQUEST, Json(error_response));
            }
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

    if request.id.is_none() {
        return (
            StatusCode::NO_CONTENT,
            Json(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: None,
            }),
        );
    }

    let response = JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id,
        result: Some(result),
        error: None,
    };

    (StatusCode::OK, Json(response))
}

async fn update_mcp_endpoint(
    Path(collection_id): Path<CollectionId>,
    State(write_side): State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(request): Json<UpdateCollectionMcpRequest>,
) -> impl IntoResponse {
    match write_side
        .update_collection_mcp_description(write_api_key, collection_id, request.mcp_description)
        .await
    {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "message": "MCP description updated successfully" })),
        ),
        Err(WriteError::CollectionNotFound(_)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Collection not found" })),
        ),
        Err(WriteError::InvalidWriteApiKey(_)) => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Unauthorized" })),
        ),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Internal server error: {}", err) })),
        ),
    }
}
