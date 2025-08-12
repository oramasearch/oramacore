use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{utoipa::ToSchema, *};

use serde::{Deserialize, Serialize};
use utoipa::IntoParams;

use crate::collection_manager::sides::read::AnalyticSearchEventInvocationType;

use crate::{
    collection_manager::sides::read::{ReadError, ReadSide},
    types::{ApiKey, CollectionId, SearchParams, SearchResult},
};

#[derive(Clone)]
pub struct StructuredOutputServer {
    read_side: Arc<ReadSide>,
    read_api_key: ApiKey,
    collection_id: CollectionId,
}

impl StructuredOutputServer {
    pub fn new(
        read_side: Arc<ReadSide>,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Self {
        Self {
            read_side,
            read_api_key,
            collection_id,
        }
    }

    pub async fn perform_search(
        &self,
        search_params: SearchParams,
    ) -> Result<SearchResult, ReadError> {
        self.read_side
            .search(
                self.read_api_key,
                self.collection_id,
                search_params,
                AnalyticSearchEventInvocationType::Action,
            )
            .await
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
    Path(collection_id): Path<CollectionId>,
    Query(query): Query<McpQueryParams>,
    State(read_side): State<Arc<ReadSide>>,
    _headers: HeaderMap,
    body: Body,
) -> impl IntoResponse {
    let api_key = query.api_key;

    match read_side
        .check_read_api_key(collection_id, api_key.clone())
        .await
    {
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

    // Handle different MCP methods
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
            serde_json::json!({
                "tools": [
                    {
                        "name": "search",
                        "description": "Perform a search operation on all the indexes",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "term": {
                                    "type": "string",
                                    "description": "Search term"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results",
                                    "default": 10
                                }
                            },
                            "required": []
                        }
                    }
                ]
            })
        }
        "tools/call" => {
            // Create MCP server instance and handle tool call
            let server = StructuredOutputServer::new(
                read_side.clone(),
                api_key.clone(),
                collection_id.clone(),
            );

            // Extract tool name and arguments from request params
            if let Some(params) = request.params.as_ref() {
                if let Some(tool_name) = params.get("name").and_then(|v| v.as_str()) {
                    match tool_name {
                        "search" => {
                            // Extract search parameters from the arguments
                            let search_params = if let Some(args) = params.get("arguments") {
                                match serde_json::from_value(args.clone()) {
                                    Ok(params) => params,
                                    Err(err) => {
                                        return (
                                            StatusCode::BAD_REQUEST,
                                            Json(JsonRpcResponse {
                                                jsonrpc: "2.0".to_string(),
                                                id: request.id,
                                                result: None,
                                                error: Some(JsonRpcError {
                                                    code: -32602,
                                                    message: format!(
                                                        "Invalid search parameters: {}",
                                                        err
                                                    ),
                                                }),
                                            }),
                                        );
                                    }
                                }
                            } else {
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(JsonRpcResponse {
                                        jsonrpc: "2.0".to_string(),
                                        id: request.id,
                                        result: None,
                                        error: Some(JsonRpcError {
                                            code: -32602,
                                            message: "Missing search parameters".to_string(),
                                        }),
                                    }),
                                );
                            };

                            // Perform the search
                            match server.perform_search(search_params).await {
                                Ok(result) => {
                                    serde_json::json!({
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                                            }
                                        ]
                                    })
                                }
                                Err(err) => {
                                    serde_json::json!({
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": format!("Search error: {}", err)
                                            }
                                        ]
                                    })
                                }
                            }
                        }
                        _ => {
                            serde_json::json!({
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Unknown tool"
                                    }
                                ]
                            })
                        }
                    }
                } else {
                    serde_json::json!({
                        "content": [
                            {
                                "type": "text",
                                "text": "Missing tool name"
                            }
                        ]
                    })
                }
            } else {
                serde_json::json!({
                    "content": [
                        {
                            "type": "text",
                            "text": "Missing parameters"
                        }
                    ]
                })
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

    let response = JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id,
        result: Some(result),
        error: None,
    };

    (StatusCode::OK, Json(response))
}
