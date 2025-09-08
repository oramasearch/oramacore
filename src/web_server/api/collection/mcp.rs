use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};

use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::{
    ai::advanced_autoquery::QueryMappedSearchResult,
    collection_manager::sides::read::SearchAnalyticEventOrigin,
};

use crate::{
    collection_manager::sides::read::{ReadError, ReadSide},
    types::{ApiKey, CollectionId, NLPSearchRequest, SearchParams, SearchResult},
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
                Some(SearchAnalyticEventOrigin::MCP),
            )
            .await
    }

    pub async fn perform_nlp_search(
        &self,
        nlp_request: NLPSearchRequest,
    ) -> Result<Vec<QueryMappedSearchResult>, ReadError> {
        let logs = self.read_side.get_hook_logs();
        let log_sender = logs.get_sender(&self.collection_id);

        self.read_side
            .nlp_search(
                axum::extract::State(self.read_side.clone()),
                self.read_api_key,
                self.collection_id,
                nlp_request,
                log_sender,
            )
            .await
    }
}

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/mcp", post(mcp_endpoint))
        .with_state(read_side)
}

#[derive(Deserialize)]
struct McpQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
    _jsonrpc: String,
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
            let search_params_schema = schemars::schema_for!(SearchParams);
            let nlp_search_params_schema = schemars::schema_for!(NLPSearchRequest);
            serde_json::json!({
                "tools": [
                    {
                        "name": "search",
                        "description": "Perform a full-text, vector, or hybrid search operation",
                        "inputSchema": search_params_schema
                    },
                    {
                        "name": "nlp_search",
                        "description": "Perform an advanced NLP search powered by AI",
                        "inputSchema": nlp_search_params_schema
                    }
                ]
            })
        }
        "tools/call" => {
            // Create MCP server instance and handle tool call
            let server = StructuredOutputServer::new(read_side.clone(), api_key, collection_id);

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
                                                        "Invalid search parameters: {err}"
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
                        "nlp_search" => {
                            // Extract NLP search parameters from the arguments
                            let nlp_params = if let Some(args) = params.get("arguments") {
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
                                                        "Invalid NLP search parameters: {err}"
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
                                            message: "Missing NLP search parameters".to_string(),
                                        }),
                                    }),
                                );
                            };

                            // Perform the NLP search
                            debug!("Performing NLP search with params: {:?}", nlp_params);
                            match server.perform_nlp_search(nlp_params).await {
                                Ok(result) => {
                                    debug!(
                                        "NLP search successful, result type: {}",
                                        std::any::type_name_of_val(&result)
                                    );
                                    match serde_json::to_string_pretty(&result) {
                                        Ok(json_str) => {
                                            debug!(
                                                "Successfully serialized NLP result, length: {}",
                                                json_str.len()
                                            );
                                            serde_json::json!({
                                                "content": [
                                                    {
                                                        "type": "text",
                                                        "text": json_str
                                                    }
                                                ]
                                            })
                                        }
                                        Err(serialize_err) => {
                                            error!(
                                                "Failed to serialize NLP search result: {}",
                                                serialize_err
                                            );
                                            serde_json::json!({
                                                "content": [
                                                    {
                                                        "type": "text",
                                                        "text": format!("NLP search completed but failed to serialize result: {}. Result debug: {:?}", serialize_err, result)
                                                    }
                                                ]
                                            })
                                        }
                                    }
                                }
                                Err(err) => {
                                    error!("NLP search failed: {}", err);
                                    serde_json::json!({
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": format!("NLP search error: {}. Error debug: {:?}", err, err)
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
