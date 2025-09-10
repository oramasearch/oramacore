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
    ai::advanced_autoquery::QueryMappedSearchResult,
    collection_manager::sides::read::AnalyticSearchEventInvocationType,
    types::{HybridMode, Similarity, Threshold, VectorMode},
};

use crate::{
    collection_manager::sides::{
        read::{ReadError, ReadSide},
        write::{WriteError, WriteSide},
    },
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, NLPSearchRequest, SearchParams, SearchResult,
        UpdateCollectionMcpRequest, WriteApiKey,
    },
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

    let search_params_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "The search term to look for"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10
            },
            "mode": {
                "type": "string",
                "description": "Search mode. Can be 'fulltext', 'vector', or 'hybrid'. Use 'fulltext' for standard keyword search, 'vector' for semantic search, and 'hybrid' for a combination of both.",
                "default": "fulltext"
            }
        },
        "required": ["term"]
    });

    let nlp_search_params_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search with. Useful for complex queries that may not be easily expressed with keywords, or for queries that needs complex filtering, sorting, etc."
            }
        },
        "required": ["query"]
    });

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
        .unwrap_or("the collection");

    let search_description = format!(
        "Perform a full-text, vector, or hybrid search operation on {collection_description}"
    );
    let nlp_search_description = format!(
        "Perform complex search queries using natural language on {collection_description}"
    );

    let tools = serde_json::json!([
        {
            "name": "search",
            "description": search_description,
            "inputSchema": search_params_schema
        },
        {
            "name": "nlp_search",
            "description": nlp_search_description,
            "inputSchema": nlp_search_params_schema
        }
    ]);

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
                "tools": tools
            })
        }
        "tools/call" => {
            let server = StructuredOutputServer::new(read_side.clone(), api_key, collection_id);

            if let Some(params) = request.params.as_ref() {
                if let Some(tool_name) = params.get("name").and_then(|v| v.as_str()) {
                    match tool_name {
                        "search" => {
                            let search_params = if let Some(args) = params.get("arguments") {
                                debug!("Attempting to deserialize search arguments: {:?}", args);

                                // @todo: check if we can get rid of this simplified version of SearchParams
                                let search_params = if let (Some(term), limit, mode) = (
                                    args.get("term").and_then(|v| v.as_str()),
                                    args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10),
                                    args.get("mode").and_then(|v| v.as_str()),
                                ) {
                                    use crate::types::{
                                        FulltextMode, Limit, Properties, SearchMode, SearchOffset,
                                        SearchParams, WhereFilter,
                                    };
                                    use std::collections::HashMap;

                                    let search_mode = match mode {
                                        Some("vector") => SearchMode::Vector(VectorMode {
                                            term: term.to_string(),
                                            similarity: Similarity(0.6),
                                        }),
                                        Some("hybrid") => SearchMode::Hybrid(HybridMode {
                                            term: term.to_string(),
                                            similarity: Similarity(0.6),
                                            threshold: Some(Threshold(1.0)),
                                            exact: false,
                                            tolerance: None,
                                        }),
                                        _ => SearchMode::FullText(FulltextMode {
                                            term: term.to_string(),
                                            threshold: None,
                                            exact: false,
                                            tolerance: None,
                                        }),
                                    };

                                    SearchParams {
                                        mode: search_mode,
                                        limit: Limit(limit as usize),
                                        offset: SearchOffset(0),
                                        boost: HashMap::new(),
                                        properties: Properties::Star,
                                        where_filter: WhereFilter::default(),
                                        facets: HashMap::new(),
                                        indexes: None,
                                        sort_by: None,
                                        group_by: None,
                                        user_id: None,
                                    }
                                } else {
                                    // Try full SearchParams deserialization as fallback
                                    match serde_json::from_value(args.clone()) {
                                        Ok(params) => params,
                                        Err(err) => {
                                            error!("Failed to deserialize search parameters: {err}, args: {:?}", args);
                                            return (
                                                StatusCode::BAD_REQUEST,
                                                Json(JsonRpcResponse {
                                                    jsonrpc: "2.0".to_string(),
                                                    id: request.id,
                                                    result: None,
                                                    error: Some(JsonRpcError {
                                                        code: -32602,
                                                        message: format!(
                                                            "Invalid search parameters: {err}. Arguments received: {args:?}"
                                                        ),
                                                    }),
                                                }),
                                            );
                                        }
                                    }
                                };
                                search_params
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

    // For notifications (requests without id), don't send a response body
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
