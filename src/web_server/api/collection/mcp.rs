use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{post, put},
    Json, Router,
};

use crate::{
    collection_manager::sides::{
        read::ReadSide,
        write::{WriteError, WriteSide},
    },
    python::mcp::McpService,
    types::{
        CollectionId, CollectionStatsRequest, ReadApiKey, UpdateCollectionMcpRequest, WriteApiKey,
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

async fn mcp_endpoint(
    Path(collection_id): Path<CollectionId>,
    State(read_side): State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
    _headers: HeaderMap,
    body: Body,
) -> impl IntoResponse {
    if let Err(_err) = read_side
        .check_read_api_key(collection_id, &read_api_key)
        .await
    {
        let error_response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": null,
            "error": {
                "code": 401,
                "message": "unauthorized"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(error_response));
    }

    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            let error_response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32700,
                    "message": "Failed to read request body"
                }
            });
            return (StatusCode::BAD_REQUEST, Json(error_response));
        }
    };

    let request_str = match String::from_utf8(body_bytes.to_vec()) {
        Ok(s) => s,
        Err(_) => {
            let error_response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32700,
                    "message": "Invalid UTF-8 in request body"
                }
            });
            return (StatusCode::BAD_REQUEST, Json(error_response));
        }
    };

    let collection_info = read_side
        .collection_stats(
            &read_api_key,
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
        collection_id,
        read_api_key,
        collection_description,
    ) {
        Ok(service) => service,
        Err(e) => {
            let error_response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32603,
                    "message": format!("Failed to initialize MCP service: {}", e)
                }
            });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response));
        }
    };

    let response_str = match mcp_service.handle_jsonrpc(request_str) {
        Ok(response) => response,
        Err(e) => {
            let error_response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32603,
                    "message": format!("Internal error: {}", e)
                }
            });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response));
        }
    };

    match serde_json::from_str::<serde_json::Value>(&response_str) {
        Ok(response_json) => (StatusCode::OK, Json(response_json)),
        Err(e) => {
            let error_response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32603,
                    "message": format!("Failed to parse Python response: {}", e)
                }
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
        }
    }
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
