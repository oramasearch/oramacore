use std::collections::HashMap;

use anyhow::Context;
use futures::FutureExt;
use oramacore_lib::hook_storage::HookType;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    python::mcp::McpService,
    tests::utils::{init_log, wait_for, TestContext},
    types::{
        CollectionStatsRequest, CustomerClaims, DocumentList, ReadApiKey,
        UpdateCollectionMcpRequest,
    },
};

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    params: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[test]
fn test_jsonrpc_structures() {
    init_log();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: None,
    };

    let serialized = serde_json::to_string(&request).unwrap();
    assert!(serialized.contains("\"jsonrpc\":\"2.0\""));
    assert!(serialized.contains("\"method\":\"initialize\""));
    assert!(serialized.contains("\"id\":1"));

    let response_json = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05"
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(response_json).unwrap();
    assert_eq!(response.jsonrpc, "2.0");
    assert_eq!(response.id, Some(json!(1)));
    assert!(response.result.is_some());
    assert!(response.error.is_none());

    let error_response_json = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "error": {
            "code": -32601,
            "message": "Method not found"
        }
    });

    let error_response: JsonRpcResponse = serde_json::from_value(error_response_json).unwrap();
    assert_eq!(error_response.jsonrpc, "2.0");
    assert_eq!(error_response.id, Some(json!(2)));
    assert!(error_response.result.is_none());
    assert!(error_response.error.is_some());

    let error = error_response.error.unwrap();
    assert_eq!(error.code, -32601);
    assert_eq!(error.message, "Method not found");
}

#[test]
fn test_mcp_tool_schemas_generation() {
    init_log();

    let search_schema = schemars::schema_for!(crate::types::SearchParams);
    let nlp_schema = schemars::schema_for!(crate::types::NLPSearchRequest);

    let search_json = serde_json::to_value(search_schema).unwrap();
    let nlp_json = serde_json::to_value(nlp_schema).unwrap();

    assert!(search_json.is_object());
    assert!(nlp_json.is_object());

    assert!(search_json.get("type").is_some() || search_json.get("properties").is_some());
    assert!(nlp_json.get("type").is_some() || nlp_json.get("properties").is_some());
}

#[test]
fn test_mcp_tool_call_structures() {
    init_log();

    let search_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "search",
            "arguments": {
                "term": "rust programming",
                "limit": 10,
                "offset": 0
            }
        })),
    };

    let serialized = serde_json::to_string(&search_request).unwrap();
    assert!(serialized.contains("\"method\":\"tools/call\""));
    assert!(serialized.contains("\"name\":\"search\""));
    assert!(serialized.contains("\"term\":\"rust programming\""));

    let nlp_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "nlp_search",
            "arguments": {
                "query": "What is Rust programming language?",
                "limit": 5
            }
        })),
    };

    let serialized_nlp = serde_json::to_string(&nlp_request).unwrap();
    assert!(serialized_nlp.contains("\"name\":\"nlp_search\""));
    assert!(serialized_nlp.contains("\"query\":\"What is Rust programming language?\""));
}

#[test]
fn test_mcp_error_response_structures() {
    init_log();

    let auth_error = json!({
        "jsonrpc": "2.0",
        "id": null,
        "error": {
            "code": 401,
            "message": "unauthorized"
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(auth_error).unwrap();
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, 401);

    // Parse error (-32700)
    let parse_error = json!({
        "jsonrpc": "2.0",
        "id": null,
        "error": {
            "code": -32700,
            "message": "Parse error"
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(parse_error).unwrap();
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, -32700);

    let method_error = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32601,
            "message": "Method not found"
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(method_error).unwrap();
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, -32601);

    let param_error = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "error": {
            "code": -32602,
            "message": "Invalid search parameters: limit must be a number"
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(param_error).unwrap();
    assert!(response.error.is_some());
    let error = response.error.unwrap();
    assert_eq!(error.code, -32602);
    assert!(error.message.contains("Invalid search parameters"));
}

#[test]
fn test_mcp_success_response_structures() {
    init_log();

    let init_response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "orama-mcp",
                "version": "1.0.0"
            }
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(init_response).unwrap();
    assert!(response.result.is_some());
    let result = response.result.unwrap();
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert_eq!(result["serverInfo"]["name"], "orama-mcp");

    let tools_response = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": [
                {
                    "name": "search",
                    "description": "Perform a full-text, vector, or hybrid search operation",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "term": { "type": "string" },
                            "limit": { "type": "number" }
                        }
                    }
                }
            ]
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(tools_response).unwrap();
    assert!(response.result.is_some());
    let result = response.result.unwrap();
    assert!(result["tools"].is_array());
    let tools = result["tools"].as_array().unwrap();
    assert!(!tools.is_empty());
    assert_eq!(tools[0]["name"], "search");

    let tool_call_response = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": "{\"count\": 2, \"hits\": [{\"id\": \"1:doc1\", \"score\": 0.95}]}"
                }
            ]
        }
    });

    let response: JsonRpcResponse = serde_json::from_value(tool_call_response).unwrap();
    assert!(response.result.is_some());
    let result = response.result.unwrap();
    assert!(result["content"].is_array());
    let content = result["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert!(content[0]["text"].is_string());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_update_endpoint() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let update_request = UpdateCollectionMcpRequest {
        mcp_description: Some("Updated MCP server for testing collection operations".to_string()),
    };

    let result = collection_client
        .writer
        .update_collection_mcp_description(
            collection_client.write_api_key,
            collection_client.collection_id,
            update_request.mcp_description.clone(),
        )
        .await;

    assert!(
        result.is_ok(),
        "Failed to update MCP description: {result:?}"
    );

    test_context.commit_all().await.unwrap();

    let stats = collection_client
        .reader
        .collection_stats(
            &collection_client.read_api_key,
            collection_client.collection_id,
            crate::types::CollectionStatsRequest { with_keys: false },
        )
        .await
        .unwrap();

    assert_eq!(
        stats.mcp_description, update_request.mcp_description,
        "MCP description was not updated correctly"
    );

    let clear_request = UpdateCollectionMcpRequest {
        mcp_description: None,
    };

    let result = collection_client
        .writer
        .update_collection_mcp_description(
            collection_client.write_api_key,
            collection_client.collection_id,
            clear_request.mcp_description.clone(),
        )
        .await;

    assert!(
        result.is_ok(),
        "Failed to clear MCP description: {result:?}"
    );

    test_context.commit_all().await.unwrap();

    let stats = collection_client
        .reader
        .collection_stats(
            &collection_client.read_api_key,
            collection_client.collection_id,
            crate::types::CollectionStatsRequest { with_keys: false },
        )
        .await
        .unwrap();

    assert_eq!(
        stats.mcp_description, None,
        "MCP description was not cleared correctly"
    );
}

/// Test that MCP search with JWT claims passes the claims to the BeforeSearch hook.
/// This verifies the complete MCP + JWT integration:
/// 1. MCP service is created with ReadApiKey::Claims
/// 2. When search is called via MCP, claims are extracted
/// 3. BeforeSearch hook receives the claims
/// 4. Search results are filtered based on claims
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_search_with_jwt_claims() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents with different countries
    let documents: DocumentList = json!([
        {"id": "1", "name": "Document from US", "country": "US"},
        {"id": "2", "name": "Document from IT", "country": "IT"},
        {"id": "3", "name": "Document from UK", "country": "UK"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    // Insert BeforeSearch hook that filters by country from JWT claims
    collection_client
        .insert_hook(
            HookType::BeforeSearch,
            r#"
const beforeSearch = function(search_params, claim) {
    if (claim && claim.country) {
        search_params.where = search_params.where || {};
        search_params.where.country = claim.country;
        return search_params;
    }
}
export default { beforeSearch }
"#
            .to_string(),
        )
        .await
        .unwrap();

    // Wait for hook and documents
    let collection_id = collection_client.collection_id;
    let read_api_key = collection_client.read_api_key.clone();
    wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        let read_api_key = read_api_key.clone();
        async move {
            let stats = reader
                .collection_stats(
                    &read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("stats failed")?;

            if stats.hooks.is_empty() {
                return Err(anyhow::anyhow!("hooks not arrived yet"));
            }
            if stats.document_count != 3 {
                return Err(anyhow::anyhow!(
                    "documents not arrived yet: {}",
                    stats.document_count
                ));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    // Get the raw API key to use as orak
    let read_api_key_raw = match &collection_client.read_api_key {
        ReadApiKey::ApiKey(key) => *key,
        ReadApiKey::Claims(claims) => claims.orak,
    };

    // Create CustomerClaims with country="IT"
    let mut extra = HashMap::new();
    extra.insert("country".to_string(), Value::String("IT".to_string()));

    let claims_read_api_key = ReadApiKey::Claims(CustomerClaims {
        orak: read_api_key_raw,
        extra,
    });

    // Create MCP service with JWT claims
    let mcp_service = McpService::new(
        test_context.reader.clone(),
        collection_id,
        claims_read_api_key,
        "Test collection".to_string(),
    )
    .expect("Failed to create MCP service");

    // Call search via MCP JSON-RPC interface
    let jsonrpc_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {
                "term": "Document"
            }
        }
    });

    let response_str = mcp_service
        .handle_jsonrpc(jsonrpc_request.to_string())
        .expect("MCP JSON-RPC request failed");

    let response: Value =
        serde_json::from_str(&response_str).expect("Failed to parse JSON-RPC response");

    // Extract the result from JSON-RPC response
    let result = response
        .get("result")
        .expect("No result in JSON-RPC response");

    // Parse the result - it's wrapped in MCP content format
    // The result is: {"content": [{"type": "text", "text": "{...json...}"}]}
    let content = result
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("text"))
        .and_then(|t| t.as_str())
        .expect("Invalid MCP result format");

    let search_result: Value =
        serde_json::from_str(content).expect("Failed to parse search result JSON");

    // Should only get 1 document (the IT document) due to hook filtering
    let count = search_result
        .get("count")
        .and_then(|c| c.as_u64())
        .unwrap_or(0);
    assert_eq!(count, 1, "Expected 1 document, got {count}");

    // Verify it's the IT document
    let hits = search_result.get("hits").and_then(|h| h.as_array());
    assert!(hits.is_some(), "No hits in result");
    let hits = hits.unwrap();
    assert_eq!(hits.len(), 1, "Expected 1 hit");

    // The document ID format is "index_id:doc_id"
    let doc_id = hits[0]
        .get("id")
        .and_then(|id| id.as_str())
        .expect("No document ID");
    assert!(
        doc_id.ends_with(":2"),
        "Expected document 2 (IT), got {doc_id}"
    );

    drop(test_context);
}

/// Test that MCP search with plain API key passes null claims to the BeforeSearch hook.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_search_with_plain_api_key() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents
    let documents: DocumentList = json!([
        {"id": "1", "name": "Document A"},
        {"id": "2", "name": "Document B"},
        {"id": "3", "name": "Document C"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    // Insert BeforeSearch hook that limits results to 1 when claim is null
    collection_client
        .insert_hook(
            HookType::BeforeSearch,
            r#"
const beforeSearch = function(search_params, claim) {
    if (claim === null) {
        search_params.limit = 1;
        return search_params;
    }
}
export default { beforeSearch }
"#
            .to_string(),
        )
        .await
        .unwrap();

    // Wait for hook and documents
    let collection_id = collection_client.collection_id;
    let read_api_key = collection_client.read_api_key.clone();
    wait_for(&test_context, |t| {
        let reader = t.reader.clone();
        let read_api_key = read_api_key.clone();
        async move {
            let stats = reader
                .collection_stats(
                    &read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("stats failed")?;

            if stats.hooks.is_empty() {
                return Err(anyhow::anyhow!("hooks not arrived yet"));
            }
            if stats.document_count != 3 {
                return Err(anyhow::anyhow!(
                    "documents not arrived yet: {}",
                    stats.document_count
                ));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    // Create MCP service with plain API key (not JWT claims)
    let mcp_service = McpService::new(
        test_context.reader.clone(),
        collection_id,
        collection_client.read_api_key.clone(),
        "Test collection".to_string(),
    )
    .expect("Failed to create MCP service");

    // Call search via MCP JSON-RPC interface
    let jsonrpc_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {
                "term": "Document"
            }
        }
    });

    let response_str = mcp_service
        .handle_jsonrpc(jsonrpc_request.to_string())
        .expect("MCP JSON-RPC request failed");

    let response: Value =
        serde_json::from_str(&response_str).expect("Failed to parse JSON-RPC response");

    // Extract the result from JSON-RPC response
    let result = response
        .get("result")
        .expect("No result in JSON-RPC response");

    // Parse the result
    let content = result
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("text"))
        .and_then(|t| t.as_str())
        .expect("Invalid MCP result format");

    let search_result: Value =
        serde_json::from_str(content).expect("Failed to parse search result JSON");

    // Should only get 1 hit due to hook limiting when claim is null
    let hits = search_result
        .get("hits")
        .and_then(|h| h.as_array())
        .expect("No hits in result");
    assert_eq!(
        hits.len(),
        1,
        "Expected 1 hit (limited by hook), got {}",
        hits.len()
    );

    drop(test_context);
}
