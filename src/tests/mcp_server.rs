use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    tests::utils::{init_log, TestContext},
    types::UpdateCollectionMcpRequest,
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
            collection_client.read_api_key,
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
            collection_client.read_api_key,
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
