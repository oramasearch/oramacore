use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener};

use anyhow::Context;
use axum::{routing::get, routing::post, Json, Router};
use chrono::Utc;
use futures::FutureExt;
use jsonwebtoken::{encode, EncodingKey, Header};
use oramacore_lib::hook_storage::HookType;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::auth::{JwksProviderConfig, JwtConfig, JwtManager};
use crate::collection_manager::sides::read::SearchRequest;
use crate::tests::utils::{extrapolate_ids_from_result, init_log, wait_for, TestContext};
use crate::types::{
    ApiKey, CollectionStatsRequest, CustomerClaims, DocumentList, ReadApiKey, SearchParams,
    StackString,
};

/// Helper to create SearchParams from JSON
fn search_params_from_json(json: serde_json::Value) -> SearchParams {
    json.try_into().expect("Invalid search params JSON")
}

/// Test that the BeforeSearch hook receives the JWT claims and can modify search params based on them.
#[tokio::test(flavor = "multi_thread")]
async fn test_before_search_hook_receives_claims() {
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

    // Insert a BeforeSearch hook that filters by country from JWT claims
    // The hook signature is: function beforeSearch(search_params, claim)
    collection_client
        .insert_hook(
            HookType::BeforeSearch,
            r#"
const beforeSearch = function(search_params, claim) {
    // If claim has a country, filter to only show documents from that country
    if (claim && claim.country) {
        search_params.where = search_params.where || {};
        search_params.where.country = claim.country;
        return search_params;
    }
    // No modification if no country in claim
}
export default { beforeSearch }
"#
            .to_string(),
        )
        .await
        .unwrap();

    // Wait for hook to be registered
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

    // Create CustomerClaims with country="IT"
    let read_api_key_raw = match &collection_client.read_api_key {
        ReadApiKey::ApiKey(key) => *key,
        ReadApiKey::Claims(claims) => claims.orak,
    };

    let mut extra = HashMap::new();
    extra.insert("country".to_string(), Value::String("IT".to_string()));

    let claims_read_api_key = ReadApiKey::Claims(CustomerClaims {
        orak: read_api_key_raw,
        extra,
    });

    // Search with JWT claims containing country="IT"
    // The hook should filter to only return IT documents
    let search_params = search_params_from_json(json!({
        "term": "Document"
    }));

    let result = test_context
        .reader
        .search(
            &claims_read_api_key,
            collection_id,
            SearchRequest {
                search_params,
                analytics_metadata: None,
                interaction_id: None,
                search_analytics_event_origin: None,
            },
        )
        .await
        .unwrap();

    // Should only get the IT document
    assert_eq!(result.count, 1, "Expected 1 document, got {}", result.count);
    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(
        ids,
        vec!["2"],
        "Expected document with id='2' (IT document)"
    );

    drop(test_context);
}

/// Test that the BeforeSearch hook receives null claim when using plain API key.
#[tokio::test(flavor = "multi_thread")]
async fn test_before_search_hook_with_plain_api_key() {
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

    // Insert a BeforeSearch hook that checks if claim is null
    // and modifies search params differently
    collection_client
        .insert_hook(
            HookType::BeforeSearch,
            r#"
const beforeSearch = function(search_params, claim) {
    // When using plain API key, claim should be null
    // In this case, we limit results to 1
    if (claim === null) {
        search_params.limit = 1;
        return search_params;
    }
    // With claims, return all results
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

    // Search with plain API key (no claims)
    // The hook should limit results to 1
    let search_params = search_params_from_json(json!({
        "term": "Document"
    }));

    let result = collection_client.search(search_params).await.unwrap();

    // Should get only 1 document due to hook limiting
    assert_eq!(
        result.hits.len(),
        1,
        "Expected 1 hit (limited by hook), got {}",
        result.hits.len()
    );

    drop(test_context);
}

/// Test that the BeforeSearch hook can access arbitrary extra claims from JWT.
#[tokio::test(flavor = "multi_thread")]
async fn test_before_search_hook_with_multiple_claims() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents with different categories and regions
    let documents: DocumentList = json!([
        {"id": "1", "name": "Product A", "category": "electronics", "region": "north"},
        {"id": "2", "name": "Product B", "category": "electronics", "region": "south"},
        {"id": "3", "name": "Product C", "category": "clothing", "region": "north"},
        {"id": "4", "name": "Product D", "category": "clothing", "region": "south"}
    ])
    .try_into()
    .unwrap();
    index_client.insert_documents(documents).await.unwrap();

    // Insert a BeforeSearch hook that uses multiple claims
    collection_client
        .insert_hook(
            HookType::BeforeSearch,
            r#"
const beforeSearch = function(search_params, claim) {
    if (claim) {
        search_params.where = search_params.where || {};
        // Apply category filter if present in claims
        if (claim.allowed_category) {
            search_params.where.category = claim.allowed_category;
        }
        // Apply region filter if present in claims
        if (claim.allowed_region) {
            search_params.where.region = claim.allowed_region;
        }
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
            if stats.document_count != 4 {
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

    // Create claims with category="electronics" and region="north"
    let read_api_key_raw = match &collection_client.read_api_key {
        ReadApiKey::ApiKey(key) => *key,
        ReadApiKey::Claims(claims) => claims.orak,
    };

    let mut extra = HashMap::new();
    extra.insert(
        "allowed_category".to_string(),
        Value::String("electronics".to_string()),
    );
    extra.insert(
        "allowed_region".to_string(),
        Value::String("north".to_string()),
    );

    let claims_read_api_key = ReadApiKey::Claims(CustomerClaims {
        orak: read_api_key_raw,
        extra,
    });

    // Search with claims
    let search_params = search_params_from_json(json!({
        "term": "Product"
    }));

    let result = test_context
        .reader
        .search(
            &claims_read_api_key,
            collection_id,
            SearchRequest {
                search_params,
                analytics_metadata: None,
                interaction_id: None,
                search_analytics_event_origin: None,
            },
        )
        .await
        .unwrap();

    // Should only get document 1: electronics + north
    assert_eq!(result.count, 1, "Expected 1 document, got {}", result.count);
    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(
        ids,
        vec!["1"],
        "Expected document with id='1' (electronics + north)"
    );

    drop(test_context);
}

/// Test that invalid orak in CustomerClaims is rejected.
#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_orak_in_claims_rejected() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert a document
    let documents: DocumentList = json!([{"id": "1", "name": "Test"}]).try_into().unwrap();
    index_client.insert_documents(documents).await.unwrap();

    // Wait for document
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

            if stats.document_count != 1 {
                return Err(anyhow::anyhow!("documents not arrived yet"));
            }

            Ok(stats)
        }
        .boxed()
    })
    .await
    .unwrap();

    // Create claims with wrong orak (doesn't match collection's read API key)
    let wrong_orak = ApiKey::try_new("wrong-api-key").unwrap();
    let claims_read_api_key = ReadApiKey::Claims(CustomerClaims {
        orak: wrong_orak,
        extra: HashMap::new(),
    });

    // Search should fail with invalid API key
    let search_params = search_params_from_json(json!({
        "term": "Test"
    }));

    let result = test_context
        .reader
        .search(
            &claims_read_api_key,
            collection_id,
            SearchRequest {
                search_params,
                analytics_metadata: None,
                interaction_id: None,
                search_analytics_event_origin: None,
            },
        )
        .await;

    assert!(
        result.is_err(),
        "Expected search to fail with invalid orak in claims"
    );

    drop(test_context);
}

// =============================================================================
// JWKS Server and JWT Generation Helpers for E2E Tests
// =============================================================================

/// JWKS response handler - returns a symmetric key for HS256 algorithm.
/// The key "foo" is encoded as base64url "Zm9v".
async fn jwks_handler() -> Json<serde_json::Value> {
    Json(json!({
        "keys": [
            {
                "kty": "oct",
                "kid": "test-key-id",
                "k": "Zm9v",  // "foo" base64url encoded
                "alg": "HS256",
                "use": "sig"
            }
        ]
    }))
}

/// Claims structure for generating CustomerClaims JWTs.
#[derive(Debug, Serialize, Deserialize)]
struct TestCustomerJwtClaims {
    orak: String,
    iss: String,
    aud: String,
    exp: u64,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

/// JWT generation endpoint for tests.
async fn generate_customer_jwt(
    Json(params): Json<TestCustomerJwtClaims>,
) -> Json<serde_json::Value> {
    let token = encode(
        &Header::default(),
        &params,
        &EncodingKey::from_secret(b"foo"), // Same key as JWKS
    )
    .expect("Failed to encode JWT");

    Json(serde_json::Value::String(token))
}

/// Starts a mock JWKS server for testing JWT validation.
/// Returns the server address.
async fn start_jwks_server() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
    let addr = listener.local_addr().expect("Failed to get address");
    drop(listener);

    let router = Router::new()
        .route("/.well-known/jwks.json", get(jwks_handler))
        .route("/jwt/generate", post(generate_customer_jwt));

    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .expect("Failed to bind async");
        axum::serve(listener, router).await.expect("Server failed");
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    addr
}

/// Test the complete JWT validation flow: JWKS fetch → JWT validation → search with claims.
/// This verifies that:
/// 1. JWKS keys are fetched correctly from a remote server
/// 2. JWT tokens are validated against those keys
/// 3. CustomerClaims are extracted and passed to BeforeSearch hook
/// 4. The hook can filter results based on JWT claims
#[tokio::test(flavor = "multi_thread")]
async fn test_jwt_validation_with_jwks_server() {
    init_log();

    // Start JWKS server
    let jwks_addr = start_jwks_server().await;
    let port = jwks_addr.port();

    // Create JWT manager with the test JWKS server
    let jwt_config = JwtConfig {
        providers: vec![JwksProviderConfig {
            name: "test-provider".to_string(),
            jwks_url: format!("http://localhost:{port}/.well-known/jwks.json")
                .parse()
                .expect("Invalid URL"),
            refresh_interval: None,
            issuers: vec![StackString::try_new("https://test-issuer").expect("Invalid issuer")],
            audiences: vec![
                StackString::try_new("https://test-audience").expect("Invalid audience")
            ],
        }],
    };

    // Create TestContext with JWT config
    let test_context = TestContext::new_with_jwt_config(jwt_config.clone()).await;
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

    // Get the raw API key to use as orak in JWT
    let read_api_key_raw = match &collection_client.read_api_key {
        ReadApiKey::ApiKey(key) => *key,
        ReadApiKey::Claims(claims) => claims.orak,
    };

    // Create JWT claims with the collection's API key and country claim
    let mut extra = HashMap::new();
    extra.insert("country".to_string(), Value::String("IT".to_string()));

    let jwt_claims = TestCustomerJwtClaims {
        orak: read_api_key_raw.expose().to_string(),
        iss: "https://test-issuer".to_string(),
        aud: "https://test-audience".to_string(),
        exp: Utc::now().timestamp() as u64 + 3600, // 1 hour from now
        extra,
    };

    // Generate JWT token via the test server
    let jwt_token: String = reqwest::Client::new()
        .post(format!("http://localhost:{port}/jwt/generate"))
        .json(&jwt_claims)
        .send()
        .await
        .expect("Failed to generate JWT")
        .json()
        .await
        .expect("Failed to parse JWT response");

    // Validate JWT using JwtManager
    let jwt_manager = JwtManager::<CustomerClaims>::new(Some(jwt_config))
        .await
        .expect("Failed to create JWT manager");

    let validated_claims = jwt_manager
        .check(&jwt_token)
        .await
        .expect("JWT validation failed");

    // Verify the claims were extracted correctly
    assert_eq!(
        validated_claims.orak.expose(),
        read_api_key_raw.expose(),
        "orak should match the collection's read API key"
    );
    assert_eq!(
        validated_claims.extra.get("country"),
        Some(&Value::String("IT".to_string())),
        "country claim should be IT"
    );

    // Perform search using the validated claims
    let claims_read_api_key = ReadApiKey::Claims(validated_claims);

    let search_params = search_params_from_json(json!({
        "term": "Document"
    }));

    let result = test_context
        .reader
        .search(
            &claims_read_api_key,
            collection_id,
            SearchRequest {
                search_params,
                analytics_metadata: None,
                interaction_id: None,
                search_analytics_event_origin: None,
            },
        )
        .await
        .unwrap();

    // Should only get the IT document due to hook filtering
    assert_eq!(result.count, 1, "Expected 1 document, got {}", result.count);
    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(
        ids,
        vec!["2"],
        "Expected document with id='2' (IT document)"
    );

    drop(test_context);
}
