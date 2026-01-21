use axum::{
    body::Body,
    extract::FromRequestParts,
    http::{Request, StatusCode},
};
use http::request::Parts;

use crate::types::ApiKey;

/// Helper to create request parts with Authorization header
fn create_parts_with_header(api_key: &str) -> Parts {
    let request = Request::builder()
        .uri("/test")
        .header("Authorization", format!("Bearer {api_key}"))
        .body(Body::empty())
        .unwrap();

    let (parts, _) = request.into_parts();
    parts
}

/// Helper to create request parts with query parameter
fn create_parts_with_query(api_key: &str) -> Parts {
    let request = Request::builder()
        .uri(format!("/test?api-key={api_key}"))
        .body(Body::empty())
        .unwrap();

    let (parts, _) = request.into_parts();
    parts
}

/// Helper to create request parts with no authentication
fn create_parts_without_auth() -> Parts {
    let request = Request::builder().uri("/test").body(Body::empty()).unwrap();

    let (parts, _) = request.into_parts();
    parts
}

/// Unit state for testing (no actual state needed for ApiKey extraction)
struct DummyState;

#[tokio::test]
async fn test_api_key_extraction_from_header_valid() {
    let mut parts = create_parts_with_header("my-valid-api-key-12345");
    let state = DummyState;

    let result = ApiKey::from_request_parts(&mut parts, &state).await;

    assert!(result.is_ok(), "Valid API key in header should succeed");
    let api_key = result.unwrap();
    assert_eq!(api_key.expose(), "my-valid-api-key-12345");
}

#[tokio::test]
async fn test_api_key_extraction_from_query_valid() {
    let mut parts = create_parts_with_query("my-valid-api-key-12345");
    let state = DummyState;

    let result = ApiKey::from_request_parts(&mut parts, &state).await;

    assert!(result.is_ok(), "Valid API key in query should succeed");
    let api_key = result.unwrap();
    assert_eq!(api_key.expose(), "my-valid-api-key-12345");
}

#[tokio::test]
async fn test_api_key_extraction_query_takes_precedence() {
    // Create parts with both query and header
    let request = Request::builder()
        .uri("/test?api-key=query-key-12345")
        .header("Authorization", "Bearer header-key-67890")
        .body(Body::empty())
        .unwrap();

    let (mut parts, _) = request.into_parts();
    let state = DummyState;

    let result = ApiKey::from_request_parts(&mut parts, &state).await;

    assert!(result.is_ok(), "Should extract API key successfully");
    let api_key = result.unwrap();
    assert_eq!(api_key.expose(), "query-key-12345");
}

#[tokio::test]
async fn test_api_key_extraction_missing_auth() {
    let mut parts = create_parts_without_auth();
    let state = DummyState;

    let result = ApiKey::from_request_parts(&mut parts, &state).await;

    assert!(result.is_err(), "Missing API key should fail");
    let (status, body) = result.unwrap_err();
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let json_body = body.0;
    assert!(json_body.get("message").is_some());
    let message = json_body["message"].as_str().unwrap();
    assert!(message.contains("Missing API key"));
    assert!(message.contains("api-key"));
    assert!(message.contains("Authorization"));
}

#[tokio::test]
async fn test_api_key_extraction_too_long_query() {
    // Test that validation also works for query parameters
    // The query param is deserialized as String, then validated via ApiKey::try_new()
    let too_long_key = "a".repeat(65);
    let mut parts = create_parts_with_query(&too_long_key);
    let state = DummyState;

    let result = ApiKey::from_request_parts(&mut parts, &state).await;

    assert!(
        result.is_err(),
        "API key longer than 64 chars in query should fail"
    );
    let (status, body) = result.unwrap_err();
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Verify the error message mentions the API key validation issue
    let json_body = body.0;
    assert!(json_body.get("message").is_some());
    let message = json_body["message"].as_str().unwrap();
    assert!(message.contains("Invalid API key"));
}

#[tokio::test]
async fn test_api_key_extraction_empty_string() {
    let mut parts = create_parts_with_header("");
    let state = DummyState;

    let result = ApiKey::from_request_parts(&mut parts, &state).await;

    assert!(result.is_err(), "Empty API key should fail");
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}
