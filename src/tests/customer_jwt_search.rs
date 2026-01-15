use std::collections::HashMap;

use anyhow::Context;
use futures::FutureExt;
use oramacore_lib::hook_storage::HookType;
use serde_json::{json, Value};

use crate::collection_manager::sides::read::SearchRequest;
use crate::tests::utils::{extrapolate_ids_from_result, init_log, wait_for, TestContext};
use crate::types::{
    ApiKey, CollectionStatsRequest, CustomerClaims, DocumentList, ReadApiKey, SearchParams,
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
        ids, vec!["2"],
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
        ids, vec!["1"],
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
