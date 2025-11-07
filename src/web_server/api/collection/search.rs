use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;

use crate::collection_manager::sides::read::{
    AnalyticsMetadataFromRequest, SearchAnalyticEventOrigin, SearchRequest,
};
use crate::{
    collection_manager::sides::read::ReadSide,
    types::{ApiKey, CollectionId, CollectionStatsRequest, SearchParams},
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/search", post(search))
        .route("/v1/collections/{collection_id}/stats", get(stats))
        .route(
            "/v1/collections/{collection_id}/filterable_fields",
            get(filterable_fields),
        )
        .with_state(read_side)
}

#[derive(Deserialize)]
struct SearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize)]
struct FilterableFieldsSearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
    #[serde(rename = "with-keys")]
    with_keys: Option<bool>,
}

async fn search(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    analytics_metadata: AnalyticsMetadataFromRequest,
    Query(query): Query<SearchQueryParams>,
    Json(search_params): Json<SearchParams>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    read_side
        .search(
            read_api_key,
            collection_id,
            SearchRequest {
                search_params,
                analytics_metadata: Some(analytics_metadata),
                interaction_id: None,
                search_analytics_event_origin: Some(SearchAnalyticEventOrigin::Direct),
            },
        )
        .await
        .map(Json)
}

async fn stats(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    // We don't want to expose the variants on HTTP API, so we force `with_keys: false`
    // Anyway, if requested, we could add it on this enpoint in the future.
    read_side
        .collection_stats(
            read_api_key,
            collection_id,
            CollectionStatsRequest { with_keys: false },
        )
        .await
        .map(Json)
}

async fn filterable_fields(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<FilterableFieldsSearchQueryParams>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;
    let with_keys = query.with_keys.unwrap_or(false);

    read_side
        .filterable_fields(read_api_key, collection_id, with_keys)
        .await
        .map(Json)
}
