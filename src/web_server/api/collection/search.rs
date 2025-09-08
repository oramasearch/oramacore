use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;

use crate::collection_manager::sides::read::SearchAnalyticEventOrigin;
use crate::{
    collection_manager::sides::read::ReadSide,
    types::{ApiKey, CollectionId, CollectionStatsRequest, SearchParams},
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/search", post(search))
        .route("/v1/collections/{collection_id}/stats", get(stats))
        .with_state(read_side)
}

#[derive(Deserialize)]
struct SearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn search(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<SearchParams>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    read_side
        .search(
            read_api_key,
            collection_id,
            json,
            Some(SearchAnalyticEventOrigin::Direct),
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
