use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
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
    types::{
        ApiKey, BatchGetDocumentsRequest, BatchGetDocumentsResponse, CollectionId,
        CollectionStatsRequest, SearchParams,
    },
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/search", post(search))
        .route("/v1/collections/{collection_id}/stats", get(stats))
        .route(
            "/v1/collections/{collection_id}/filterable_fields",
            get(filterable_fields),
        )
        .route(
            "/v1/collections/{collection_id}/documents/batch-get",
            post(batch_get_documents),
        )
        .with_state(read_side)
}

#[derive(Deserialize)]
struct FilterableFieldsSearchQueryParams {
    #[serde(rename = "with-keys")]
    with_keys: Option<bool>,
}

async fn search(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    analytics_metadata: AnalyticsMetadataFromRequest,
    read_api_key: ApiKey,
    Json(search_params): Json<SearchParams>,
) -> impl IntoResponse {
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
    read_api_key: ApiKey,
) -> impl IntoResponse {
    read_side
        .collection_stats(
            read_api_key,
            collection_id,
            CollectionStatsRequest { with_keys: true },
        )
        .await
        .map(Json)
}

async fn filterable_fields(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ApiKey,
    Query(query): Query<FilterableFieldsSearchQueryParams>,
) -> impl IntoResponse {
    let with_keys = query.with_keys.unwrap_or(false);

    read_side
        .filterable_fields(read_api_key, collection_id, with_keys)
        .await
        .map(Json)
}

async fn batch_get_documents(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ApiKey,
    Json(request): Json<BatchGetDocumentsRequest>,
) -> impl IntoResponse {
    // Validate request size
    if let Err(e) = request.validate() {
        return (StatusCode::BAD_REQUEST, e).into_response();
    }

    match read_side
        .batch_get_documents(read_api_key, collection_id, request.ids)
        .await
    {
        Ok(documents) => Json(BatchGetDocumentsResponse { documents }).into_response(),
        Err(e) => e.into_response(),
    }
}
