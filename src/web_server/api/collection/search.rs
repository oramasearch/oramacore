use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::*;
use serde::Deserialize;
use serde_json::json;
use utoipa::IntoParams;

use crate::{
    collection_manager::sides::ReadSide,
    types::{ApiKey, CollectionId, SearchParams},
    web_server::api::collection::admin::print_error,
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(search())
        .add(stats())
        .with_state(read_side)
}

#[derive(Deserialize, IntoParams)]
struct SearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/search",
    description = "Search Endpoint"
)]
async fn search(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<SearchParams>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let read_api_key = query.api_key;

    let output = read_side.search(read_api_key, collection_id, json).await;

    match output {
        Ok(data) => Ok((StatusCode::OK, Json(data))),
        Err(e) => {
            print_error(&e, "Error searching collection");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/stats",
    description = "Stats Endpoint"
)]
async fn stats(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let read_api_key = query.api_key;

    match read_side
        .collection_stats(read_api_key, collection_id)
        .await
    {
        Ok(data) => Ok(Json(data)),
        Err(e) => {
            print_error(&e, "Error getting collection stats");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}
