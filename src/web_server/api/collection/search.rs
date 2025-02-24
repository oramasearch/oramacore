use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::*;
use serde::Deserialize;
use serde_json::json;
use tracing::error;
use utoipa::IntoParams;

use crate::{
    collection_manager::{
        dto::{ApiKey, SearchParams},
        sides::ReadSide,
    },
    types::CollectionId,
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(search())
        .add(stats())
        // .route("/:collection_id/documents/:document_id", get(get_doc_by_id))
        .with_state(read_side)
}

#[derive(Deserialize, IntoParams)]
struct SearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/search",
    description = "Search Endpoint"
)]
async fn search(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<SearchParams>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let read_api_key = query.api_key;

    let output = read_side.search(read_api_key, collection_id, json).await;

    match output {
        Ok(data) => Ok((StatusCode::OK, Json(data))),
        Err(e) => {
            error!("Error deleting documents to collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{id}/stats",
    description = "Stats Endpoint"
)]
async fn stats(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let read_api_key = query.api_key;

    match read_side
        .collection_stats(read_api_key, collection_id)
        .await
    {
        Ok(data) => Ok(Json(data)),
        Err(e) => {
            error!("Error getting stats for collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}
/*
async fn get_doc_by_id(
    Path((collection_id, document_id)): Path<(String, String)>,
    readers: State<Arc<CollectionsReader>>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(collection_id);

    let collection = readers.get_collection(collection_id).await;

    let collection = match collection {
        Some(collection) => collection,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "collection not found" })),
            ));
        }
    };

    let output = collection
        .get_doc_by_unique_field("id".to_string(), document_id)
        .await;

    match output {
        Ok(Some(data)) => Ok((StatusCode::OK, Json(data))),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "document not found" })),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}
*/
