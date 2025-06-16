use std::collections::HashMap;

use axum::{
    extract::{FromRequestParts, Path},
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
use http::{request::Parts, StatusCode};
use serde_json::json;
use tracing::error;

use crate::{
    collection_manager::sides::{segments::SegmentError, write::WriteError},
    types::{ApiKey, CollectionId, IndexId},
};

impl<S> FromRequestParts<S> for ApiKey
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let bearer_token = TypedHeader::<Authorization<Bearer>>::from_request_parts(parts, state)
            .await
            .map_err(|e| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!("missing api key: {:?}", e)
                    })),
                )
            })?;
        let bearer_token = bearer_token.0 .0;

        let api_key = ApiKey::try_new(bearer_token.token()).map_err(|e| {
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "message": format!("Bad API key: {:?}", e)
                })),
            )
        })?;

        Ok(api_key)
    }
}

impl<S> FromRequestParts<S> for CollectionId
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let a = Path::<HashMap<String, String>>::from_request_parts(parts, state)
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "message": format!("missing collection id: {:?}", e)
                    })),
                )
            })?;

        let coll_id = a.get("collection_id").ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "message": "missing collection id"
                })),
            )
        })?;

        let coll_id = CollectionId::try_new(coll_id).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "message": format!("Bad collection id: {:?}", e)
                })),
            )
        })?;

        Ok(coll_id)
    }
}

impl<S> FromRequestParts<S> for IndexId
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let a = Path::<HashMap<String, String>>::from_request_parts(parts, state)
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "message": format!("missing index id: {:?}", e)
                    })),
                )
            })?;

        let index_id = a.get("index_id").ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "message": "missing index id"
                })),
            )
        })?;

        let index_id = IndexId::try_new(index_id).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "message": format!("Bad index id: {:?}", e)
                })),
            )
        })?;

        Ok(index_id)
    }
}

pub fn print_error(e: &anyhow::Error, msg: &'static str) {
    error!(error = ?e, msg);
    e.chain()
        .skip(1)
        .for_each(|cause| println!("because: {:?}", cause));
}

impl IntoResponse for WriteError {
    fn into_response(self) -> Response {
        match self {
            WriteError::Generic(e) => {
                print_error(&e, "Unhandled error in write side");
                error!(error = ?e, "Generic write error");
                let body = format!("Cannot process the request: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
            }
            WriteError::InvalidMasterApiKey => {
                let body = "Invalid master API key".to_string();
                (StatusCode::UNAUTHORIZED, body).into_response()
            }
            WriteError::CollectionAlreadyExists(collection_id) => {
                let body = format!("Collection with id {} already exists", collection_id);
                (StatusCode::CONFLICT, body).into_response()
            }
            WriteError::InvalidWriteApiKey(collection_id)
            | WriteError::CollectionNotFound(collection_id) => {
                let body = format!(
                    "Collection with id {} not found or invalid write api key",
                    collection_id
                );
                (StatusCode::NOT_FOUND, body).into_response()
            }
            WriteError::IndexAlreadyExists(collection_id, index_id) => {
                let body = format!(
                    "Index with id {} already exists in collection {}",
                    index_id, collection_id
                );
                (StatusCode::CONFLICT, body).into_response()
            }
            WriteError::IndexNotFound(collection_id, index_id) => {
                let body = format!(
                    "Index {} not found in collection {}",
                    index_id, collection_id
                );
                (StatusCode::NOT_FOUND, body).into_response()
            }
            WriteError::TempIndexNotFound(collection_id, index_id) => {
                let body = format!(
                    "Temporary index {} not found in collection {}",
                    index_id, collection_id
                );
                (StatusCode::NOT_FOUND, body).into_response()
            }
        }
    }
}

impl IntoResponse for SegmentError {
    fn into_response(self) -> Response {
        match self {
            SegmentError::Generic(e) => {
                print_error(&e, "Unhandled error in segment side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {:?}", e),
                )
                    .into_response()
            }
            SegmentError::WriteError(e) => e.into_response(),
            SegmentError::RepairError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Cannot repair JSON: {:?}", e),
            )
                .into_response(),
            SegmentError::DeserializationError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Invalid JSON format: {:?}", e),
            )
                .into_response(),
        }
    }
}
