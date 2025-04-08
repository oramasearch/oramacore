use std::collections::HashMap;

use axum::{
    extract::{FromRequestParts, Path},
    Json,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
use http::{request::Parts, StatusCode};
use serde_json::json;

use crate::types::{ApiKey, CollectionId};

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

        let api_key = ApiKey::try_from(bearer_token.token()).map_err(|e| {
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

        let coll_id = CollectionId::try_from(coll_id).map_err(|e| {
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
