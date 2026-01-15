use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{FromRef, FromRequestParts, Path, Query},
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
use http::{request::Parts, StatusCode};
use serde::Deserialize;
use serde_json::json;
use tracing::error;

#[derive(Deserialize)]
struct ReaderAPIKeyParams {
    #[serde(rename = "api-key")]
    api_key: String,
}

use crate::{
    ai::{
        answer::{AnswerError, SuggestionsError},
        tools::ToolError,
    },
    auth::{JwtError, JwtManager},
    collection_manager::sides::{
        read::{AnalyticsMetadataFromRequest, OramaCoreAnalytics, ReadError, ReadSide},
        write::{WriteError, WriteSide},
    },
    types::{
        ApiKey, CollectionId, DashboardClaims, IndexId, ReadApiKey, TrainingSetId, WriteApiKey,
    },
};

// This is ugly, but our Dashboard api keys doesn't contains dots, while JWTs contains dots.
fn is_jwt_token(token: &str) -> bool {
    token.starts_with("ey") && token.contains('.')
}

impl<S> FromRequestParts<S> for WriteApiKey
where
    JwtManager<DashboardClaims>: FromRef<S>,
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
        let bearer_token = bearer_token.token();
        let api_key = if is_jwt_token(bearer_token) {
            let manager = JwtManager::from_ref(state);
            let claims = match manager.check(bearer_token).await {
                Ok(claims) => claims,
                Err(JwtError::Generic(e)) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": format!("JWT Auth error: {:?}", e)
                        })),
                    ));
                }
                Err(JwtError::InvalidIssuer { wanted }) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": format!(
                                "Invalid issuer. Wanted one of {:?}",
                                wanted
                            )
                        })),
                    ));
                }
                Err(JwtError::InvalidAudience { wanted }) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": format!(
                                "Invalid audience. Wanted '{:?}'",
                                wanted
                            )
                        })),
                    ));
                }
                Err(JwtError::NotConfigured) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({
                            "message": "JWT is not configured on this instance"
                        })),
                    ));
                }
                Err(JwtError::ExpiredToken) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": "JWT token is expired"
                        })),
                    ));
                }
                Err(JwtError::MissingRequiredClaim(c)) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": format!("Missing required claim {c}"),
                        })),
                    ));
                }
                Err(JwtError::NoProviderForIssuer { issuer, providers }) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": format!(
                                "No JWT provider found for issuer '{}'. Configured providers: {:?}",
                                issuer,
                                providers
                            )
                        })),
                    ));
                }
                Err(JwtError::UnableToDecodeIssuer { reason }) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": format!("Unable to decode JWT issuer: {}", reason)
                        })),
                    ));
                }
                Err(JwtError::AllProvidersFailed { issuer, errors }) => {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "message": format!(
                                "All JWT providers failed for issuer '{}'. Errors: {:?}",
                                issuer,
                                errors
                            )
                        })),
                    ));
                }
            };
            WriteApiKey::from_claims(claims)
        } else {
            let api_key = ApiKey::try_new(bearer_token).map_err(|e| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!("Bad API key: {:?}", e)
                    })),
                )
            })?;

            WriteApiKey::from_api_key(api_key)
        };

        Ok(api_key)
    }
}

impl<S> FromRequestParts<S> for ApiKey
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        // First check query parameter (takes precedence)
        let query_api_key = Query::<ReaderAPIKeyParams>::from_request_parts(parts, state)
            .await
            .ok();

        if let Some(Query(ReaderAPIKeyParams { api_key })) = query_api_key {
            let api_key = ApiKey::try_new(&api_key).map_err(|e| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!("Invalid API key: {:?}", e)
                    })),
                )
            })?;
            return Ok(api_key);
        }

        // Then fall back to Authorization header
        let header_api_key = TypedHeader::<Authorization<Bearer>>::from_request_parts(parts, state)
            .await
            .ok()
            .map(|bearer| bearer.0 .0.token().to_string());

        let Some(raw_api_key) = header_api_key else {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "message": "Missing API key. Please provide it either as a query parameter '?api-key=...' or in the Authorization header 'Bearer <api-key>'"
                })),
            ));
        };

        let api_key = ApiKey::try_new(&raw_api_key).map_err(|e| {
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "message": format!("Invalid API key: {:?}", e)
                })),
            )
        })?;
        Ok(api_key)
    }
}

impl<S> FromRequestParts<S> for ReadApiKey
where
    JwtManager<DashboardClaims>: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let query_api_key = Query::<ReaderAPIKeyParams>::from_request_parts(parts, state)
            .await
            .ok();

        let jwt_manager = JwtManager::<DashboardClaims>::from_ref(state);

        if let Some(Query(ReaderAPIKeyParams { api_key })) = query_api_key {
            return get_read_api_key(&jwt_manager, &api_key).await;
        }

        match TypedHeader::<Authorization<Bearer>>::from_request_parts(parts, state).await {
            Ok(header) => get_read_api_key(&jwt_manager, header.0 .0.token()).await,
            Err(err) => Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "message": format!("missing api key: {:?}", err)
                })),
            )),
        }
    }
}

async fn get_read_api_key(
    jwt_manager: &JwtManager<DashboardClaims>,
    token: &str,
) -> Result<ReadApiKey, (StatusCode, Json<serde_json::Value>)> {
    let api_key = if is_jwt_token(token) {
        let claims = match jwt_manager.check(token).await {
            Ok(claims) => claims,
            Err(JwtError::Generic(e)) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!("JWT Auth error: {:?}", e)
                    })),
                ));
            }
            Err(JwtError::InvalidIssuer { wanted }) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!(
                            "Invalid issuer. Wanted one of {:?}",
                            wanted
                        )
                    })),
                ));
            }
            Err(JwtError::InvalidAudience { wanted }) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!(
                            "Invalid audience. Wanted '{:?}'",
                            wanted
                        )
                    })),
                ));
            }
            Err(JwtError::NotConfigured) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "message": "JWT is not configured on this instance"
                    })),
                ));
            }
            Err(JwtError::ExpiredToken) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": "JWT token is expired"
                    })),
                ));
            }
            Err(JwtError::MissingRequiredClaim(c)) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!("Missing required claim {c}"),
                    })),
                ));
            }
            Err(JwtError::NoProviderForIssuer { issuer, providers }) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!(
                            "No JWT provider found for issuer '{}'. Configured providers: {:?}",
                            issuer,
                            providers
                        )
                    })),
                ));
            }
            Err(JwtError::UnableToDecodeIssuer { reason }) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!("Unable to decode JWT issuer: {}", reason)
                    })),
                ));
            }
            Err(JwtError::AllProvidersFailed { issuer, errors }) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "message": format!(
                            "All JWT providers failed for issuer '{}'. Errors: {:?}",
                            issuer,
                            errors
                        )
                    })),
                ));
            }
        };

        ReadApiKey::from_claims(claims)
    } else {
        let api_key = ApiKey::try_new(token).map_err(|e| {
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "message": format!("Invalid API key: {:?}", e)
                })),
            )
        })?;

        ReadApiKey::from_api_key(api_key)
    };

    Ok(api_key)
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

impl<S> FromRequestParts<S> for TrainingSetId
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
                        "message": format!("missing training set id: {:?}", e)
                    })),
                )
            })?;

        let training_set_id = a.get("training_set").ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "message": "missing training set id"
                })),
            )
        })?;

        let training_set_id = TrainingSetId::try_new(training_set_id).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "message": format!("Bad training set id: {:?}", e)
                })),
            )
        })?;

        Ok(training_set_id)
    }
}

pub fn print_error(e: &anyhow::Error, msg: &'static str) {
    error!(error = ?e, msg);
    e.chain()
        .skip(1)
        .for_each(|cause| println!("because: {cause:?}"));
}

impl IntoResponse for WriteError {
    fn into_response(self) -> Response {
        match self {
            Self::Generic(e) => {
                print_error(&e, "Unhandled error in write side");
                error!(error = ?e, "Generic write error");
                let body = format!("Cannot process the request: {e:?}");
                (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
            }
            Self::InvalidMasterApiKey => {
                let body = "Invalid master API key".to_string();
                (StatusCode::UNAUTHORIZED, body).into_response()
            }
            Self::CollectionAlreadyExists(collection_id) => {
                let body = format!("Collection with id {collection_id} already exists");
                (StatusCode::CONFLICT, body).into_response()
            }
            Self::InvalidWriteApiKey(collection_id)
            | Self::CollectionNotFound(collection_id)
            | Self::JwtBelongToAnotherCollection(collection_id) => {
                let body = format!(
                    "Collection with id {collection_id} not found or invalid write api key"
                );
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            Self::IndexAlreadyExists(collection_id, index_id) => {
                let body = format!(
                    "Index with id {index_id} already exists in collection {collection_id}"
                );
                (StatusCode::CONFLICT, body).into_response()
            }
            Self::IndexNotFound(collection_id, index_id) => {
                let body = format!("Index {index_id} not found in collection {collection_id}");
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            Self::TempIndexNotFound(collection_id, index_id) => {
                let body =
                    format!("Temporary index {index_id} not found in collection {collection_id}");
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            Self::HookWriterError(e) => {
                let body = format!("Invalid hook: {e:?}");
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            Self::PinRulesError(e) => {
                let body = format!("Invalid pin rule: {e:?}");
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            Self::DocumentLimitExceeded(collection_id, limit) => {
                let body = format!(
                    "Document limit exceeded for collection {collection_id}. Limit: {limit}"
                );
                (StatusCode::PAYMENT_REQUIRED, body).into_response()
            }
            Self::ShelfError(e) => {
                let body = format!("Invalid shelf: {e:?}");
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            Self::ShelfDocumentLimitExceeded(actual, max) => {
                let body = format!("Too many documents in shelf: {actual} (max: {max})");
                (StatusCode::PAYLOAD_TOO_LARGE, body).into_response()
            }
        }
    }
}

impl IntoResponse for ToolError {
    fn into_response(self) -> Response {
        match self {
            ToolError::Generic(e) => {
                print_error(&e, "Unhandled error in tool side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
                    .into_response()
            }
            ToolError::WriteError(e) => e.into_response(),
            ToolError::ReadError(e) => e.into_response(),
            ToolError::ValidationError(tool_id, msg) => (
                StatusCode::BAD_REQUEST,
                format!("Tool {tool_id} contains invalid code: {msg}"),
            )
                .into_response(),
            ToolError::CompilationError(tool_id, msg) => (
                StatusCode::BAD_REQUEST,
                format!("Tool {tool_id} doesn't compile: {msg}"),
            )
                .into_response(),
            ToolError::Duplicate(tool_id) => (
                StatusCode::CONFLICT,
                format!("Tool {tool_id} already exists"),
            )
                .into_response(),
            ToolError::NotFound(tool_id, collection_id) => (
                StatusCode::BAD_REQUEST,
                format!("Tool {tool_id} not found in collection {collection_id}"),
            )
                .into_response(),
            ToolError::NoTools(collection_id) => (
                StatusCode::BAD_REQUEST,
                format!("Collection {collection_id} doesn't have any tool"),
            )
                .into_response(),
            ToolError::ExecutionSerializationError(collection_id, tool_id, e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Tool {tool_id} from collection {collection_id} returns an JSON error: {e:?}"
                ),
            )
                .into_response(),
            ToolError::ExecutionTimeout(collection_id, tool_id) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Tool {tool_id} from collection {collection_id} goes in timeout"),
            )
                .into_response(),
            ToolError::ExecutionError(collection_id, tool_id, e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Tool {tool_id} from collection {collection_id} exited with this error: {e:?}"
                ),
            )
                .into_response(),
        }
    }
}

impl IntoResponse for ReadError {
    fn into_response(self) -> Response {
        match self {
            Self::Generic(e) => {
                print_error(&e, "Unhandled error in read side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
                    .into_response()
            }
            Self::NotFound(collection_id) => (
                StatusCode::BAD_REQUEST,
                format!("Collection {collection_id} not found"),
            )
                .into_response(),
            Self::Hook(e) => {
                (StatusCode::BAD_REQUEST, format!("Hook error: {e:?}")).into_response()
            }
            Self::IndexNotFound(collection_id, index_id) => (
                StatusCode::BAD_REQUEST,
                format!("Index {index_id:?} not found in Collection {collection_id:?}"),
            )
                .into_response(),
            Self::FilterFieldNotFound(field_name) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot filter by \"{field_name}\": unknown field"),
            )
                .into_response(),
            Self::InvalidSortField(field_name, field_type) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot sort by \"{field_name}\": only number, date or boolean fields are supported for sorting, but got {field_type}"),
            )
                .into_response(),
            Self::SortFieldNotFound(field_name) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot sort by \"{field_name}\": no index has that field"),
            )
                .into_response(),
            Self::FacetFieldNotFound(field_names) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot facet by \"{field_names:?}\": no index has that field"),
            )
                .into_response(),
            Self::UnknownIndex(unknown_index_ids, available_index_ids) => (
                StatusCode::BAD_REQUEST,
                format!(
                    "Unknown indexes requested: {unknown_index_ids:?}. Available indexes: {available_index_ids:?}"
                ),
            )
                .into_response(),
            Self::ShelfNotFound(shelf_id) => (
                StatusCode::BAD_REQUEST,
                format!("Shelf '{shelf_id}' not found")
            ).into_response(),
        }
    }
}

impl IntoResponse for AnswerError {
    fn into_response(self) -> Response {
        match self {
            AnswerError::Generic(e) => {
                print_error(&e, "Unhandled error in answer side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
                    .into_response()
            }
            AnswerError::ReadError(e) => e.into_response(),
            AnswerError::ChannelClosed(e) => {
                print_error(&anyhow::anyhow!(e), "Channel closed in answer side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Channel closed unexpectedly",
                )
                    .into_response()
            }
            AnswerError::HookError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Hook error {e:?}"),
            )
                .into_response(),
            AnswerError::JSError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error running JS code: {e:?}"),
            )
                .into_response(),
            AnswerError::TitleError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error generating title: {e:?}"),
            )
                .into_response(),
        }
    }
}

impl IntoResponse for SuggestionsError {
    fn into_response(self) -> Response {
        match self {
            SuggestionsError::Generic(e) => {
                print_error(&e, "Unhandled error in suggestions side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
                    .into_response()
            }
            SuggestionsError::RepairError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error repairing suggestions: {e:?}"),
            )
                .into_response(),
            SuggestionsError::ParseError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error parsing suggestions: {e:?}"),
            )
                .into_response(),
        }
    }
}

impl<S> FromRequestParts<S> for AnalyticsMetadataFromRequest
where
    OramaCoreAnalyticsOption: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let manager = OramaCoreAnalyticsOption::from_ref(state);

        let headers = if let Some(manager) = &manager.0 {
            let headers = manager
                .get_metadata_from_headers()
                .iter()
                .filter_map(|pair| {
                    let header_value = parts
                        .headers
                        .get(&pair.header)
                        .and_then(|v| v.to_str().ok())?;
                    Some((pair.metadata_key.clone(), header_value.to_string()))
                })
                .collect();

            headers
        } else {
            Default::default()
        };

        Ok(AnalyticsMetadataFromRequest { headers })
    }
}

pub struct OramaCoreAnalyticsOption(pub Option<OramaCoreAnalytics>);

impl FromRef<Arc<ReadSide>> for OramaCoreAnalyticsOption {
    fn from_ref(app_state: &Arc<ReadSide>) -> OramaCoreAnalyticsOption {
        OramaCoreAnalyticsOption(app_state.get_analytics_logs())
    }
}

impl FromRef<Arc<WriteSide>> for JwtManager<DashboardClaims> {
    fn from_ref(app_state: &Arc<WriteSide>) -> JwtManager<DashboardClaims> {
        app_state.get_jwt_manager()
    }
}

impl FromRef<Arc<ReadSide>> for JwtManager<DashboardClaims> {
    fn from_ref(app_state: &Arc<ReadSide>) -> JwtManager<DashboardClaims> {
        app_state.get_jwt_manager()
    }
}
