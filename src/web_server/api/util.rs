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
        ApiKey, CollectionId, CustomerClaims, DashboardClaims, IndexId, ReadApiKey, TrainingSetId,
        WriteApiKey,
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
    JwtManager<CustomerClaims>: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let query_api_key = Query::<ReaderAPIKeyParams>::from_request_parts(parts, state)
            .await
            .ok();

        let jwt_manager = JwtManager::<CustomerClaims>::from_ref(state);

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
    jwt_manager: &JwtManager<CustomerClaims>,
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

/// Wraps an error message in a JSON response: `{"error": "<message>"}`
fn json_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

pub fn print_error(e: &anyhow::Error, msg: &'static str) {
    error!(error = ?e, msg);
    e.chain()
        .skip(1)
        .for_each(|cause| println!("because: {cause:?}"));
}

impl IntoResponse for WriteError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::Generic(e) => {
                print_error(&e, "Unhandled error in write side");
                error!(error = ?e, "Generic write error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
            }
            Self::InvalidMasterApiKey => {
                (StatusCode::UNAUTHORIZED, "Invalid master API key".into())
            }
            Self::CollectionAlreadyExists(collection_id) => (
                StatusCode::CONFLICT,
                format!("Collection with id {collection_id} already exists"),
            ),
            Self::InvalidWriteApiKey(collection_id)
            | Self::CollectionNotFound(collection_id)
            | Self::JwtBelongToAnotherCollection(collection_id) => (
                StatusCode::BAD_REQUEST,
                format!("Collection with id {collection_id} not found or invalid write api key"),
            ),
            Self::IndexAlreadyExists(collection_id, index_id) => (
                StatusCode::CONFLICT,
                format!("Index with id {index_id} already exists in collection {collection_id}"),
            ),
            Self::IndexNotFound(collection_id, index_id) => (
                StatusCode::BAD_REQUEST,
                format!("Index {index_id} not found in collection {collection_id}"),
            ),
            Self::TempIndexNotFound(collection_id, index_id) => (
                StatusCode::BAD_REQUEST,
                format!("Temporary index {index_id} not found in collection {collection_id}"),
            ),
            Self::HookExec(msg) => (StatusCode::BAD_REQUEST, format!("Hook error: {msg}")),
            Self::HookError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Hook storage error: {e}"),
            ),
            Self::PinRulesError(e) => (StatusCode::BAD_REQUEST, format!("Invalid pin rule: {e:?}")),
            Self::DocumentLimitExceeded(collection_id, limit) => (
                StatusCode::PAYMENT_REQUIRED,
                format!("Document limit exceeded for collection {collection_id}. Limit: {limit}"),
            ),
            Self::ShelfError(e) => (StatusCode::BAD_REQUEST, format!("Invalid shelf: {e:?}")),
            Self::ShelfDocumentLimitExceeded(actual, max) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Too many documents in shelf: {actual} (max: {max})"),
            ),
        };

        json_error(status, message)
    }
}

impl IntoResponse for ToolError {
    fn into_response(self) -> Response {
        // Delegate to inner error types that already produce json_error responses
        match self {
            ToolError::WriteError(e) => return e.into_response(),
            ToolError::ReadError(e) => return e.into_response(),
            _ => {}
        }

        let (status, message) = match self {
            ToolError::Generic(e) => {
                print_error(&e, "Unhandled error in tool side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
            }
            ToolError::ValidationError(tool_id, msg) => (
                StatusCode::BAD_REQUEST,
                format!("Tool {tool_id} contains invalid code: {msg}"),
            ),
            ToolError::CompilationError(tool_id, msg) => (
                StatusCode::BAD_REQUEST,
                format!("Tool {tool_id} doesn't compile: {msg}"),
            ),
            ToolError::Duplicate(tool_id) => (
                StatusCode::CONFLICT,
                format!("Tool {tool_id} already exists"),
            ),
            ToolError::NotFound(tool_id, collection_id) => (
                StatusCode::BAD_REQUEST,
                format!("Tool {tool_id} not found in collection {collection_id}"),
            ),
            ToolError::NoTools(collection_id) => (
                StatusCode::BAD_REQUEST,
                format!("Collection {collection_id} doesn't have any tool"),
            ),
            ToolError::ExecutionSerializationError(collection_id, tool_id, e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Tool {tool_id} from collection {collection_id} returns an JSON error: {e:?}"
                ),
            ),
            ToolError::ExecutionTimeout(collection_id, tool_id) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Tool {tool_id} from collection {collection_id} goes in timeout"),
            ),
            ToolError::ExecutionError(collection_id, tool_id, e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Tool {tool_id} from collection {collection_id} exited with this error: {e:?}"
                ),
            ),
            // Already handled above
            ToolError::WriteError(_) | ToolError::ReadError(_) => unreachable!(),
        };

        json_error(status, message)
    }
}

impl IntoResponse for ReadError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::Generic(e) => {
                print_error(&e, "Unhandled error in read side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
            }
            Self::NotFound(collection_id) => (
                StatusCode::BAD_REQUEST,
                format!("Collection {collection_id} not found"),
            ),
            Self::Hook(e) => (
                StatusCode::BAD_REQUEST,
                format!("Hook error: {e:?}"),
            ),
            Self::IndexNotFound(collection_id, index_id) => (
                StatusCode::BAD_REQUEST,
                format!("Index {index_id:?} not found in Collection {collection_id:?}"),
            ),
            Self::FilterFieldNotFound(field_name) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot filter by \"{field_name}\": unknown field"),
            ),
            Self::InvalidSortField(field_name, field_type) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot sort by \"{field_name}\": only number, date or boolean fields are supported for sorting, but got {field_type}"),
            ),
            Self::SortFieldNotFound(field_name) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot sort by \"{field_name}\": no index has that field"),
            ),
            Self::FacetFieldNotFound(field_names) => (
                StatusCode::BAD_REQUEST,
                format!("Cannot facet by \"{field_names:?}\": no index has that field"),
            ),
            Self::UnknownIndex(unknown_index_ids, available_index_ids) => (
                StatusCode::BAD_REQUEST,
                format!(
                    "Unknown indexes requested: {unknown_index_ids:?}. Available indexes: {available_index_ids:?}"
                ),
            ),
            Self::ShelfNotFound(shelf_id) => (
                StatusCode::BAD_REQUEST,
                format!("Shelf '{shelf_id}' not found"),
            ),
        };

        json_error(status, message)
    }
}

impl IntoResponse for AnswerError {
    fn into_response(self) -> Response {
        // Delegate to ReadError which already produces json_error responses
        if let AnswerError::ReadError(e) = self {
            return e.into_response();
        }

        let (status, message) = match self {
            AnswerError::Generic(e) => {
                print_error(&e, "Unhandled error in answer side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
            }
            AnswerError::ChannelClosed(e) => {
                print_error(&anyhow::anyhow!(e), "Channel closed in answer side");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Channel closed unexpectedly".into(),
                )
            }
            AnswerError::BeforeRetrievalHookError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Before retrieval hook error: {e}"),
            ),
            AnswerError::BeforeAnswerHookError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Before answer hook error: {e}"),
            ),
            AnswerError::JSError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error running JS code: {e:?}"),
            ),
            AnswerError::TitleError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error generating title: {e:?}"),
            ),
            // Already handled above
            AnswerError::ReadError(_) => unreachable!(),
        };

        json_error(status, message)
    }
}

impl IntoResponse for SuggestionsError {
    fn into_response(self) -> Response {
        match self {
            SuggestionsError::Generic(e) => {
                print_error(&e, "Unhandled error in suggestions side");
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Cannot process the request: {e:?}"),
                )
            }
            SuggestionsError::RepairError(e) => json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error repairing suggestions: {e:?}"),
            ),
            SuggestionsError::ParseError(e) => json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error parsing suggestions: {e:?}"),
            ),
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

impl FromRef<Arc<ReadSide>> for JwtManager<CustomerClaims> {
    fn from_ref(app_state: &Arc<ReadSide>) -> JwtManager<CustomerClaims> {
        app_state.get_jwt_manager()
    }
}
