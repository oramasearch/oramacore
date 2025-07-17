use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{endpoint, utoipa::IntoParams, AddRoute};
use serde::Deserialize;

use crate::{
    collection_manager::sides::read::{AnalyticSearchEventInvocationType, ReadSide},
    types::{ApiKey, CollectionId, ExecuteActionPayload, ExecuteActionPayloadName, SearchParams},
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new().add(execute_action_v0()).with_state(read_side)
}

use axum_openapi3::utoipa;
#[derive(Deserialize, IntoParams)]
struct ActionQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[endpoint(
    method = "POST",
    path = "/v1/{collection_id}/actions/execute",
    description = "Execute an action. Typically used by the Python server to perform actions that requires access to OramaCore components."
)]
#[axum::debug_handler]
async fn execute_action_v0(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<ActionQueryParams>,
    Json(params): Json<ExecuteActionPayload>,
) -> impl IntoResponse {
    let ExecuteActionPayload { name, context } = params;

    let read_api_key = query.api_key;

    match name {
        ExecuteActionPayloadName::Search => {
            let search_context: SearchParams = serde_json::from_str(&context).unwrap(); // @todo: handle error
            read_side
                .search(
                    read_api_key,
                    collection_id,
                    search_context,
                    AnalyticSearchEventInvocationType::Action,
                )
                .await
                .map(Json)
        }
    }
}
