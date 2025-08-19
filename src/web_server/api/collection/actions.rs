use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde::Deserialize;

use crate::{
    collection_manager::sides::read::{AnalyticSearchEventInvocationType, ReadSide},
    types::{ApiKey, CollectionId, ExecuteActionPayload, ExecuteActionPayloadName, SearchParams},
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/{collection_id}/actions/execute",
            post(execute_action_v0),
        )
        .with_state(read_side)
}

#[derive(Deserialize)]
struct ActionQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[axum::debug_handler]
async fn execute_action_v0(
    Path(collection_id): Path<CollectionId>,
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
