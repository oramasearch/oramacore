use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::post,
    Json, Router,
};

use crate::{
    collection_manager::sides::read::{ReadSide, SearchRequest},
    types::{
        CollectionId, ExecuteActionPayload, ExecuteActionPayloadName, ReadApiKey, SearchParams,
    },
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/{collection_id}/actions/execute",
            post(execute_action_v0),
        )
        .with_state(read_side)
}

#[axum::debug_handler]
async fn execute_action_v0(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
    Json(params): Json<ExecuteActionPayload>,
) -> impl IntoResponse {
    let ExecuteActionPayload { name, context } = params;

    match name {
        ExecuteActionPayloadName::Search => {
            let search_params: SearchParams = serde_json::from_str(&context).unwrap(); // @todo: handle error
            read_side
                .search(
                    &read_api_key,
                    collection_id,
                    SearchRequest {
                        search_params,
                        analytics_metadata: None,
                        interaction_id: None,
                        search_analytics_event_origin: None,
                    },
                )
                .await
                .map(Json)
        }
    }
}
