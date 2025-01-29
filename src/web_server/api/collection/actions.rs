use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{endpoint, AddRoute};
use http::StatusCode;
use serde_json::json;

use crate::{
    collection_manager::{
        dto::{ExecuteActionPayload, SearchParams},
        sides::ReadSide,
    },
    types::CollectionId,
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new().add(execute_action_v0()).with_state(read_side)
}

#[endpoint(
    method = "POST",
    path = "/v0/{collection_id}/actions/execute",
    description = "Execute an action. Typically used by the Python server to perform actions that requires access to OramaCore components."
)]
async fn execute_action_v0(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Json(params): Json<ExecuteActionPayload>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let ExecuteActionPayload { name, context } = params;

    match name.as_str() {
        "search" => {
            let search_context: SearchParams = serde_json::from_str(&context).unwrap(); // @todo: handle error
            let output = read_side.search(collection_id, search_context).await;

            match output {
                Ok(data) => Ok((StatusCode::OK, Json(data))),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e.to_string() })),
                )),
            }
        }
        _ => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": format!("Action {} was not found", name) })),
        )),
    }
}
