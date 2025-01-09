use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::*;
use serde_json::json;
use tracing::error;

use crate::{
    collection_manager::{dto::SearchParams, sides::{document_storage::DocumentStorage, read::CollectionsReader}},
    types::CollectionId,
};

struct ReaderAPIState {
    readers: Arc<CollectionsReader>,
    doc: Arc<dyn DocumentStorage>,
}

pub fn apis(readers: Arc<CollectionsReader>, doc: Arc<dyn DocumentStorage>) -> Router {
    Router::new()
        .add(search())
        .add(dump_all())
        // .route("/:collection_id/documents/:document_id", get(get_doc_by_id))
        .with_state(Arc::new(ReaderAPIState {
            readers,
            doc,
        }))
}


#[endpoint(
    method = "POST",
    path = "/v0/reader/dump_all",
    description = "Dump"
)]
async fn dump_all(
    state: State<Arc<ReaderAPIState>>,
) -> impl IntoResponse {

    match state.readers.commit().await {
        Ok(_) => {}
        Err(e) => {
            error!("Error dumping all collections: {}", e);
            // e.chain()
            //     .skip(1)
            //     .for_each(|cause| error!("because: {}", cause));
        }
    }

    match state.doc.commit() {
        Ok(_) => {}
        Err(e) => {
            error!("Error dumping all documents: {}", e);
            // e.chain()
            //     .skip(1)
            //     .for_each(|cause| error!("because: {}", cause));
        }
    }

    axum::Json(())
}

#[endpoint(
    method = "POST",
    path = "/v0/collections/:id/search",
    description = "Search Endpoint"
)]
#[axum::debug_handler]
async fn search(
    Path(id): Path<String>,
    state: State<Arc<ReaderAPIState>>,
    Json(json): Json<SearchParams>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);

    let collection = state.readers.get_collection(collection_id).await;

    let collection = match collection {
        Some(collection) => collection,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "collection not found" })),
            ));
        }
    };

    let output = collection.search(json).await;

    match output {
        Ok(data) => Ok((StatusCode::OK, Json(data))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
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
