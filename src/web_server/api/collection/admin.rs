use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, patch, post},
    Json, Router,
};
use serde_json::json;
use tracing::error;

use crate::{
    collection_manager::{
        dto::{CollectionDTO, CreateCollectionOptionDTO},
        CollectionId, CollectionManager,
    },
    types::DocumentList,
};

pub fn apis() -> Router<Arc<CollectionManager>> {
    Router::<Arc<CollectionManager>>::new()
        .route("/", get(get_collections))
        .route("/:id", get(get_collection_by_id))
        .route("/", post(create_collection))
        .route("/:id/documents", patch(add_documents))
}

async fn get_collections(manager: State<Arc<CollectionManager>>) -> Json<Vec<CollectionDTO>> {
    let collections = manager.list().await;
    Json(collections)
}

async fn get_collection_by_id(
    Path(id): Path<String>,
    manager: State<Arc<CollectionManager>>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let collection = manager.get(collection_id).await;

    match collection {
        Some(collection) => Ok((StatusCode::OK, Json(collection.as_dto()))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "collection not found" })),
        )),
    }
}

async fn create_collection(
    manager: State<Arc<CollectionManager>>,
    Json(json): Json<CreateCollectionOptionDTO>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = match manager.create_collection(json).await {
        Ok(collection_id) => collection_id,
        Err(e) => {
            error!("Error creating collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ));
        }
    };
    Ok((
        StatusCode::CREATED,
        Json(json!({ "collection_id": collection_id })),
    ))
}

async fn add_documents(
    Path(id): Path<String>,
    manager: State<Arc<CollectionManager>>,
    Json(json): Json<DocumentList>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);

    let collection = match manager.get(collection_id).await {
        Some(collection) => collection,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "collection not found" })),
            ));
        }
    };

    let output = collection.insert_batch(json).await;

    match output {
        Ok(_) => Ok((
            StatusCode::OK,
            Json(json!({ "message": "documents added" })),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}
