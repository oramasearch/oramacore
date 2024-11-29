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
        dto::{CollectionDTO, CreateCollectionOptionDTO}, sides::write::CollectionsWriter, CollectionId,
    },
    types::DocumentList,
};

pub fn apis(writers: Arc<CollectionsWriter>) -> Router {
    Router::new()
        .route("/", get(get_collections))
        .route("/:id", get(get_collection_by_id))
        .route("/", post(create_collection))
        .route("/:id/documents", patch(add_documents))
        .with_state(writers)
}

async fn get_collections(writer: State<Arc<CollectionsWriter>>) -> Json<Vec<CollectionDTO>> {
    let collections = writer.list();
    Json(collections)
}

async fn get_collection_by_id(
    Path(id): Path<String>,
    writer: State<Arc<CollectionsWriter>>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let collection_dto = writer.get_collection_dto(collection_id);

    match collection_dto {
        Some(collection_dto) => Ok((StatusCode::OK, Json(collection_dto))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "collection not found" })),
        )),
    }
}

async fn create_collection(
    writer: State<Arc<CollectionsWriter>>,
    Json(json): Json<CreateCollectionOptionDTO>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = match writer.create_collection(json).await {
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
    writer: State<Arc<CollectionsWriter>>,
    Json(json): Json<DocumentList>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);

    match writer.write(collection_id, json).await {
        Ok(_) => {}
        Err(e) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ));
        }
    };

    Ok((
        StatusCode::OK,
        Json(json!({ "message": "documents added" })),
    ))
}
