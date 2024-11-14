
use std::sync::Arc;

use axum::{extract::{Path, State}, http::StatusCode, response::IntoResponse, routing::{get, patch, post}, Json, Router};
use collection_manager::{
    dto::{CollectionDTO, CreateCollectionOptionDTO, SearchParams},
    CollectionManager,
};
use serde_json::json;
use types::{CollectionId, DocumentList};

pub fn apis() -> Router<Arc<CollectionManager>> {
    Router::<Arc<CollectionManager>>::new()
        .route("/", get(get_collections))
        .route("/:id", get(get_collection_by_id))
        .route("/", post(create_collection))
        .route("/:id/documents", patch(add_documents))
        .route("/:id/search", post(search))
        .route("/:collection_id/documents/:document_id", get(get_doc_by_id))
}

async fn get_collections(manager: State<Arc<CollectionManager>>) -> Json<Vec<CollectionDTO>> {
    let collections = manager.list();
    Json(collections)
}

async fn get_collection_by_id(
    Path(id): Path<String>,
    manager: State<Arc<CollectionManager>>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let output = manager.get(collection_id, |collection| collection.as_dto());

    match output {
        Some(data) => Ok((StatusCode::OK, Json(data))),
        None => Err((StatusCode::NOT_FOUND, Json(json!({ "error": "collection not found" })))),
    }
}

async fn create_collection(
    manager: State<Arc<CollectionManager>>,
    Json(json): Json<CreateCollectionOptionDTO>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = match manager.create_collection(json) {
        Ok(collection_id) => collection_id,
        Err(e) => {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))));
        }
    };
    Ok((StatusCode::CREATED, Json(json!({ "collection_id": collection_id }))))
}

async fn add_documents(
    Path(id): Path<String>,
    manager: State<Arc<CollectionManager>>,
    Json(json): Json<DocumentList>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let output = manager.get(collection_id, |collection| {
        collection.insert_batch(json)
    });

    match output {
        Some(Ok(())) => Ok((StatusCode::OK, Json(json!({ "message": "documents added" })))),
        Some(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))),
        None => Err((StatusCode::NOT_FOUND, Json(json!({ "error": "collection not found" })))),
    }
}

async fn search(
    Path(id): Path<String>,
    manager: State<Arc<CollectionManager>>,
    Json(json): Json<SearchParams>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let output = manager.get(collection_id, |collection| {
        collection.search(json)
    });

    match output {
        Some(Ok(data)) => Ok((StatusCode::OK, Json(data))),
        Some(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))),
        None => Err((StatusCode::NOT_FOUND, Json(json!({ "error": "collection not found" })))),
    }
}

async fn get_doc_by_id(
    Path((collection_id, document_id)): Path<(String, String)>,
    manager: State<Arc<CollectionManager>>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(collection_id);
    let output = manager.get(collection_id, |collection| {
        collection.get_doc_by_unique_field("id".to_string(), document_id)
    });

    match output {
        Some(Ok(Some(data))) => Ok((StatusCode::OK, Json(data))),
        Some(Ok(None)) => Err((StatusCode::NOT_FOUND, Json(json!({ "error": "document not found" })))),
        Some(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))),
        None => Err((StatusCode::NOT_FOUND, Json(json!({ "error": "collection not found" })))),
    }
}
