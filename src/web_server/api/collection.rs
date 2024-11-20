use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, patch, post},
    Json, Router,
};
use serde_json::json;

use crate::{
    collection_manager::{
        dto::{CollectionDTO, CreateCollectionOptionDTO, SearchParams},
        CollectionManager,
    },
    types::{CollectionId, DocumentList},
};

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

async fn search(
    Path(id): Path<String>,
    manager: State<Arc<CollectionManager>>,
    Json(json): Json<SearchParams>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);

    let collection = manager.get(collection_id).await;

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

async fn get_doc_by_id(
    Path((collection_id, document_id)): Path<(String, String)>,
    manager: State<Arc<CollectionManager>>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(collection_id);

    let collection = manager.get(collection_id).await;

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
