use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    collection_manager::sides::{read::ReadSide, write::WriteSide},
    types::{ApiKey, CollectionId, DocumentId, Shelf, ShelfData, ShelfName, WriteApiKey},
};

#[derive(Debug, Deserialize)]
pub struct CreateShelfRequest {
    pub name: ShelfName,
    pub documents: Vec<DocumentId>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateShelfRequest {
    pub documents: Vec<DocumentId>,
}

#[derive(Debug, Serialize)]
pub struct ShelfResponse {
    pub name: ShelfName,
    pub documents: Vec<DocumentId>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl From<Shelf> for ShelfResponse {
    fn from(shelf: Shelf) -> Self {
        ShelfResponse {
            name: shelf.name,
            documents: shelf.documents,
            created_at: shelf.created_at,
            updated_at: shelf.updated_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ShelfListResponse {
    pub shelves: Vec<String>,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/:collection_id/shelves",
            get(list_shelves_reader),
        )
        .route(
            "/v1/collections/:collection_id/shelves/:shelf_name",
            get(get_shelf_reader),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/:collection_id/shelves",
            post(create_shelf).get(list_shelves_writer),
        )
        .route(
            "/v1/collections/:collection_id/shelves/:shelf_name",
            get(get_shelf_writer)
                .put(update_shelf_writer)
                .delete(delete_shelf_writer),
        )
        .with_state(write_side)
}

async fn create_shelf(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(req): Json<CreateShelfRequest>,
) -> impl IntoResponse {
    let shelf_data = ShelfData {
        name: req.name,
        documents: req.documents,
    };

    write_side
        .create_shelf(write_api_key, collection_id, shelf_data)
        .await
        .map(|_| Json(json!({})))
}

async fn get_shelf_writer(
    Path((collection_id, shelf_name)): Path<(CollectionId, ShelfName)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    write_side
        .get_shelf(write_api_key, collection_id, shelf_name)
        .await
        .map(Json)
}

async fn list_shelves_writer(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    write_side
        .list_shelves(write_api_key, collection_id)
        .await
        .map(Json)
}

async fn update_shelf_writer(
    Path((collection_id, shelf_name)): Path<(CollectionId, ShelfName)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(req): Json<UpdateShelfRequest>,
) -> impl IntoResponse {
    let shelf_data = ShelfData {
        name: shelf_name,
        documents: req.documents,
    };

    write_side
        .update_shelf(write_api_key, collection_id, shelf_name, shelf_data)
        .await
        .map(|_| Json(json!({})))
}

async fn delete_shelf_writer(
    Path((collection_id, shelf_name)): Path<(CollectionId, ShelfName)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    write_side
        .delete_shelf(write_api_key, collection_id, shelf_name)
        .await
        .map(|_| Json(json!({})))
}

async fn get_shelf_reader(
    Path((collection_id, shelf_name)): Path<(CollectionId, ShelfName)>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ApiKey,
) -> impl IntoResponse {
    read_side
        .get_shelf(collection_id, shelf_name, read_api_key)
        .await
        .map(Json)
}

async fn list_shelves_reader(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ApiKey,
) -> impl IntoResponse {
    read_side
        .list_shelves(collection_id, read_api_key)
        .await
        .map(Json)
}
