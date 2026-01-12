use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use oramacore_lib::shelves::{Shelf, ShelfId};
use serde::Serialize;
use serde_json::json;

use crate::{
    collection_manager::sides::{
        read::{ReadError, ReadSide},
        write::{WriteError, WriteSide},
    },
    types::{ApiKey, CollectionId, WriteApiKey},
};

#[derive(Debug, Serialize)]
pub struct ShelfListResponse {
    pub shelves: Vec<String>,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/merchandising/shelves/{shelf_id}/get",
            get(get_merchandising_shelf),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/merchandising/shelves/insert",
            post(insert_merchandising_shelf),
        )
        .route(
            "/v1/collections/{collection_id}/merchandising/shelves/list",
            get(list_merchandising_shelves),
        )
        .route(
            "/v1/collections/{collection_id}/merchandising/shelves/{shelf_id}/delete",
            post(delete_merchandising_shelves),
        )
        .with_state(write_side)
}

async fn insert_merchandising_shelf(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(shelf): Json<Shelf<String>>,
) -> impl IntoResponse {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.insert_shelf(shelf).await?;

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "success": true })))
}

async fn delete_merchandising_shelves(
    Path((collection_id, shelf_id)): Path<(CollectionId, ShelfId)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    collection.delete_shelf(shelf_id).await?;

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "success": true })))
}

async fn list_merchandising_shelves(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    let collection = write_side
        .get_collection(collection_id, write_api_key)
        .await?;

    let shelf_list = collection.list_shelves().await?;

    Result::<Json<serde_json::Value>, WriteError>::Ok(Json(json!({ "data": shelf_list })))
}

async fn get_merchandising_shelf(
    Path((collection_id, shelf_id)): Path<(CollectionId, ShelfId)>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ApiKey,
) -> impl IntoResponse {
    let collection = read_side
        .get_collection(collection_id, read_api_key)
        .await?;

    let shelf_with_documents = collection.get_shelf_documents(shelf_id.clone()).await?;

    Result::<Json<serde_json::Value>, ReadError>::Ok(Json(json!({ "data": shelf_with_documents })))
}
