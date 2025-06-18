use std::sync::Arc;

use axum::{extract::State, response::IntoResponse, Json, Router};
use axum_openapi3::*;
use serde_json::json;
use tracing::info;

use crate::{
    collection_manager::sides::{write::WriteSide, ReplaceIndexReason},
    types::{
        ApiKey, CollectionId, CreateCollection, CreateIndexRequest, DeleteCollection,
        DeleteDocuments, DeleteIndex, DocumentList, IndexId, ListDocumentInCollectionRequest,
        ReindexConfig, ReplaceIndexRequest, UpdateDocumentRequest, WriteApiKey,
    },
};

pub fn apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(get_collections())
        .add(get_collection_by_id())
        .add(create_collection())
        .add(add_documents())
        .add(delete_documents())
        .add(delete_collection())
        .add(create_index())
        .add(list_document_in_collection())
        .add(delete_index())
        .add(reindex())
        .add(create_temp_index())
        .add(replace_index())
        .with_state(write_side)
}

#[endpoint(
    method = "GET",
    path = "/v1/collections",
    description = "List all collections"
)]
async fn get_collections(
    write_side: State<Arc<WriteSide>>,
    master_api_key: ApiKey,
) -> impl IntoResponse {
    write_side.list_collections(master_api_key).await.map(Json)
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}",
    description = "Get a collection by id"
)]
async fn get_collection_by_id(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    master_api_key: ApiKey,
) -> impl IntoResponse {
    write_side
        .get_collection_dto(master_api_key, collection_id)
        .await
        .map(Json)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/create",
    description = "Create a collection"
)]
async fn create_collection(
    write_side: State<Arc<WriteSide>>,
    master_api_key: ApiKey,
    Json(json): Json<CreateCollection>,
) -> impl IntoResponse {
    write_side
        .create_collection(master_api_key, json)
        .await
        .map(Json)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/delete",
    description = "Delete a collection"
)]
async fn delete_collection(
    write_side: State<Arc<WriteSide>>,
    master_api_key: ApiKey,
    Json(json): Json<DeleteCollection>,
) -> impl IntoResponse {
    let collection_id = json.id;
    write_side
        .delete_collection(master_api_key, collection_id)
        .await
        .map(|_| Json(json!({})))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/list",
    description = "Return all documents in a collection"
)]
async fn list_document_in_collection(
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<ListDocumentInCollectionRequest>,
) -> impl IntoResponse {
    let collection_id = json.id;
    write_side
        .list_document(write_api_key, collection_id)
        .await
        .map(Json)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/create",
    description = "Create index inside a collection"
)]
async fn create_index(
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    collection_id: CollectionId,
    Json(json): Json<CreateIndexRequest>,
) -> impl IntoResponse {
    write_side
        .create_index(write_api_key, collection_id, json)
        .await
        .map(|_| Json(json!({})))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/delete",
    description = "Create index inside a collection"
)]
async fn delete_index(
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    collection_id: CollectionId,
    Json(json): Json<DeleteIndex>,
) -> impl IntoResponse {
    write_side
        .delete_index(write_api_key, collection_id, json.id)
        .await
        .map(|_| Json(json!({})))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/{index_id}/insert",
    description = "Add documents to a collection"
)]
async fn add_documents(
    collection_id: CollectionId,
    index_id: IndexId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<DocumentList>,
) -> impl IntoResponse {
    info!("Adding documents to collection {:?}", collection_id);
    write_side
        .insert_documents(write_api_key, collection_id, index_id, json)
        .await
        .map(Json)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/{index_id}/update",
    description = "Update documents to an index"
)]
async fn update_documents(
    collection_id: CollectionId,
    index_id: IndexId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<UpdateDocumentRequest>,
) -> impl IntoResponse {
    info!("Update documents to collection {:?}", collection_id);
    write_side
        .update_documents(write_api_key, collection_id, index_id, json)
        .await
        .map(Json)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/{index_id}/delete",
    description = "Delete documents from a collection"
)]
async fn delete_documents(
    collection_id: CollectionId,
    index_id: IndexId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<DeleteDocuments>,
) -> impl IntoResponse {
    info!("Delete documents to collection {:?}", collection_id);
    write_side
        .delete_documents(write_api_key, collection_id, index_id, json)
        .await
        .map(|_| Json(json!({ "message": "documents deleted" })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/reindex",
    description = "Reindex collection after a runtime change"
)]
async fn reindex(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<ReindexConfig>,
) -> impl IntoResponse {
    write_side
        .reindex(
            write_api_key,
            collection_id,
            json.language,
            json.embedding_model.0,
            json.reference,
        )
        .await
        .map(|_| Json(json!({ "message": "collection reindexed" })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/{index_id}/create-temporary-index",
    description = "Create temporary index for a collection"
)]
async fn create_temp_index(
    collection_id: CollectionId,
    index_id: IndexId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<CreateIndexRequest>,
) -> impl IntoResponse {
    write_side
        .create_temp_index(write_api_key, collection_id, index_id, json)
        .await
        .map(|_| Json(json!({ "message": "temp collection created" })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/replace-index",
    description = "Substitute index for a collection"
)]
async fn replace_index(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<ReplaceIndexRequest>,
) -> impl IntoResponse {
    write_side
        .replace_index(
            write_api_key,
            collection_id,
            json,
            ReplaceIndexReason::IndexResynced,
        )
        .await
        .map(|_| Json(json!({ "message": "Index replaced" })))
}
