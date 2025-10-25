use std::sync::Arc;

use axum::{
    extract::{FromRef, Path, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tracing::info;

use crate::{
    collection_manager::sides::{
        write::{jwt_manager::JwtManager, WriteSide},
        ReplaceIndexReason,
    },
    types::{
        ApiKey, CollectionId, CreateCollection, CreateIndexRequest, DeleteCollection,
        DeleteDocuments, DeleteIndex, DocumentList, IndexId, ListDocumentInCollectionRequest,
        ReindexConfig, ReplaceIndexRequest, UpdateDocumentRequest, WriteApiKey,
    },
};

impl FromRef<Arc<WriteSide>> for JwtManager {
    fn from_ref(app_state: &Arc<WriteSide>) -> JwtManager {
        app_state.get_jwt_manager()
    }
}

pub fn apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route("/v1/collections", get(get_collections))
        .route("/v1/collections/{collection_id}", get(get_collection_by_id))
        .route("/v1/collections/create", post(create_collection))
        .route("/v1/collections/delete", post(delete_collection))
        .route("/v1/collections/list", post(list_document_in_collection))
        .route(
            "/v1/collections/{collection_id}/indexes/create",
            post(create_index),
        )
        .route(
            "/v1/collections/{collection_id}/indexes/delete",
            post(delete_index),
        )
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/datasource/{datasource_id}/delete",
            post(remove_datasource_from_index),
        )
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/insert",
            post(add_documents),
        )
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/documents/upsert",
            post(update_documents),
        )
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/delete",
            post(delete_documents),
        )
        .route("/v1/collections/{collection_id}/reindex", post(reindex))
        .route(
            "/v1/collections/{collection_id}/indexes/{index_id}/create-temporary-index",
            post(create_temp_index),
        )
        .route(
            "/v1/collections/{collection_id}/replace-index",
            post(replace_index),
        )
        .with_state(write_side)
}

async fn get_collections(
    write_side: State<Arc<WriteSide>>,
    master_api_key: ApiKey,
) -> impl IntoResponse {
    write_side.list_collections(master_api_key).await.map(Json)
}

async fn get_collection_by_id(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    master_api_key: ApiKey,
) -> impl IntoResponse {
    write_side
        .get_collection_dto(master_api_key, collection_id)
        .await
        .map(Json)
}

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

async fn create_index(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<CreateIndexRequest>,
) -> impl IntoResponse {
    write_side
        .create_index(write_api_key, collection_id, json)
        .await
        .map(|_| Json(json!({})))
}

async fn delete_index(
    Path(collection_id): Path<CollectionId>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<DeleteIndex>,
) -> impl IntoResponse {
    write_side
        .delete_index(write_api_key, collection_id, json.id)
        .await
        .map(|_| Json(json!({})))
}

async fn remove_datasource_from_index(
    Path((collection_id, index_id, datasource_id)): Path<(CollectionId, IndexId, IndexId)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    write_side
        .remove_datasource_from_index(write_api_key, collection_id, index_id, datasource_id)
        .await
        .map(|_| Json(json!({})))
}

async fn add_documents(
    Path((collection_id, index_id)): Path<(CollectionId, IndexId)>,
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

async fn update_documents(
    Path((collection_id, index_id)): Path<(CollectionId, IndexId)>,
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

async fn delete_documents(
    Path((collection_id, index_id)): Path<(CollectionId, IndexId)>,
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

async fn reindex(
    Path(collection_id): Path<CollectionId>,
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

async fn create_temp_index(
    Path((collection_id, index_id)): Path<(CollectionId, IndexId)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(json): Json<CreateIndexRequest>,
) -> impl IntoResponse {
    write_side
        .create_temp_index(write_api_key, collection_id, index_id, json)
        .await
        .map(|_| Json(json!({ "message": "temp collection created" })))
}

async fn replace_index(
    Path(collection_id): Path<CollectionId>,
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
