use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json, Router};
use axum_openapi3::*;
use serde_json::json;
use tracing::info;

use crate::{
    collection_manager::sides::{write::WriteSide, ReplaceIndexReason},
    types::{
        ApiKey, CollectionId, CreateCollection, CreateIndexRequest, DeleteCollection,
        DeleteDocuments, DeleteIndex, DescribeCollectionResponse, DocumentList, IndexId,
        ListDocumentInCollectionRequest, ReindexConfig, ReplaceIndexRequest, UpdateDocumentRequest,
    },
    web_server::api::util::print_error,
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
) -> Result<Json<Vec<DescribeCollectionResponse>>, (StatusCode, impl IntoResponse)> {
    let collections = match write_side.list_collections(master_api_key).await {
        Ok(collections) => collections,
        Err(e) => {
            print_error(&e, "Error listing collections");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ));
        }
    };
    Ok(Json(collections))
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
) -> Result<Json<DescribeCollectionResponse>, (StatusCode, impl IntoResponse)> {
    let collection_dto = write_side
        .get_collection_dto(master_api_key, collection_id)
        .await;

    match collection_dto {
        Ok(Some(collection_dto)) => Ok(Json(collection_dto)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "collection not found" })),
        )),
        Err(e) => {
            print_error(&e, "Error getting collection by id");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
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
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    match write_side.create_collection(master_api_key, json).await {
        Ok(collection_id) => collection_id,
        Err(e) => {
            // If master_api_key is wrong we return 500
            // This is not correct, we should return a different http status code
            // TODO: do it
            print_error(&e, "Error creating collection");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ));
        }
    };
    Ok((StatusCode::CREATED, Json(json!({ "collection_id": () }))))
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
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = json.id;
    match write_side
        .delete_collection(master_api_key, collection_id)
        .await
    {
        Ok(_) => {}
        Err(e) => {
            // If master_api_key is wrong we return 500
            // This is not correct, we should return a different http status code
            // TODO: do it
            print_error(&e, "Error deleting collection");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ));
        }
    };

    Ok((StatusCode::OK, Json(json!({}))))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/list",
    description = "Return all documents in a collection"
)]
async fn list_document_in_collection(
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
    Json(json): Json<ListDocumentInCollectionRequest>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = json.id;

    match write_side.list_document(write_api_key, collection_id).await {
        Ok(docs) => Ok(Json(docs)),
        Err(e) => {
            print_error(&e, "Error in listing collection");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/create",
    description = "Create index inside a collection"
)]
async fn create_index(
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
    collection_id: CollectionId,
    Json(json): Json<CreateIndexRequest>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    match write_side
        .create_index(write_api_key, collection_id, json)
        .await
    {
        Ok(_) => Ok(Json(json!({}))),
        Err(e) => {
            print_error(&e, "Error creating index");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/indexes/delete",
    description = "Create index inside a collection"
)]
async fn delete_index(
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
    collection_id: CollectionId,
    Json(json): Json<DeleteIndex>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    match write_side
        .delete_index(write_api_key, collection_id, json.id)
        .await
    {
        Ok(_) => Ok(Json(json!({}))),
        Err(e) => {
            print_error(&e, "Error deleting index");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
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
    write_api_key: ApiKey,
    Json(json): Json<DocumentList>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    info!("Adding documents to collection {:?}", collection_id);
    match write_side
        .insert_documents(write_api_key, collection_id, index_id, json)
        .await
    {
        Ok(r) => Ok((StatusCode::OK, Json(r))),
        Err(e) => {
            print_error(&e, "Error adding documents to collection");
            Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ))
        }
    }
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
    write_api_key: ApiKey,
    Json(json): Json<UpdateDocumentRequest>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    info!("Update documents to collection {:?}", collection_id);
    match write_side
        .update_documents(write_api_key, collection_id, index_id, json)
        .await
    {
        Ok(r) => Ok((StatusCode::OK, Json(r))),
        Err(e) => {
            print_error(&e, "Error update documents to collection");
            Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ))
        }
    }
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
    write_api_key: ApiKey,
    Json(json): Json<DeleteDocuments>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    info!("Delete documents to collection {:?}", collection_id);
    match write_side
        .delete_documents(write_api_key, collection_id, index_id, json)
        .await
    {
        Ok(_) => {
            info!("Documents deleted to collection");
        }
        Err(e) => {
            print_error(&e, "Error deleting documents to collection");
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ));
        }
    };

    Ok((
        StatusCode::OK,
        Json(json!({ "message": "documents deleted" })),
    ))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/reindex",
    description = "Reindex collection after a runtime change"
)]
async fn reindex(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
    Json(json): Json<ReindexConfig>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    match write_side
        .reindex(
            write_api_key,
            collection_id,
            json.language,
            json.embedding_model.0,
            json.reference,
        )
        .await
    {
        Ok(_) => {
            info!("Collection reindexed");
        }
        Err(e) => {
            print_error(&e, "Error reindexing collection");
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ));
        }
    };

    Ok((
        StatusCode::OK,
        Json(json!({ "message": "collection reindexed" })),
    ))
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
    write_api_key: ApiKey,
    Json(json): Json<CreateIndexRequest>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    match write_side
        .create_temp_index(write_api_key, collection_id, index_id, json)
        .await
    {
        Ok(_) => {
            info!(" collection");
        }
        Err(e) => {
            print_error(&e, "Error reindexing collection");
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ));
        }
    };

    Ok((
        StatusCode::OK,
        Json(json!({ "message": "temp collection created" })),
    ))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/replace-index",
    description = "Substitute index for a collection"
)]
async fn replace_index(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: ApiKey,
    Json(json): Json<ReplaceIndexRequest>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    match write_side
        .replace_index(
            write_api_key,
            collection_id,
            json,
            ReplaceIndexReason::IndexResynced,
        )
        .await
    {
        Ok(_) => {}
        Err(e) => {
            print_error(&e, "Error index replacement");
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ));
        }
    };

    Ok((StatusCode::OK, Json(json!({ "message": "Index replaced" }))))
}
