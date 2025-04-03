use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json, Router,
};
use axum_extra::{headers, TypedHeader};
use axum_openapi3::*;
use redact::Secret;
use serde_json::json;
use tracing::{error, info};

use crate::{
    collection_manager::{
        dto::{
            ApiKey, CollectionDTO, CreateCollection, CreateCollectionFrom, DeleteDocuments,
            ReindexConfig, SwapCollections,
        },
        sides::WriteSide,
    },
    types::{CollectionId, DocumentList},
};

pub fn apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(get_collections())
        .add(get_collection_by_id())
        .add(create_collection())
        .add(add_documents())
        .add(delete_documents())
        .add(delete_collection())
        .add(reindex())
        .add(create_collection_from())
        .add(swap_collections())
        .with_state(write_side)
}

type AuthorizationBearerHeader =
    TypedHeader<headers::Authorization<headers::authorization::Bearer>>;

#[endpoint(
    method = "GET",
    path = "/v1/collections",
    description = "List all collections"
)]
async fn get_collections(
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
) -> Result<Json<Vec<CollectionDTO>>, (StatusCode, impl IntoResponse)> {
    let master_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    let collections = match write_side.list_collections(master_api_key).await {
        Ok(collections) => collections,
        Err(e) => {
            error!("Error listing collections: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
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
    path = "/v1/collections/{id}",
    description = "Get a collection by id"
)]
async fn get_collection_by_id(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
) -> Result<Json<CollectionDTO>, (StatusCode, impl IntoResponse)> {
    let master_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    let collection_id = CollectionId::from(id);
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
            error!("Error get collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
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
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(json): Json<CreateCollection>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let master_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side.create_collection(master_api_key, json).await {
        Ok(collection_id) => collection_id,
        Err(e) => {
            // If master_api_key is wrong we return 500
            // This is not correct, we should return a different http status code
            // TODO: do it
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
    Ok((StatusCode::CREATED, Json(json!({ "collection_id": () }))))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/delete",
    description = "Delete a collection"
)]
async fn delete_collection(
    write_side: State<Arc<WriteSide>>,
    Path(id): Path<String>,
    TypedHeader(auth): AuthorizationBearerHeader,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId::from(id);
    let master_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side
        .delete_collection(master_api_key, collection_id)
        .await
    {
        Ok(_) => {}
        Err(e) => {
            // If master_api_key is wrong we return 500
            // This is not correct, we should return a different http status code
            // TODO: do it
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

    Ok((StatusCode::OK, Json(json!({}))))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/insert",
    description = "Add documents to a collection"
)]
async fn add_documents(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(json): Json<DocumentList>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId::from(id);

    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    info!("Adding documents to collection {:?}", collection_id);
    match write_side
        .insert_documents(write_api_key, collection_id, json)
        .await
    {
        Ok(r) => Ok((StatusCode::OK, Json(r))),
        Err(e) => {
            error!("{e:?}");
            error!("Error adding documents to collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
            Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/create-from",
    description = "Create a collection with same configuration as another collection"
)]
async fn create_collection_from(
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(json): Json<CreateCollectionFrom>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side.create_collection_from(write_api_key, json).await {
        Ok(collection_id) => Ok((
            StatusCode::OK,
            Json(json!({ "collection_id": collection_id })),
        )),
        Err(e) => {
            error!("{e:?}");
            error!("Error creating collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("{}", e) })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/swap",
    description = "Swap two collections"
)]
async fn swap_collections(
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(json): Json<SwapCollections>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side.swap_collections(write_api_key, json).await {
        Ok(collection_id) => Ok((
            StatusCode::OK,
            Json(json!({ "collection_id": collection_id })),
        )),
        Err(e) => {
            error!("{e:?}");
            error!("Error swapping collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("{}", e) })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/delete",
    description = "Delete documents from a collection"
)]
async fn delete_documents(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(json): Json<DeleteDocuments>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId::from(id);

    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    info!("Delete documents to collection {:?}", collection_id);
    match write_side
        .delete_documents(write_api_key, collection_id, json)
        .await
    {
        Ok(_) => {
            info!("Documents deleted to collection");
        }
        Err(e) => {
            error!("Error deleting documents to collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
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
    path = "/v1/collections/{id}/reindex",
    description = "Reindex documents of a collection"
)]
async fn reindex(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(json): Json<ReindexConfig>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId::from(id);

    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    info!("Reindex collection {:?}", collection_id);
    match write_side.reindex(write_api_key, collection_id, json).await {
        Ok(_) => {
            info!("Done");
        }
        Err(e) => {
            error!("Error reindexing documents to collection: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("collection not found {}", e) })),
            ));
        }
    };

    Ok((
        StatusCode::OK,
        Json(json!({ "message": "collection re-indexed" })),
    ))
}
