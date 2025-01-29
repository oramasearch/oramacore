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
        dto::{ApiKey, CollectionDTO, CreateCollection},
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
        .add(dump_all())
        .with_state(write_side)
}

#[endpoint(
    method = "POST",
    path = "/v0/writer/dump_all",
    description = "List all collections"
)]
async fn dump_all(write_side: State<Arc<WriteSide>>) -> impl IntoResponse {
    match write_side.commit().await {
        Ok(_) => {}
        Err(e) => {
            error!("Error dumping all collections: {}", e);
            e.chain()
                .skip(1)
                .for_each(|cause| error!("because: {}", cause));
        }
    }
    // writer.commit(data_dir)

    // Json(collections)
    Json(())
}

#[endpoint(
    method = "GET",
    path = "/v0/collections",
    description = "List all collections"
)]
async fn get_collections(write_side: State<Arc<WriteSide>>) -> Json<Vec<CollectionDTO>> {
    let collections = write_side.list_collections().await;
    Json(collections)
}

#[endpoint(
    method = "GET",
    path = "/v0/collections/{id}",
    description = "Get a collection by id"
)]
async fn get_collection_by_id(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
) -> Result<Json<CollectionDTO>, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);
    let collection_dto = write_side.get_collection_dto(collection_id).await;

    match collection_dto {
        Some(collection_dto) => Ok(Json(collection_dto)),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "collection not found" })),
        )),
    }
}

type AuthorizationBearerHeader = TypedHeader<headers::Authorization<headers::authorization::Bearer>>;

#[endpoint(
    method = "POST",
    path = "/v0/collections",
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
    method = "PATCH",
    path = "/v0/collections/{id}/documents",
    description = "Add documents to a collection"
)]
async fn add_documents(
    Path(id): Path<String>,
    write_side: State<Arc<WriteSide>>,
    TypedHeader(auth): AuthorizationBearerHeader,
    Json(json): Json<DocumentList>,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let collection_id = CollectionId(id);

    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    info!("Adding documents to collection {:?}", collection_id);
    match write_side.write(write_api_key, collection_id, json).await {
        Ok(_) => {
            info!("Documents added to collection");
        }
        Err(e) => {
            error!("Error adding documents to collection: {}", e);
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
        Json(json!({ "message": "documents added" })),
    ))
}
