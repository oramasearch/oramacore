use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_extra::{headers, TypedHeader};
use axum_openapi3::{utoipa::IntoParams, *};
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::{
        dto::{ApiKey, DeleteTriggerParams, InsertTriggerParams, UpdateTriggerParams},
        sides::{triggers::Trigger, ReadSide, WriteSide},
    },
    types::CollectionId,
};

type AuthorizationBearerHeader =
    TypedHeader<headers::Authorization<headers::authorization::Bearer>>;

#[derive(Deserialize, IntoParams)]
struct ApiKeyQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize, IntoParams)]
struct GetTriggerQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
    #[serde(rename = "trigger_id")]
    trigger_id: String,
    #[serde(rename = "segment_id")]
    segment_id: Option<String>,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(get_trigger_v1())
        .add(get_all_triggers_v1())
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(insert_trigger_v1())
        .add(delete_trigger_v1())
        .add(update_trigger_v1())
        .with_state(write_side)
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{id}/triggers/get",
    description = "Get a single trigger by ID"
)]
async fn get_trigger_v1(
    Path(id): Path<String>,
    Query(query): Query<GetTriggerQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let trigger_id = query.trigger_id;
    let segment_id = query.segment_id;

    match read_side
        .get_trigger(collection_id, trigger_id, segment_id)
        .await
    {
        Ok(Some(trigger)) => Ok((StatusCode::OK, Json(json!({ "trigger": trigger })))),
        Ok(None) => Ok((StatusCode::OK, Json(json!({ "trigger": null })))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{id}/triggers/all",
    description = "Get all triggers in a collection"
)]
async fn get_all_triggers_v1(
    Path(id): Path<String>,
    Query(query): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    match read_side
        .get_all_triggers_by_collection(collection_id)
        .await
    {
        Ok(triggers) => Ok((StatusCode::OK, Json(json!({ "triggers": triggers })))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/triggers/insert",
    description = "Insert a new trigger"
)]
async fn insert_trigger_v1(
    Path(id): Path<String>,
    TypedHeader(_auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertTriggerParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    let trigger = Trigger {
        id: params.id.unwrap_or(cuid2::create_id()),
        name: params.name.clone(),
        description: params.description.clone(),
        response: params.response.clone(),
        segment_id: params.segment_id.clone(),
    };

    match write_side
        .insert_trigger(collection_id, trigger.clone())
        .await
    {
        Ok(_) => Ok((
            StatusCode::OK,
            Json(json!({ "success": true, "id": trigger.id, "trigger": trigger })),
        )),
        Err(e) => {
            e.chain()
                .skip(1)
                .for_each(|cause| println!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/triggers/delete",
    description = "Deletes an existing trigger"
)]
async fn delete_trigger_v1(
    Path(id): Path<String>,
    TypedHeader(_auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteTriggerParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    match write_side.delete_trigger(collection_id, params.id).await {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            e.chain()
                .skip(1)
                .for_each(|cause| println!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/triggers/update",
    description = "Updates an existing trigger"
)]
async fn update_trigger_v1(
    Path(id): Path<String>,
    TypedHeader(_auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateTriggerParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    let segment_id: Option<String> = params
        .id
        .clone()
        .split(":")
        .find(|s| s.starts_with("s_"))
        .map(|s| s.to_string());

    let trigger = Trigger {
        id: params.id.clone(),
        name: params.name.clone(),
        description: params.description.clone(),
        response: params.response.clone(),
        segment_id,
    };

    match write_side.update_trigger(collection_id, trigger).await {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            e.chain()
                .skip(1)
                .for_each(|cause| println!("because: {}", cause));
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}
