use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_extra::{headers, TypedHeader};
use axum_openapi3::{utoipa::IntoParams, *};
use http::StatusCode;
use redact::Secret;
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::{
        dto::{ApiKey, DeleteTriggerParams, InsertTriggerParams, UpdateTriggerParams},
        sides::{
            triggers::{parse_trigger_id, Trigger},
            ReadSide, WriteSide,
        },
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

#[derive(Deserialize, IntoParams, Debug)]
struct GetTriggerQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
    #[serde(rename = "trigger_id")]
    trigger_id: String,
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
    let read_api_key = query.api_key;

    match read_side
        .get_trigger(read_api_key, collection_id, trigger_id)
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
    let read_api_key = query.api_key;

    match read_side
        .get_all_triggers_by_collection(read_api_key, collection_id)
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
    TypedHeader(auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertTriggerParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side
        .insert_trigger(write_api_key, collection_id, params, None)
        .await
    {
        Ok(new_trigger) => Ok((
            StatusCode::OK,
            Json(json!({ "success": true, "id": new_trigger.id.clone(), "trigger": new_trigger })),
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
    TypedHeader(auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteTriggerParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    match write_side
        .delete_trigger(write_api_key, collection_id, params.id)
        .await
    {
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
    TypedHeader(auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateTriggerParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let trigger_id_parts = parse_trigger_id(params.id.clone());
    let write_api_key = ApiKey(Secret::new(auth.0.token().to_string()));

    if Some(trigger_id_parts.segment_id.clone()) != Some(params.segment_id.clone()) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(
                json!({ "error": "You can not update a segment ID. Please create a new trigger and link it to a new segment instead." }),
            ),
        ));
    }

    let trigger = Trigger {
        id: params.id,
        name: params.name.clone(),
        description: params.description.clone(),
        response: params.response.clone(),
        segment_id: trigger_id_parts.segment_id,
    };

    match write_side
        .update_trigger(write_api_key, collection_id, trigger.clone())
        .await
    {
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
