use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{utoipa::IntoParams, *};
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;

use crate::{
    collection_manager::sides::{
        read::ReadSide,
        triggers::{parse_trigger_id, Trigger},
        write::WriteSide,
    },
    types::{ApiKey, CollectionId, DeleteTriggerParams, InsertTriggerParams, UpdateTriggerParams},
    web_server::api::util::print_error,
};

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
    path = "/v1/collections/{collection_id}/triggers/get",
    description = "Get a single trigger by ID"
)]
async fn get_trigger_v1(
    collection_id: CollectionId,
    Query(query): Query<GetTriggerQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let trigger_id = query.trigger_id;
    let read_api_key = query.api_key;

    match read_side
        .get_trigger(read_api_key, collection_id, trigger_id)
        .await
    {
        Ok(Some(trigger)) => match parse_trigger_id(trigger.id.clone()) {
            Some(parsed_trigger_id) => Ok((
                StatusCode::OK,
                Json(json!({ "success": true, "trigger": Trigger {
                            id: parsed_trigger_id.trigger_id,
                            ..trigger
                        } })),
            )),
            None => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "Failed to parse trigger ID" })),
            )),
        },
        Ok(None) => Ok((StatusCode::OK, Json(json!({ "trigger": null })))),
        Err(e) => {
            print_error(&e, "Error getting trigger");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/triggers/all",
    description = "Get all triggers in a collection"
)]
async fn get_all_triggers_v1(
    collection_id: CollectionId,
    Query(query): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    match read_side
        .get_all_triggers_by_collection(read_api_key, collection_id)
        .await
    {
        Ok(triggers) => {
            let mut output = vec![];
            for trigger in triggers.iter() {
                match parse_trigger_id(trigger.id.clone()) {
                    Some(parsed_trigger_id) => {
                        output.push(Trigger {
                            id: parsed_trigger_id.trigger_id,
                            ..trigger.clone()
                        });
                    }
                    None => {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": "Failed to parse trigger ID" })),
                        ));
                    }
                }
            }

            Ok((StatusCode::OK, Json(json!({ "triggers": output }))))
        }
        Err(e) => {
            print_error(&e, "Error getting all triggers");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/triggers/insert",
    description = "Insert a new trigger"
)]
async fn insert_trigger_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertTriggerParams>,
) -> impl IntoResponse {
    let trigger = Trigger {
        id: params.id.unwrap_or(cuid2::create_id()),
        name: params.name.clone(),
        description: params.description.clone(),
        response: params.response.clone(),
        segment_id: params.segment_id.clone(),
    };

    write_side
        .insert_trigger(
            write_api_key,
            collection_id,
            trigger.clone(),
            Some(trigger.id),
        )
        .await
        .map(|new_trigger| {
            match parse_trigger_id(new_trigger.id.clone()) {
                Some(parsed_trigger_id) => Ok((
                    StatusCode::CREATED,
                    Json(
                        json!({ "success": true, "id": parsed_trigger_id.trigger_id, "trigger": Trigger {
                            id: parsed_trigger_id.trigger_id,
                            ..new_trigger
                        } }),
                    ),
                )),
                None => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "Failed to parse trigger ID" })),
                )),
            }
        })
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/triggers/delete",
    description = "Deletes an existing trigger"
)]
async fn delete_trigger_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteTriggerParams>,
) -> impl IntoResponse {
    write_side
        .delete_trigger(write_api_key, collection_id, params.id)
        .await
        .map(|_| Json(json!({ "success": true })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/triggers/update",
    description = "Updates an existing trigger"
)]
async fn update_trigger_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateTriggerParams>,
) -> impl IntoResponse {
    match write_side
        .get_trigger(write_api_key, collection_id, params.id.clone())
        .await
    {
        Ok(existing_trigger) => {
            // Make sure we don't update the segment ID. Not supported for now.
            match (
                existing_trigger.segment_id.as_ref(),
                params.segment_id.as_ref(),
            ) {
                (Some(existing), Some(new)) if existing != new => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(
                            json!({ "error": "You can not update a segment linked to a trigger." }),
                        ),
                    ));
                }
                _ => {}
            }

            // Determine which segment_id to use
            // If params.segment_id is None but existing_trigger has one, use the existing one
            let segment_id = match (
                params.segment_id.clone(),
                existing_trigger.segment_id.clone(),
            ) {
                (Some(new_id), _) => Some(new_id),
                (None, Some(existing_id)) => Some(existing_id),
                (None, None) => None,
            };

            let trigger = Trigger {
                id: params.id.clone(),
                name: params.name.clone(),
                description: params.description.clone(),
                response: params.response.clone(),
                segment_id,
            };

            let update_result = write_side
                .update_trigger(write_api_key, collection_id, trigger)
                .await;

            match update_result {
                Ok(updated_trigger) => match updated_trigger {
                    Some(trigger) => Ok((
                        StatusCode::OK,
                        Json(json!({ "success": true, "trigger": trigger })),
                    )),
                    None => Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": "Unable to get the trigger after updating it." })),
                    )),
                },
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e.to_string() })),
                )),
            }
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}
