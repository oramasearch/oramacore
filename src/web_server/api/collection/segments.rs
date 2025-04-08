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
    collection_manager::sides::{segments::Segment, ReadSide, WriteSide},
    types::{ApiKey, CollectionId, DeleteSegmentParams, InsertSegmentParams, UpdateSegmentParams},
    web_server::api::collection::admin::print_error,
};

#[derive(Deserialize, IntoParams)]
struct ApiKeyQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize, IntoParams)]
struct GetSegmentQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
    #[serde(rename = "segment_id")]
    segment_id: String,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(get_segment_v1())
        .add(get_all_segments_v1())
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(insert_segment_v1())
        .add(delete_segment_v1())
        .add(update_segment_v1())
        .with_state(write_side)
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/segments/get",
    description = "Get a single segment by ID"
)]
async fn get_segment_v1(
    collection_id: CollectionId,
    Query(query): Query<GetSegmentQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let segment_id = query.segment_id;
    let read_api_key = query.api_key;

    match read_side
        .get_segment(read_api_key, collection_id, segment_id)
        .await
    {
        Ok(Some(segment)) => Ok((StatusCode::OK, Json(json!({ "segment": segment })))),
        Ok(None) => Ok((StatusCode::OK, Json(json!({ "segment": null })))),
        Err(e) => {
            print_error(&e, "Error getting segment");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/segments/all",
    description = "Get all segments in a collection"
)]
async fn get_all_segments_v1(
    collection_id: CollectionId,
    Query(query): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    match read_side
        .get_all_segments_by_collection(read_api_key, collection_id)
        .await
    {
        Ok(segments) => Ok((StatusCode::OK, Json(json!({ "segments": segments })))),
        Err(e) => {
            print_error(&e, "Error getting all segments");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/segments/insert",
    description = "Insert a new segment"
)]
async fn insert_segment_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertSegmentParams>,
) -> impl IntoResponse {
    let segment = Segment {
        id: params.id.unwrap_or(cuid2::create_id()),
        name: params.name.clone(),
        description: params.description.clone(),
        goal: params.goal.clone(),
    };

    match write_side
        .insert_segment(write_api_key, collection_id, segment.clone())
        .await
    {
        Ok(_) => Ok((
            StatusCode::CREATED,
            Json(json!({ "success": true, "id": segment.id, "segment": segment })),
        )),
        Err(e) => {
            print_error(&e, "Error inserting segment");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/segments/delete",
    description = "Deletes an existing segment"
)]
async fn delete_segment_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteSegmentParams>,
) -> impl IntoResponse {
    match write_side
        .delete_segment(write_api_key, collection_id, params.id)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            print_error(&e, "Error deleting segment");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/segments/update",
    description = "Updates an existing segment"
)]
async fn update_segment_v1(
    collection_id: CollectionId,
    write_api_key: ApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateSegmentParams>,
) -> impl IntoResponse {
    let segment = Segment {
        id: params.id.clone(),
        name: params.name.clone(),
        description: params.description.clone(),
        goal: params.goal.clone(),
    };

    match write_side
        .update_segment(write_api_key, collection_id, segment)
        .await
    {
        Ok(_) => Ok((StatusCode::OK, Json(json!({ "success": true })))),
        Err(e) => {
            print_error(&e, "Error updating segment");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            ))
        }
    }
}
