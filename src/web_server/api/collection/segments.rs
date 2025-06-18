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
    collection_manager::sides::{read::ReadSide, segments::Segment, write::WriteSide},
    types::{
        ApiKey, CollectionId, DeleteSegmentParams, InsertSegmentParams, UpdateSegmentParams,
        WriteApiKey,
    },
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

    let segment_interface = read_side
        .get_segments_manager(read_api_key, collection_id)
        .await?;

    segment_interface.get(segment_id).await.map(|r| match r {
        Some(segment) => Json(json!({ "segment": segment })),
        None => Json(json!({ "segment": null })),
    })
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
    let segment_interface = read_side
        .get_segments_manager(read_api_key, collection_id)
        .await?;

    segment_interface
        .list()
        .await
        .map(|segments| (StatusCode::OK, Json(json!({ "segments": segments }))))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/segments/insert",
    description = "Insert a new segment"
)]
async fn insert_segment_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertSegmentParams>,
) -> impl IntoResponse {
    let segment = Segment {
        id: params.id.unwrap_or(cuid2::create_id()),
        name: params.name.clone(),
        description: params.description.clone(),
        goal: params.goal.clone(),
    };

    let segment_interface = write_side
        .get_segments_manager(write_api_key, collection_id)
        .await?;

    segment_interface
        .insert(segment.clone())
        .await
        .map(|_| Json(json!({ "success": true, "id": segment.id, "segment": segment })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/segments/delete",
    description = "Deletes an existing segment"
)]
async fn delete_segment_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteSegmentParams>,
) -> impl IntoResponse {
    let segment_interface = write_side
        .get_segments_manager(write_api_key, collection_id)
        .await?;
    segment_interface
        .delete(params.id)
        .await
        .map(|_| Json(json!({ "success": true })))
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/segments/update",
    description = "Updates an existing segment"
)]
async fn update_segment_v1(
    collection_id: CollectionId,
    write_api_key: WriteApiKey,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateSegmentParams>,
) -> impl IntoResponse {
    let segment = Segment {
        id: params.id.clone(),
        name: params.name.clone(),
        description: params.description.clone(),
        goal: params.goal.clone(),
    };

    let segment_interface = write_side
        .get_segments_manager(write_api_key, collection_id)
        .await?;

    segment_interface
        .update(segment)
        .await
        .map(|_| Json(json!({ "success": true })))
}
