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
        dto::{ApiKey, DeleteSegmentParams, InsertSegmentParams, UpdateSegmentParams},
        sides::{segments::Segment, ReadSide, WriteSide},
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
    path = "/v1/collections/{id}/segments/get",
    description = "Get a single segment by ID"
)]
async fn get_segment_v1(
    Path(id): Path<String>,
    Query(query): Query<GetSegmentQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);
    let segment_id = query.segment_id;

    match read_side.get_segment(collection_id, segment_id).await {
        Ok(Some(segment)) => Ok((StatusCode::OK, Json(json!({ "segment": segment })))),
        Ok(None) => Ok((StatusCode::OK, Json(json!({ "segment": null })))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{id}/segments/all",
    description = "Get all segments in a collection"
)]
async fn get_all_segments_v1(
    Path(id): Path<String>,
    Query(query): Query<ApiKeyQueryParams>,
    read_side: State<Arc<ReadSide>>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    match read_side
        .get_all_segments_by_collection(collection_id)
        .await
    {
        Ok(segments) => Ok((StatusCode::OK, Json(json!({ "segments": segments })))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{id}/segments/insert",
    description = "Insert a new segment"
)]
async fn insert_segment_v1(
    Path(id): Path<String>,
    TypedHeader(_auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<InsertSegmentParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    let segment = Segment {
        id: params.id.unwrap_or(cuid2::create_id()),
        name: params.name.clone(),
        description: params.description.clone(),
        goal: params.goal.clone(),
    };

    match write_side
        .insert_segment(collection_id, segment.clone())
        .await
    {
        Ok(_) => Ok((
            StatusCode::OK,
            Json(json!({ "success": true, "id": segment.id, "segment": segment })),
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
    path = "/v1/collections/{id}/segments/delete",
    description = "Deletes an existing segment"
)]
async fn delete_segment_v1(
    Path(id): Path<String>,
    TypedHeader(_auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<DeleteSegmentParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    match write_side.delete_segment(collection_id, params.id).await {
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
    path = "/v1/collections/{id}/segments/update",
    description = "Updates an existing segment"
)]
async fn update_segment_v1(
    Path(id): Path<String>,
    TypedHeader(_auth): AuthorizationBearerHeader,
    write_side: State<Arc<WriteSide>>,
    Json(params): Json<UpdateSegmentParams>,
) -> impl IntoResponse {
    let collection_id = CollectionId(id);

    let segment = Segment {
        id: params.id.clone(),
        name: params.name.clone(),
        description: params.description.clone(),
        goal: params.goal.clone(),
    };

    match write_side.update_segment(collection_id, segment).await {
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
