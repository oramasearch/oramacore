use crate::ai::training_sets::TrainingSet;
use crate::collection_manager::sides::read::ReadSide;
use crate::collection_manager::sides::write::WriteSide;
use crate::types::{
    ApiKey, CollectionId, InsertTrainingSetParams, InteractionLLMConfig, TrainingSetId,
    TrainingSetsQueryOptimizerParams, WriteApiKey,
};
use crate::web_server::api::util::print_error;
use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{utoipa::IntoParams, *};
use axum_openapi3::{utoipa::ToSchema, *};
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Deserialize, Debug, Clone)]
struct TrainingSetsQueryBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_config: Option<InteractionLLMConfig>,
}

#[derive(Deserialize, IntoParams, ToSchema)]
struct ApiKeyOnlyQuery {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(generate_training_sets_v1)
        .route(get_training_sets_v1)
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(insert_training_sets_v1)
        .add(delete_training_sets_v1)
        .with_state(write_side)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/training_sets/{training_set}/generate",
    description = "Generate training set data"
)]
async fn generate_training_sets_v1(
    collection_id: CollectionId,
    training_set_destination: TrainingSetId,
    Query(query_params): Query<TrainingSetsQueryOptimizerParams>,
    read_side: State<Arc<ReadSide>>,
    Json(body): Json<TrainingSetsQueryBody>,
) -> IntoResponse {
    let read_api_key = query_params.api_key;
    let llm_config = body.llm_config;
    let llm_service = read_side.get_llm_service();

    let training_set = TrainingSet::new(
        collection_id,
        read_side.0.clone(),
        read_api_key,
        llm_service,
        llm_config,
    );

    match training_set_destination.try_into_destination() {
        Ok(destination) => {
            let results = training_set.generate_training_data_for(destination).await?;

            Ok((StatusCode::OK, Json(json!({ "queries": results }))))
        }
        Err(err) => {
            print_error(&err, "Error generating training sets");
            Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Invalid training set ID" })),
            ))
        }
    }
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/training_sets/{training_set}/get",
    description = "Get existing training set data"
)]
async fn get_training_sets_v1(
    collection_id: CollectionId,
    training_set: TrainingSetId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<ApiKeyOnlyQuery>,
) -> IntoResponse {
    match training_set.try_into_destination() {
        Ok(destination) => {
            let read_api_key = query_params.api_key;

            match read_side
                .get_training_data_for(collection_id, read_api_key, destination)
                .await
            {
                Some(Ok(training_data)) => Ok((
                    StatusCode::OK,
                    Json(json!({ "training_sets": training_data })),
                )),
                Some(Err(error)) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "Error retrieving existing training sets" })),
                )),
                None => Ok((StatusCode::OK, Json(json!({ "training_sets": null })))),
            }
        }
        Err(error) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid training set ID" })),
        )),
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/training_sets/{training_set}/insert",
    description = "Insert a new training set"
)]
async fn insert_training_sets_v1(
    collection_id: CollectionId,
    training_set: TrainingSetId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<InsertTrainingSetParams>,
) -> IntoResponse {
    match training_set.try_into_destination() {
        Ok(destination) => {
            let training_set_interface = write_side
                .get_training_set_interface(write_api_key, collection_id)
                .await?;
            match serde_json::to_string(&params.training_set) {
                Ok(training_data_as_json) => {
                    let _ = training_set_interface
                        .insert_training_set(collection_id, destination, training_data_as_json)
                        .await?;
                    Ok((StatusCode::CREATED, json!({ "inserted": true })))
                }
                Err(error) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({ "error": "Failed to serialize training set data" }),
                )),
            }
        }
        Err(error) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid training set ID" })),
        )),
    }
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/training_sets/{training_set}/delete",
    description = "Delete an existing training set"
)]
async fn delete_training_sets_v1(
    collection_id: CollectionId,
    training_set: TrainingSetId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> IntoResponse {
    match training_set.try_into_destination() {
        Ok(destination) => {
            let training_set_interface = write_side
                .get_training_set_interface(write_api_key, collection_id)
                .await?;
            let _ = training_set_interface
                .delete_training_set(collection_id, destination)
                .await?;

            Ok((StatusCode::NO_CONTENT, json!({ "deleted": true })))
        }
        Err(error) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid training set ID" })),
        )),
    }
}
