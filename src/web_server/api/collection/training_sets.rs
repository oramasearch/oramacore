use crate::ai::training_sets::{TrainingDestination, TrainingSet};
use crate::collection_manager::sides::read::{ReadError, ReadSide};

use crate::collection_manager::sides::write::{self, WriteError, WriteSide};
use crate::types::{
    ApiKey, CollectionId, InsertTrainingSetParams, InteractionLLMConfig,
    TrainingSetsQueryOptimizerParams, WriteApiKey,
};
use axum::extract::Query;
use axum::routing::{get, post};
use axum::{extract::State, Json, Router};
use axum_openapi3::utoipa::IntoParams;
use axum_openapi3::{utoipa::ToSchema, *};
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

#[derive(Deserialize, IntoParams, ToSchema)]
struct WriteApiKeyOnlyQuery {
    #[serde(rename = "api-key")]
    api_key: WriteApiKey,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/training_sets/{training_set}/generate",
            post(generate_training_sets_v1),
        )
        .route(
            "/v1/collections/{collection_id}/training_sets/{training_set}/list",
            get(list_training_sets_v1),
        )
        .route(
            "/v1/collections/{collection_id}/training_sets/{training_set}/get",
            get(get_training_sets_v1),
        )
        .with_state(read_side)
}

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/training_sets/{training_set}/insert",
            post(insert_training_sets_v1),
        )
        .with_state(write_side)
}

async fn generate_training_sets_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<TrainingSetsQueryOptimizerParams>,
    Json(body): Json<TrainingSetsQueryBody>,
) -> Result<Json<serde_json::Value>, ReadError> {
    let read_api_key = query_params.api_key;
    let llm_config = body.llm_config;
    let llm_service = read_side.get_llm_service();

    let training_set = TrainingSet::new(
        collection_id,
        read_side,
        read_api_key,
        llm_service,
        llm_config,
    );

    let results = training_set
        .generate_training_data_for(TrainingDestination::QueryOptimizer)
        .await?;

    Ok(Json(json!({ "queries": results })))
}

async fn list_training_sets_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<ApiKeyOnlyQuery>,
    training_set: String,
) -> Result<Json<serde_json::Value>, ReadError> {
    match TrainingDestination::try_from_str(&training_set) {
        Some(destination) => {
            let read_api_key = query_params.api_key;

            let training_data = read_side
                .list_training_data_for(collection_id, read_api_key, destination)
                .await?;

            Ok(Json(json!({ "training_data": training_data })))
        }
        None => Err(ReadError::Generic(anyhow::Error::msg(
            "Invalid training set",
        ))),
    }
}

async fn get_training_sets_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<ApiKeyOnlyQuery>,
    training_set: String,
) -> Result<Json<serde_json::Value>, ReadError> {
    match TrainingDestination::try_from_str(&training_set) {
        Some(destination) => {
            let read_api_key = query_params.api_key;

            match read_side
                .get_training_data_for(collection_id, read_api_key, destination)
                .await
            {
                Some(Ok(training_data)) => Ok(Json(json!({ "training_data": training_data }))),
                Some(Err(error)) => Err(error),
                None => Err(ReadError::NotFound(collection_id)),
            }
        }
        None => Err(ReadError::Generic(anyhow::Error::msg(
            "Invalid training set",
        ))),
    }
}

async fn insert_training_sets_v1(
    collection_id: CollectionId,
    State(write_side): State<Arc<WriteSide>>,
    Query(query_params): Query<WriteApiKeyOnlyQuery>,
    Json(params): Json<InsertTrainingSetParams>,
    training_set: String,
) -> Result<Json<serde_json::Value>, WriteError> {
    match TrainingDestination::try_from_str(&training_set) {
        Some(destination) => {
            let write_api_key = query_params.api_key;
            let training_set_interface = write_side
                .get_training_set_interface(write_api_key, collection_id)
                .await?;
            match serde_json::to_string(&params.training_set) {
                Ok(training_data_as_json) => {
                    let _ = training_set_interface
                        .insert_training_set(collection_id, destination, training_data_as_json)
                        .await?;
                    Ok(Json(json!({ "inserted": true })))
                }
                Err(error) => Err(WriteError::Generic(anyhow::Error::new(error))),
            }
        }
        None => Err(WriteError::Generic(anyhow::Error::msg(
            "Invalid training set",
        ))),
    }
}

async fn delete_training_sets_v1(
    collection_id: CollectionId,
    State(write_side): State<Arc<WriteSide>>,
    Query(query_params): Query<WriteApiKeyOnlyQuery>,
    training_set: String,
) -> Result<Json<serde_json::Value>, WriteError> {
    match TrainingDestination::try_from_str(&training_set) {
        Some(destination) => {
            let write_api_key = query_params.api_key;
            let training_set_interface = write_side
                .get_training_set_interface(write_api_key, collection_id)
                .await?;
            let _ = training_set_interface
                .delete_training_set(collection_id, destination)
                .await?;
            Ok(Json(json!({ "deleted": true })))
        }
        None => Err(WriteError::Generic(anyhow::Error::msg(
            "Invalid training set",
        ))),
    }
}
