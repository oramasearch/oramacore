use crate::ai::training_sets::{TrainingDestination, TrainingSet};
use crate::collection_manager::sides::read::{ReadError, ReadSide};
use crate::collection_manager::sides::write::WriteSide;
use crate::types::{ApiKey, CollectionId, InteractionLLMConfig, TrainingSetsQueryOptimizerParams};
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
struct SearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
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
    Router::new().with_state(write_side)
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
