use crate::ai::training_sets::TrainingSet;
use crate::collection_manager::sides::read::ReadSide;
use crate::collection_manager::sides::write::WriteSide;
use crate::types::{
    CollectionId, InsertTrainingSetParams, InteractionLLMConfig, ReadApiKey, TrainingSetId,
    WriteApiKey,
};
use crate::web_server::api::util::print_error;
use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use http::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Deserialize, Debug, Clone)]
struct TrainingSetsQueryBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_config: Option<InteractionLLMConfig>,
}

pub fn read_apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/training_sets/{training_set}/generate",
            post(generate_training_sets_v1),
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
        .route(
            "/v1/collections/{collection_id}/training_sets/{training_set}/delete",
            post(delete_training_sets_v1),
        )
        .with_state(write_side)
}

async fn generate_training_sets_v1(
    Path((collection_id, training_set_destination)): Path<(CollectionId, TrainingSetId)>,
    read_side: State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
    Json(body): Json<TrainingSetsQueryBody>,
) -> impl IntoResponse {
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
        Ok(destination) => match training_set.generate_training_data_for(destination).await {
            Ok(results) => Ok((StatusCode::OK, Json(json!({ "queries": results })))),
            Err(e) => {
                print_error(&e, "Error generating training data");
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "Failed to generate training data" })),
                ))
            }
        },
        Err(err) => {
            print_error(&err, "Error generating training sets");
            Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Invalid training set ID" })),
            ))
        }
    }
}

async fn get_training_sets_v1(
    Path((collection_id, training_set)): Path<(CollectionId, TrainingSetId)>,
    State(read_side): State<Arc<ReadSide>>,
    read_api_key: ReadApiKey,
) -> impl IntoResponse {
    match training_set.try_into_destination() {
        Ok(destination) => {
            match read_side
                .get_training_data_for(collection_id, read_api_key, destination)
                .await
            {
                Some(Ok(training_data)) => Ok((
                    StatusCode::OK,
                    Json(json!({ "training_sets": training_data })),
                )),
                Some(Err(_error)) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "Error retrieving existing training sets" })),
                )),
                None => Ok((StatusCode::OK, Json(json!({ "training_sets": null })))),
            }
        }
        Err(_error) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid training set ID" })),
        )),
    }
}

async fn insert_training_sets_v1(
    Path((collection_id, training_set)): Path<(CollectionId, TrainingSetId)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<InsertTrainingSetParams>,
) -> impl IntoResponse {
    match training_set.try_into_destination() {
        Ok(destination) => {
            let training_set_interface = match write_side
                .get_training_set_interface(write_api_key, collection_id)
                .await
            {
                Ok(interface) => interface,
                Err(e) => {
                    print_error(&e.into(), "Error getting training set interface");
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": "Failed to access training set interface" })),
                    ));
                }
            };
            match serde_json::to_string(&params.training_set) {
                Ok(training_data_as_json) => {
                    match training_set_interface
                        .insert_training_set(collection_id, destination, training_data_as_json)
                        .await
                    {
                        Ok(_) => Ok((StatusCode::CREATED, Json(json!({ "inserted": true })))),
                        Err(e) => {
                            print_error(&e, "Error inserting training set");
                            Err((
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": "Failed to insert training set" })),
                            ))
                        }
                    }
                }
                Err(error) => {
                    print_error(&error.into(), "Failed to serialize training set data");
                    Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": "Failed to serialize training set data" })),
                    ))
                }
            }
        }
        Err(_error) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid training set ID" })),
        )),
    }
}

async fn delete_training_sets_v1(
    Path((collection_id, training_set)): Path<(CollectionId, TrainingSetId)>,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> impl IntoResponse {
    match training_set.try_into_destination() {
        Ok(destination) => {
            let training_set_interface = match write_side
                .get_training_set_interface(write_api_key, collection_id)
                .await
            {
                Ok(interface) => interface,
                Err(e) => {
                    print_error(&e.into(), "Error getting training set interface for delete");
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": "Failed to access training set interface" })),
                    ));
                }
            };

            match training_set_interface
                .delete_training_set(collection_id, destination)
                .await
            {
                Ok(_) => Ok((StatusCode::NO_CONTENT, Json(json!({ "deleted": true })))),
                Err(e) => {
                    print_error(&e, "Error deleting training set");
                    Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": "Failed to delete training set" })),
                    ))
                }
            }
        }
        Err(_error) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid training set ID" })),
        )),
    }
}
