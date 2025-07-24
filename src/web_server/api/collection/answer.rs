use crate::ai::answer::{Answer, AnswerError, AnswerEvent};
use crate::collection_manager::sides::read::ReadSide;
use crate::types::CollectionId;
use crate::types::{ApiKey, Interaction};
use anyhow::Context;
use axum::extract::Query;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use axum::{extract::State, Json, Router};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error};
use axum::http::StatusCode;

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/answer", post(answer_v1))
        .route(
            "/v1/collections/{collection_id}/planned_answer",
            post(planned_answer_v1),
        )
        .route(
            "/v1/collections/{collection_id}/suggestions",
            post(answer_suggestions_v1),
        )
        .route("/v1/collections/{collection_id}/logs", get(answer_logs_v1))
        .with_state(read_side)
}

#[derive(Deserialize)]
struct AnswerQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn planned_answer_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();
    let answer = Answer::try_new(read_side.clone(), collection_id, query_params.api_key).await?;

    tokio::spawn(async {
        let r = answer.planned_answer(interaction, answer_sender).await;
        if let Err(e) = r {
            error!(error = ?e, "Failed to run planned answer");
        }
    });

    let (http_sender, http_receiver) = mpsc::channel(10);
    tokio::spawn(async move {
        while let Some(event) = answer_receiver.recv().await {
            let ev = match Event::default().json_data(event) {
                Ok(e) => e,
                Err(err) => {
                    error!(error = ?err, "Cannot build event. Stop loop.");
                    break;
                }
            };

            if let Err(err) = http_sender.send(Ok(ev)).await {
                error!(error = ?err, "Cannot stream to FE");
            }
        }
    });

    let rx_stream = ReceiverStream::new(http_receiver);
    Ok(Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    ))
}

async fn answer_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();

    let answer = Answer::try_new(read_side.clone(), collection_id, query.api_key).await?;

    let logs = read_side.get_hook_logs();
    let log_sender = logs.get_sender(&collection_id);
    tokio::spawn(async move {
        let r = answer.answer(interaction, answer_sender, log_sender).await;
        if let Err(e) = r {
            error!(error = ?e, "Failed to run answer");
        }
    });

    let (http_sender, http_receiver) = mpsc::channel(10);
    tokio::spawn(async move {
        while let Some(event) = answer_receiver.recv().await {
            let ev = match Event::default().json_data(event) {
                Ok(e) => e,
                Err(err) => {
                    error!(error = ?err, "Cannot build event. Stop loop.");
                    break;
                }
            };

            if let Err(err) = http_sender.send(Ok(ev)).await {
                error!(error = ?err, "Cannot stream to FE");
            }
        }
    });

    let rx_stream = ReceiverStream::new(http_receiver);
    Ok(Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    ))
}

async fn answer_suggestions_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> impl IntoResponse {
    let answer = match Answer::try_new(read_side.clone(), collection_id, query.api_key)
        .await {
            Ok(answer) => answer,
            Err(e) => {
                error!(error = ?e, "Failed to create Answer instance");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": "Failed to initialize answer service"
                    }))
                );
            }
        };

    let logs = read_side.get_logs();
    let log_sender = logs.get_sender(&collection_id);

    let response = match answer.suggestions(interaction, log_sender).await {
        Ok(response) => response,
        Err(e) => {
            error!(error = ?e, "Failed to generate suggestions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "Failed to generate suggestions"
                }))
            );
        }
    };

    (StatusCode::OK, Json(json!({
        "suggestions": response
    })))
}

async fn answer_logs_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query): Query<AnswerQueryParams>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    read_side
        .check_read_api_key(collection_id, query.api_key)
        .await?;

    let logs = read_side.get_hook_logs();
    let mut answer_receiver = logs.get_or_create_receiver(collection_id);
    let (http_sender, http_receiver) = mpsc::channel(10);

    http_sender
        .send(Ok(Event::default().data("Connected")))
        .await
        .context("Cannot send data")?;

    tokio::spawn(async move {
        while let Ok(event) = answer_receiver.recv().await {
            let ev = Event::default().data(event.1);
            if let Err(err) = http_sender.send(Ok(ev)).await {
                // FE is disconnected. Don't care about it.
                debug!(error = ?err, "Cannot stream to FE");
                break; // Clean up the pending tokio task
            }
        }
    });

    let rx_stream = ReceiverStream::new(http_receiver);
    Ok(Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("# keepalive"),
    ))
}

impl Serialize for AnswerEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut s = serializer.serialize_struct("", 2)?;
        s.serialize_field("type", "response")?;
        match self {
            AnswerEvent::Acknowledged => {
                s.serialize_field("message", "Acknowledged")?;
            }
            AnswerEvent::SelectedLLM(config) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "SELECTED_LLM",
                        "result": serde_json::to_string(config).unwrap(),
                        "done": true,
                    }),
                )?;
            }
            AnswerEvent::ResultAction { action, result } => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": action,
                        "result": result,
                    }),
                )?;
            }
            AnswerEvent::FailedToRunRelatedQuestion(err) => {
                s.serialize_field(
                    "message",
                    &format!("Failed to run related questions stream, {err:?}"),
                )?;
            }
            AnswerEvent::RelatedQueries(chunk) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "RELATED_QUERIES",
                        "result": chunk,
                        "done": true,
                    }),
                )?;
            }
            AnswerEvent::FailedToFetchRelatedQuestion(err) => {
                s.serialize_field("message", &format!("Error during streaming, {err:?}"))?;
            }
            AnswerEvent::OptimizeingQuery(query) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "OPTIMIZING_QUERY",
                        "result": query,
                        "done": true,
                    }),
                )?;
            }
            AnswerEvent::SearchResults(results) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "SEARCH_RESULTS",
                        "result": serde_json::to_string(results).unwrap(),
                        "done": true,
                    }),
                )?;
            }
            AnswerEvent::FailedToRunPrompt(err) => {
                s.serialize_field("message", &format!("Failed to run prompt: {err:?}"))?;
            }
            AnswerEvent::AnswerResponse(chunk) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "ANSWER_RESPONSE",
                        "result": chunk,
                        "done": false,
                    }),
                )?;
            }
            AnswerEvent::FailedToFetchAnswer(err) => {
                s.serialize_field("message", &format!("Failed to fetch answer: {err:?}"))?;
            }
        }

        s.end()
    }
}
