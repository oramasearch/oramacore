use crate::ai::answer::{Answer, AnswerError, AnswerEvent};
use crate::collection_manager::sides::read::ReadSide;
use crate::types::CollectionId;
use crate::types::{ApiKey, Interaction};
use anyhow::Context;
use axum::extract::{FromRef, Query};
use axum::response::sse::Event;
use axum::response::Sse;
use axum::routing::{get, post};
use axum::{extract::State, Json, Router};
use futures::Stream;
use orama_js_pool::OutputChannel;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tracing::error;

#[derive(Clone)]
struct AnswerReadSide {
    read_side: Arc<ReadSide>,
    channels: Arc<RwLock<HashMap<CollectionId, Arc<broadcast::Sender<(OutputChannel, String)>>>>>,
}

impl FromRef<AnswerReadSide> for Arc<ReadSide> {
    fn from_ref(app_state: &AnswerReadSide) -> Arc<ReadSide> {
        app_state.read_side.clone()
    }
}

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    let answer_read_side = AnswerReadSide {
        read_side,
        channels: Arc::new(RwLock::new(HashMap::new())),
    };

    Router::new()
        .route("/v1/collections/{collection_id}/answer", post(answer_v1))
        .route(
            "/v1/collections/{collection_id}/planned_answer",
            post(planned_answer_v1),
        )
        .route(
            "/v1/collections/{collection_id}/answer-logs",
            get(answer_logs_v1),
        )
        .with_state(answer_read_side)
}

#[derive(Deserialize)]
struct AnswerQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn planned_answer_v1(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query_params): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();
    let answer = Answer::try_new(read_side.0.clone(), collection_id, query_params.api_key).await?;

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
    answer_read_side: State<AnswerReadSide>,
    Query(query): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    let read_side = answer_read_side.read_side.clone();
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();

    let answer = Answer::try_new(read_side, collection_id, query.api_key).await?;

    let channels = answer_read_side.channels.clone();
    tokio::spawn(async move {
        let lock = channels.read().await;
        let log_sender = lock.get(&collection_id);
        let log_sender = log_sender.cloned();
        let r = answer.answer(interaction, answer_sender, log_sender).await;
        if let Err(e) = r {
            error!(error = ?e, "Failed to run answer");
        }
        drop(lock);
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

async fn answer_logs_v1(
    collection_id: CollectionId,
    answer_read_side: State<AnswerReadSide>,
    Query(query): Query<AnswerQueryParams>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    answer_read_side
        .read_side
        .check_read_api_key(collection_id, query.api_key)
        .await?;

    let mut lock = answer_read_side.channels.write().await;
    let mut answer_receiver = match lock.entry(collection_id) {
        Entry::Vacant(a) => {
            let (answer_sender, answer_receiver) = broadcast::channel(100);
            a.insert(Arc::new(answer_sender));
            answer_receiver
        }
        Entry::Occupied(o) => o.get().subscribe(),
    };
    drop(lock);

    let (http_sender, http_receiver) = mpsc::channel(10);

    http_sender
        .send(Ok(Event::default().data("Connected")))
        .await
        .context("Cannot send data")?;

    tokio::spawn(async move {
        while let Ok(event) = answer_receiver.recv().await {
            let ev = Event::default().data(event.1);
            if let Err(err) = http_sender.send(Ok(ev)).await {
                error!(error = ?err, "Cannot stream to FE");
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
            AnswerEvent::GetSegment(segment) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "GET_SEGMENT",
                        "result": serde_json::to_string(segment).unwrap(),
                        "done": true,
                    }),
                )?;
            }
            AnswerEvent::GetTrigger(trigger) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "GET_TRIGGER",
                        "result": serde_json::to_string(trigger).unwrap(),
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
