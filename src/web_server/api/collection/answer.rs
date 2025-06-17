use crate::ai::answer::{Answer, AnswerError, PlannedAnswerEvent};
use crate::collection_manager::sides::read::ReadSide;
use crate::types::CollectionId;
use crate::types::{ApiKey, Interaction, InteractionLLMConfig, InteractionMessage};
use anyhow::Context;
use axum::extract::Query;
use axum::response::sse::Event;
use axum::response::Sse;
use axum::routing::post;
use axum::{extract::State, Json, Router};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

#[derive(Serialize, Deserialize, Debug)]
struct MessageChunk {
    text: String,
    is_final: bool,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
enum SseMessage {
    #[serde(rename = "acknowledgement")]
    Acknowledge { message: String },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "response")]
    Response { message: String },
}

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/answer", post(answer_v1))
        .route(
            "/v1/collections/{collection_id}/planned_answer",
            post(planned_answer_v1),
        )
        .with_state(read_side)
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
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();
    let answer = Answer::try_new(read_side.0.clone(), collection_id, query.api_key).await?;

    tokio::spawn(async {
        let r = answer.answer(interaction, answer_sender).await;
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

enum AudienceManagementResult {
    Segment(Option<crate::collection_manager::sides::segments::Segment>),
    Trigger(Option<crate::collection_manager::sides::triggers::Trigger>),
}

async fn select_triggers_and_segments(
    read_side: State<Arc<ReadSide>>,
    read_api_key: ApiKey,
    collection_id: CollectionId,
    conversation: Option<Vec<InteractionMessage>>,
    mut llm_config: Option<InteractionLLMConfig>,
) -> impl Stream<Item = AudienceManagementResult> {
    let segment_interface = read_side
        .get_segments_manager(read_api_key, collection_id)
        .await
        .expect("Failed to get segments for the collection");
    let all_segments = segment_interface
        .list()
        .await
        .expect("Failed to get segments for the collection");

    if read_side.is_gpu_overloaded() {
        match read_side.select_random_remote_llm_service() {
            Some((provider, model)) => {
                info!("GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.", provider);
                llm_config = Some(InteractionLLMConfig { model, provider });
            }
            None => {
                warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
            }
        }
    }

    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        if all_segments.is_empty() {
            tx.send(AudienceManagementResult::Segment(None))
                .await
                .unwrap();
            return;
        };

        let chosen_segment = segment_interface
            .perform_segment_selection(conversation.clone(), llm_config.clone())
            .await
            .expect("Failed to choose a segment.");

        let trigger_interface = read_side
            .get_triggers_manager(read_api_key, collection_id)
            .await
            .expect("Failed to get triggers manager");

        match chosen_segment {
            None => {
                tx.send(AudienceManagementResult::Segment(None))
                    .await
                    .unwrap();

                tx.send(AudienceManagementResult::Trigger(None))
                    .await
                    .unwrap();
            }
            Some(segment) => {
                let full_segment = segment_interface
                    .get(segment.clone().id.clone())
                    .await
                    .expect("Failed to get full segment");

                tx.send(AudienceManagementResult::Segment(full_segment.clone()))
                    .await
                    .unwrap();

                let all_segments_triggers = trigger_interface
                    .get_all_triggers_by_segment(full_segment.unwrap().id.clone())
                    .await
                    .expect("Failed to get triggers for the segment");

                if all_segments_triggers.is_empty() {
                    tx.send(AudienceManagementResult::Trigger(None))
                        .await
                        .unwrap();
                    return;
                }

                let chosen_trigger = trigger_interface
                    .perform_trigger_selection(
                        conversation,
                        all_segments_triggers,
                        llm_config.clone(),
                    )
                    .await
                    .context(
                        "Failed to choose a trigger for the given segment. Will fall back to None.",
                    )
                    .unwrap_or(None);

                match chosen_trigger {
                    None => {
                        tx.send(AudienceManagementResult::Trigger(None))
                            .await
                            .unwrap();
                    }
                    Some(chosen_trigger) => {
                        let full_trigger = trigger_interface
                            .get_trigger(chosen_trigger.id.clone())
                            .await
                            .expect("Failed to get full trigger");

                        tx.send(AudienceManagementResult::Trigger(full_trigger))
                            .await
                            .unwrap();
                    }
                }
            }
        }
    });

    ReceiverStream::new(rx)
}

fn serialize_response(action: &str, result: &str, done: bool) -> serde_json::Result<String> {
    serde_json::to_string(&SseMessage::Response {
        message: json!({
            "action": action,
            "result": result,
            "done": done,
        })
        .to_string(),
    })
}

impl Serialize for PlannedAnswerEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut s = serializer.serialize_struct("", 1)?;
        match self {
            PlannedAnswerEvent::Acknowledged => {
                s.serialize_field("message", "Acknowledged")?;
            }
            PlannedAnswerEvent::SelectedLLM(config) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "SELECTED_LLM",
                        "result": serde_json::to_string(config).unwrap(),
                        "done": true,
                    }),
                )?;
            }
            PlannedAnswerEvent::GetSegment(segment) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "GET_SEGMENT",
                        "result": serde_json::to_string(segment).unwrap(),
                        "done": true,
                    }),
                )?;
            }
            PlannedAnswerEvent::GetTrigger(trigger) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "GET_TRIGGER",
                        "result": serde_json::to_string(trigger).unwrap(),
                        "done": true,
                    }),
                )?;
            }
            PlannedAnswerEvent::ResultAction { action, result } => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": action,
                        "result": result,
                    }),
                )?;
            }
            PlannedAnswerEvent::FailedToRunRelatedQuestion(err) => {
                s.serialize_field(
                    "message",
                    &format!("Failed to run related questions stream, {:?}", err),
                )?;
            }
            PlannedAnswerEvent::RelatedQueries(chunk) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "RELATED_QUERIES",
                        "result": chunk,
                        "done": true,
                    }),
                )?;
            }
            PlannedAnswerEvent::FailedToFetchRelatedQuestion(err) => {
                s.serialize_field("message", &format!("Error during streaming, {:?}", err))?;
            }
            PlannedAnswerEvent::OptimizeingQuery(query) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "OPTIMIZING_QUERY",
                        "result": query,
                        "done": true,
                    }),
                )?;
            }
            PlannedAnswerEvent::SearchResults(results) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "SEARCH_RESULTS",
                        "result": serde_json::to_string(results).unwrap(),
                        "done": true,
                    }),
                )?;
            }
            PlannedAnswerEvent::FailedToRunPrompt(err) => {
                s.serialize_field("message", &format!("Failed to run prompt: {:?}", err))?;
            }
            PlannedAnswerEvent::AnswerResponse(chunk) => {
                s.serialize_field(
                    "message",
                    &json!({
                        "action": "ANSWER_RESPONSE",
                        "result": chunk,
                        "done": false,
                    }),
                )?;
            }
            PlannedAnswerEvent::FailedToFetchAnswer(err) => {
                s.serialize_field("message", &format!("Failed to fetch answer: {:?}", err))?;
            }
        }

        s.end()
    }
}
