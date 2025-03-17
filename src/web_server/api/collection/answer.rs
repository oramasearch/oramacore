use crate::ai::party_planner::PartyPlanner;
use crate::ai::vllm;
use crate::collection_manager::dto::{
    ApiKey, AutoMode, Interaction, InteractionMessage, Limit, Role, SearchMode, SearchParams,
};
use crate::collection_manager::sides::segments::Segment;
use crate::collection_manager::sides::triggers::Trigger;
use crate::collection_manager::sides::ReadSide;
use crate::types::CollectionId;
use anyhow::Context;
use axum::extract::Query;
use axum::response::sse::Event;
use axum::response::Sse;
use axum::routing::post;
use axum::{
    extract::{Path, State},
    Json, Router,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

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
        .route("/v1/collections/{id}/answer", post(answer_v1))
        .route(
            "/v1/collections/{id}/planned_answer",
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
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query_params): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let vllm_service = read_side.get_vllm_service();
    let collection_id = CollectionId(id.clone()).0;
    let read_side = read_side.clone();

    let query = interaction.query.clone();
    let conversation = interaction.messages.clone();
    let api_key = query_params.api_key;

    read_side
        .clone()
        .check_read_api_key(CollectionId(id.clone()), api_key.clone())
        .await
        .expect("Invalid API key");

    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::Acknowledge {
                    message: "Acknowledged".to_string(),
                })
                .unwrap(),
            )))
            .await;

        let mut trigger: Option<Trigger> = None;
        let mut segment: Option<Segment> = None;

        // Always make sure that the conversation is not empty, or else the AI will not be able to
        // determine the segment and trigger.
        let segments_and_triggers_conversation = match conversation.len() {
            0 => Some(vec![InteractionMessage {
                role: Role::User,
                content: query.clone(),
            }]),
            _ => Some(conversation.clone()),
        };

        let mut segments_and_triggers_stream = select_triggers_and_segments(
            read_side.clone(),
            api_key.clone(),
            CollectionId(id),
            segments_and_triggers_conversation,
        )
        .await;

        while let Some(result) = segments_and_triggers_stream.next().await {
            match result {
                AudienceManagementResult::Segment(s) => {
                    segment = s.clone();

                    let _ = tx
                        .send(Ok(Event::default().data(
                            serialize_response(
                                "GET_SEGMENT",
                                &serde_json::to_string(&s).unwrap(),
                                true,
                            )
                            .unwrap(),
                        )))
                        .await;
                }
                AudienceManagementResult::Trigger(t) => {
                    trigger = t.clone();

                    let _ = tx
                        .send(Ok(Event::default().data(
                            serialize_response(
                                "GET_TRIGGER",
                                &serde_json::to_string(&t).unwrap(),
                                true,
                            )
                            .unwrap(),
                        )))
                        .await;
                }
            }
        }

        let mut party_planner_stream = PartyPlanner::run(
            read_side.clone(),
            collection_id.clone(),
            api_key.clone(),
            interaction.query.clone(),
            interaction.messages.clone(),
            segment.clone(),
            trigger.clone(),
        );

        while let Some(message) = party_planner_stream.next().await {
            let _ = tx
                .send(Ok(Event::default().data(
                    serde_json::to_string(&SseMessage::Response {
                        message: json!({
                            "action": message.action,
                            "result": message.result,
                        })
                        .to_string(),
                    })
                    .unwrap(),
                )))
                .await;
        }
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}

// @todo: this function needs some cleaning. It works but it's not well structured.
async fn answer_v1(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let collection_id = CollectionId(id);
    let read_side = read_side.clone();
    let read_api_key = query.api_key;

    read_side
        .clone()
        .check_read_api_key(collection_id.clone(), read_api_key.clone())
        .await
        .expect("Invalid API key");

    let query = interaction.query;
    let conversation = interaction.messages;

    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        let ai_service = read_side.clone().get_ai_service();
        let vllm_service = read_side.clone().get_vllm_service();

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::Acknowledge {
                    message: "Acknowledged".to_string(),
                })
                .unwrap(),
            )))
            .await;

        let mut trigger: Option<Trigger> = None;
        let mut segment: Option<Segment> = None;

        // Always make sure that the conversation is not empty, or else the AI will not be able to
        // determine the segment and trigger.
        let segments_and_triggers_conversation = match conversation.len() {
            0 => Some(vec![InteractionMessage {
                role: Role::User,
                content: query.clone(),
            }]),
            _ => Some(conversation.clone()),
        };

        let mut segments_and_triggers_stream = select_triggers_and_segments(
            read_side.clone(),
            read_api_key.clone(),
            collection_id.clone(),
            segments_and_triggers_conversation,
        )
        .await;

        while let Some(result) = segments_and_triggers_stream.next().await {
            match result {
                AudienceManagementResult::Segment(s) => {
                    segment = s;
                }
                AudienceManagementResult::Trigger(t) => {
                    trigger = t;
                }
            }
        }

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response(
                    "GET_SEGMENT",
                    &serde_json::to_string(&segment).unwrap(),
                    true,
                )
                .unwrap(),
            )))
            .await;

        let optimized_query_variables = vec![("input".to_string(), query.clone())];

        let optimized_query = vllm_service
            .run_known_prompt(vllm::KnownPrompts::OptimizeQuery, optimized_query_variables)
            .await
            .unwrap_or(query.clone()); // fallback to the original query if the optimization fails

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response("OPTIMIZING_QUERY", &optimized_query, true).unwrap(),
            )))
            .await;

        // @todo: derive limit, boost, and where filters depending on the schema and the input query
        let search_results = read_side
            .search(
                read_api_key,
                collection_id,
                SearchParams {
                    mode: SearchMode::Auto(AutoMode {
                        term: optimized_query,
                    }),
                    limit: Limit(5),
                    where_filter: HashMap::new(),
                    boost: HashMap::new(),
                    facets: HashMap::new(),
                    properties: crate::collection_manager::dto::Properties::Star,
                },
            )
            .await
            .unwrap();

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response(
                    "SEARCH_RESULTS",
                    &serde_json::to_string(&search_results.hits).unwrap(),
                    true,
                )
                .unwrap(),
            )))
            .await;

        let search_result_str = serde_json::to_string(&search_results.hits).unwrap();

        let variables = vec![
            ("question".to_string(), query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];

        let mut answer_stream = vllm_service
            .run_known_prompt_stream(vllm::KnownPrompts::Answer, variables)
            .await;

        while let Some(resp) = answer_stream.next().await {
            match resp {
                Ok(chunk) => {
                    tx.send(Ok(Event::default().data(
                        serialize_response("ANSWER_RESPONSE", &chunk, false).unwrap(),
                    )))
                    .await
                    .unwrap();
                }
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default().data(
                            serde_json::to_string(&SseMessage::Error {
                                message: format!("Error during streaming: {}", e),
                            })
                            .unwrap(),
                        )))
                        .await;
                    break;
                }
            }
        }

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response("ANSWER_RESPONSE", "", true).unwrap(),
            )))
            .await;
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
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
) -> impl Stream<Item = AudienceManagementResult> {
    let all_segments = read_side
        .get_all_segments_by_collection(read_api_key.clone(), collection_id.clone())
        .await
        .expect("Failed to get segments for the collection");

    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        if all_segments.is_empty() {
            tx.send(AudienceManagementResult::Segment(None))
                .await
                .unwrap();
            return;
        };

        let chosen_segment = read_side
            .perform_segment_selection(
                read_api_key.clone(),
                collection_id.clone(),
                conversation.clone(),
            )
            .await
            .expect("Failed to choose a segment.");

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
                let full_segment = read_side
                    .get_segment(
                        read_api_key.clone(),
                        collection_id.clone(),
                        segment.clone().id.clone(),
                    )
                    .await
                    .expect("Failed to get full segment");

                tx.send(AudienceManagementResult::Segment(full_segment.clone()))
                    .await
                    .unwrap();

                let all_segments_triggers = read_side
                    .get_all_triggers_by_segment(
                        read_api_key.clone(),
                        collection_id.clone(),
                        full_segment.unwrap().id.clone(),
                    )
                    .await
                    .expect("Failed to get triggers for the segment");

                if all_segments_triggers.is_empty() {
                    tx.send(AudienceManagementResult::Trigger(None))
                        .await
                        .unwrap();
                    return;
                }

                let chosen_trigger = read_side
                    .perform_trigger_selection(
                        read_api_key.clone(),
                        collection_id.clone(),
                        conversation,
                        all_segments_triggers,
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
                        let full_trigger = read_side
                            .get_trigger(read_api_key, collection_id, chosen_trigger.id.clone())
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
