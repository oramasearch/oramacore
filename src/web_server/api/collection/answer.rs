use crate::ai::{LlmType, Segment, SegmentResponse, Trigger};
use crate::collection_manager::dto::{
    ApiKey, HybridMode, Interaction, InteractionMessage, Limit, SearchMode, SearchParams,
    SearchResult,
};
use crate::collection_manager::sides::ReadSide;
use crate::types::CollectionId;
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
    #[serde(rename = "optimizing-query")]
    OptimizingQuery { message: String },
    #[serde(rename = "optimized-query")]
    OptimizedQuery { message: String },
    #[serde(rename = "sources")]
    Sources { message: SearchResult },
    #[serde(rename = "answer_chunk")]
    AnswerChunk { message: MessageChunk },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "response")]
    Response { message: String },
}

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{id}/answer", post(answer_v0))
        .route(
            "/v1/collections/{id}/planned_answer",
            post(planned_answer_v0),
        )
        .with_state(read_side)
}

#[derive(Deserialize)]
struct AnswerQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn planned_answer_v0(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query_params): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let collection_id = CollectionId(id).0;
    let read_side = read_side.clone();

    let query = interaction.query;
    let conversation = interaction.messages;
    let api_key = query_params.api_key;

    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        let ai_service = read_side.get_ai_service();

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::Acknowledge {
                    message: "Acknowledged".to_string(),
                })
                .unwrap(),
            )))
            .await;

        let mut stream = ai_service
            .planned_answer_stream(query, collection_id, Some(conversation), api_key)
            .await
            .unwrap();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(response) => {
                    if tx
                        .send(Ok(Event::default().data(
                            serde_json::to_string(&SseMessage::Response {
                                message: response.data,
                            })
                            .unwrap(),
                        )))
                        .await
                        .is_err()
                    {
                        break; // Client disconnected
                    }
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
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}

// @todo: this function needs some cleaning. It works but it's not well structured.
async fn answer_v0(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let collection_id = CollectionId(id);
    let read_side = read_side.clone();

    // This api key should be used to access the read side
    // We should check it before doing anything
    // For now, we just use it to access the read side
    // TODO: implement the check as soon as possible
    let read_api_key = query.api_key;

    let query = interaction.query;
    let conversation = interaction.messages;

    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        let ai_service = read_side.get_ai_service();

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::Acknowledge {
                    message: "Acknowledged".to_string(),
                })
                .unwrap(),
            )))
            .await;

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::OptimizingQuery {
                    message: "Optimizing query".to_string(),
                })
                .unwrap(),
            )))
            .await;

        let optimized_query = ai_service
            .chat(LlmType::GoogleQueryTranslator, query.clone(), None, None)
            .await
            .unwrap();

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::OptimizedQuery {
                    message: optimized_query.text.to_string(),
                })
                .unwrap(),
            )))
            .await;

        let search_results = read_side
            .search(
                read_api_key,
                collection_id,
                SearchParams {
                    mode: SearchMode::Hybrid(HybridMode {
                        term: optimized_query.text,
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
                serde_json::to_string(&SseMessage::Sources {
                    message: search_results.clone(),
                })
                .unwrap(),
            )))
            .await;

        let search_result_str = serde_json::to_string(&search_results.hits).unwrap();

        let stream = ai_service
            .chat_stream(
                LlmType::Answer,
                query,
                Some(conversation),
                Some(search_result_str),
            )
            .await
            .unwrap();

        let mut stream = stream;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(response) => {
                    let chunk = MessageChunk {
                        text: response.text_chunk,
                        is_final: response.is_final,
                    };

                    if tx
                        .send(Ok(Event::default().data(
                            serde_json::to_string(&SseMessage::AnswerChunk { message: chunk })
                                .unwrap(),
                        )))
                        .await
                        .is_err()
                    {
                        break; // Client disconnected
                    }

                    if response.is_final {
                        break;
                    }
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
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}

enum AudienceManagementResult {
    Segment(Option<crate::collection_manager::sides::segments::Segment>),
    ChosenSegment(Option<SegmentResponse>),
    Trigger(Option<crate::collection_manager::sides::triggers::Trigger>),
}

async fn select_triggers_and_segments(
    read_side: Arc<ReadSide>,
    read_api_key: ApiKey,
    collection_id: CollectionId,
    conversation: Option<Vec<InteractionMessage>>,
) -> impl Stream<Item = AudienceManagementResult> {
    let ai_service = read_side.get_ai_service();

    let all_segments = read_side
        .get_all_segments_by_collection(read_api_key.clone(), collection_id.clone())
        .await
        .expect("Failed to get segments for the collection");

    let all_triggers = read_side
        .get_all_triggers_by_collection(read_api_key.clone(), collection_id.clone())
        .await
        .expect("Failed to get all triggers for the collection");

    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        if all_segments.is_empty() {
            tx.send(AudienceManagementResult::Segment(None))
                .await
                .unwrap();
            return;
        };

        let chosen_segment = ai_service
            .get_segment(all_segments, conversation.clone())
            .await
            .expect("Failed to get segment");

        let full_segment = read_side
            .get_segment(
                read_api_key.clone(),
                collection_id.clone(),
                chosen_segment.id.clone(),
            )
            .await
            .expect("Failed to get full segment");

        tx.send(AudienceManagementResult::Segment(full_segment))
            .await
            .unwrap();

        let all_segments_triggers = read_side
            .get_all_triggers_by_segment(
                read_api_key.clone(),
                collection_id.clone(),
                chosen_segment.id.clone(),
            )
            .await
            .expect("Failed to get triggers for the segment");

        let chosen_trigger = ai_service
            .get_trigger(all_segments_triggers, conversation)
            .await
            .expect("Failed to get trigger");

        let full_trigger = read_side
            .get_trigger(read_api_key, collection_id, chosen_trigger.id.clone())
            .await
            .expect("Failed to get full trigger");

        tx.send(AudienceManagementResult::Trigger(full_trigger))
            .await
            .unwrap();
    });

    ReceiverStream::new(rx)
}
