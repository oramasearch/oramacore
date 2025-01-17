use crate::ai::{Conversation, LlmType};
use crate::collection_manager::dto::{
    HybridMode, Interaction, Limit, SearchMode, SearchParams, SearchResult,
};
use crate::collection_manager::sides::read::CollectionsReader;
use crate::types::CollectionId;
use axum::response::sse::Event;
use axum::response::Sse;
use axum::routing::post;
use axum::{
    extract::{Path, State},
    Json, Router,
};
use futures::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_stream::wrappers::ReceiverStream;

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
}

pub fn apis(readers: Arc<CollectionsReader>) -> Router {
    Router::new()
        .route("/v0/collections/{id}/answer", post(answer_v0))
        .with_state(readers)
}

async fn answer_v0(
    Path(id): Path<String>,
    state: State<Arc<CollectionsReader>>,
    Json(interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let collection_id = CollectionId(id);
    let state = state.clone();

    let query = interaction.query;
    let conversation = interaction.messages;

    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        let collection = state.get_collection(collection_id).await;
        let ai_service = state.get_embedding_service().get_ai_service().unwrap();

        let acknowledgement_msg = SseMessage::Acknowledge {
            message: "Acknowledged".to_string(),
        };
        let _ = tx
            .send(Ok(
                Event::default().data(serde_json::to_string(&acknowledgement_msg).unwrap())
            ))
            .await;

        let optimizing_query_mgs = SseMessage::OptimizingQuery {
            message: "Optimizing query".to_string(),
        };
        let _ = tx
            .send(Ok(
                Event::default().data(serde_json::to_string(&optimizing_query_mgs).unwrap())
            ))
            .await;

        let optimized_query = ai_service
            .chat(
                LlmType::GoogleQueryTranslator,
                query.clone(),
                Conversation { messages: vec![] },
                None,
            )
            .await
            .unwrap();
        let optimized_query_msg = SseMessage::OptimizedQuery {
            message: optimized_query.text.to_string(),
        };
        let _ = tx
            .send(Ok(
                Event::default().data(serde_json::to_string(&optimized_query_msg).unwrap())
            ))
            .await;

        if let Some(collection) = collection {
            let search_results = collection
                .search(SearchParams {
                    mode: SearchMode::Hybrid(HybridMode {
                        term: optimized_query.text,
                    }),
                    limit: Limit(5),
                    where_filter: HashMap::new(),
                    boost: HashMap::new(),
                    facets: HashMap::new(),
                    properties: crate::collection_manager::dto::Properties::Star,
                })
                .await
                .unwrap();

            let sources_msg = SseMessage::Sources {
                message: search_results.clone(),
            };
            let _ = tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&sources_msg).unwrap())
                ))
                .await;

            // ------------------------------------------------------------
            let search_result_str = serde_json::to_string(&search_results.hits).unwrap();

            let stream = ai_service
                .chat_stream(
                    LlmType::Answer,
                    query,
                    Conversation { messages: vec![] },
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

                        let answer_chunk_msg = SseMessage::AnswerChunk { message: chunk };

                        if let Err(_) = tx
                            .send(Ok(Event::default()
                                .data(serde_json::to_string(&answer_chunk_msg).unwrap())))
                            .await
                        {
                            break; // Client disconnected
                        }

                        if response.is_final {
                            break;
                        }
                    }
                    Err(e) => {
                        let error_msg = SseMessage::Error {
                            message: format!("Error during streaming: {}", e),
                        };
                        let _ = tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&error_msg).unwrap())
                            ))
                            .await;
                        break;
                    }
                }
            }
            // ------------------------------------------------------------
        } else {
            let error_msg = SseMessage::Error {
                message: "Collection not found".to_string(),
            };
            let _ = tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&error_msg).unwrap())
                ))
                .await;
        }
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}
