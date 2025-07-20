use crate::ai::state_machines::advanced_autoquery::{
    AdvancedAutoqueryConfig, AdvancedAutoqueryEvent, AdvancedAutoqueryStateMachine,
};
use crate::ai::state_machines::answer::{AnswerConfig, AnswerEvent, AnswerStateMachine};
use crate::collection_manager::sides::read::ReadSide;
use crate::types::{ApiKey, CollectionId, Interaction, InteractionLLMConfig, RelatedRequest};
use axum::extract::Query;
use axum::response::sse::Event;
use axum::response::Sse;
use axum::routing::post;
use axum::{extract::State, Json, Router};
use futures::Stream;
use futures::StreamExt;
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/collections/{collection_id}/generate/nlp_query",
            post(nlp_query_v1),
        )
        .route(
            "/v1/collections/{collection_id}/generate/answer",
            post(answer_v1),
        )
        .with_state(read_side)
}

#[derive(Deserialize)]
struct NlpQueryQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize)]
struct AnswerQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[derive(Deserialize)]
struct NlpQueryRequest {
    pub messages: Vec<crate::types::InteractionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_config: Option<InteractionLLMConfig>,
}

#[derive(Deserialize)]
struct AnswerRequest {
    pub messages: Vec<crate::types::InteractionMessage>,
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_config: Option<InteractionLLMConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_documents: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_similarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ragat_notation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related: Option<u32>,
}

#[axum::debug_handler]
async fn nlp_query_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<NlpQueryQueryParams>,
    Json(request): Json<NlpQueryRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (event_sender, event_receiver) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(event_receiver);

    // Spawn the state machine execution
    tokio::spawn(async move {
        // Get collection stats for the state machine
        let collection_stats = match read_side
            .collection_stats(
                query_params.api_key,
                collection_id,
                crate::types::CollectionStatsRequest { with_keys: false },
            )
            .await
        {
            Ok(stats) => stats,
            Err(e) => {
                let error_event = AdvancedAutoqueryEvent::Error {
                    error: e.to_string(),
                    state: "collection_stats_error".to_string(),
                };
                let event_json = serde_json::to_string(&error_event).unwrap();
                let sse_event = Event::default().data(event_json);
                let _ = event_sender.send(Ok(sse_event)).await;
                return;
            }
        };

        // Get LLM service from read_side
        let llm_service = read_side.get_llm_service();

        // Create the state machine
        let state_machine = AdvancedAutoqueryStateMachine::new(
            AdvancedAutoqueryConfig::default(),
            llm_service,
            request.llm_config,
            collection_stats,
            read_side.clone(),
            collection_id,
            query_params.api_key,
        );

        // Start the streaming state machine
        let event_stream = match state_machine
            .run_stream(request.messages, collection_id, query_params.api_key)
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                let error_event = AdvancedAutoqueryEvent::Error {
                    error: e.to_string(),
                    state: "state_machine_init_error".to_string(),
                };
                let event_json = serde_json::to_string(&error_event).unwrap();
                let sse_event = Event::default().data(event_json);
                let _ = event_sender.send(Ok(sse_event)).await;
                return;
            }
        };

        // Convert events to SSE format
        let mut event_stream = event_stream;
        while let Some(event) = event_stream.next().await {
            let event_json = serde_json::to_string(&event).unwrap();
            let sse_event = Event::default().data(event_json);

            if event_sender.send(Ok(sse_event)).await.is_err() {
                break;
            }
        }
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}

#[axum::debug_handler]
async fn answer_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<AnswerQueryParams>,
    Json(request): Json<AnswerRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (event_sender, event_receiver) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(event_receiver);

    // Spawn the state machine execution
    tokio::spawn(async move {
        // Construct the Interaction object
        let interaction = Interaction {
            interaction_id: format!("answer_{}", chrono::Utc::now().timestamp_millis()),
            visitor_id: format!("visitor_{}", chrono::Utc::now().timestamp_millis()),
            conversation_id: format!("conv_{}", chrono::Utc::now().timestamp_millis()),
            query: request.query,
            messages: request.messages,
            llm_config: request.llm_config,
            system_prompt_id: request.system_prompt_id,
            max_documents: request.max_documents.map(|x| x as usize),
            min_similarity: request.min_similarity,
            search_mode: request.search_mode,
            ragat_notation: request.ragat_notation,
            related: request.related.map(|x| RelatedRequest {
                enabled: Some(true),
                size: Some(x as usize),
                format: None,
            }),
        };

        // Get LLM service from read_side
        let llm_service = read_side.get_llm_service();

        // Create the state machine
        let state_machine = AnswerStateMachine::new(
            AnswerConfig::default(),
            llm_service,
            read_side.clone(),
            collection_id,
            query_params.api_key,
        );

        // Start the streaming state machine
        let event_stream = match state_machine
            .run_stream(interaction, collection_id, query_params.api_key)
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                let error_event = AnswerEvent::Error {
                    error: e.to_string(),
                    state: "state_machine_init_error".to_string(),
                };
                let event_json = serde_json::to_string(&error_event).unwrap();
                let sse_event = Event::default().data(event_json);
                let _ = event_sender.send(Ok(sse_event)).await;
                return;
            }
        };

        // Convert events to SSE format
        let mut event_stream = event_stream;
        while let Some(event) = event_stream.next().await {
            let event_json = serde_json::to_string(&event).unwrap();
            let sse_event = Event::default().data(event_json);

            if event_sender.send(Ok(sse_event)).await.is_err() {
                break;
            }
        }
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}
