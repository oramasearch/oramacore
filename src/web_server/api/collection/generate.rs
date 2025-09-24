use crate::ai::state_machines::advanced_autoquery::{
    AdvancedAutoqueryConfig, AdvancedAutoqueryEvent, AdvancedAutoqueryStateMachine,
};
use crate::ai::state_machines::answer::{AnswerConfig, AnswerEvent, AnswerStateMachine};
use crate::collection_manager::sides::read::{
    AnalyticsHolder, AnalyticsMetadataFromRequest, ReadError, ReadSide,
};
use crate::types::{
    ApiKey, CollectionId, GetSystemPromptQueryParams, Interaction, InteractionLLMConfig,
};
use axum::extract::Query;
use axum::response::sse::Event;
use axum::response::Sse;
use axum::routing::{get, post};
use axum::{extract::State, Json, Router};
use futures::Stream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
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
        .route(
            "/v1/collections/{collection_id}/generate/get_default_system_prompt",
            get(get_default_system_prompt_v1),
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

async fn nlp_query_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<NlpQueryQueryParams>,
    Json(request): Json<NlpQueryRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (event_sender, event_receiver) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(event_receiver);

    tokio::spawn(async move {
        let collection_stats = match read_side
            .collection_stats(
                query_params.api_key,
                collection_id,
                crate::types::CollectionStatsRequest { with_keys: true },
            )
            .await
        {
            Ok(stats) => stats,
            Err(e) => {
                let error_event = AdvancedAutoqueryEvent::Error {
                    error: e.to_string(),
                    state: "collection_stats_error".to_string(),
                    is_terminal: Some(true),
                };
                let event_json = serde_json::to_string(&error_event).unwrap();
                let sse_event = Event::default().data(event_json);
                let _ = event_sender.send(Ok(sse_event)).await;
                return;
            }
        };

        let llm_service = read_side.get_llm_service();

        let state_machine = AdvancedAutoqueryStateMachine::new(
            AdvancedAutoqueryConfig::default(),
            llm_service,
            request.llm_config,
            collection_stats,
            read_side.clone(),
            collection_id,
            query_params.api_key,
        );

        let event_stream = match state_machine
            .run_stream(request.messages, collection_id, query_params.api_key)
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                let error_event = AdvancedAutoqueryEvent::Error {
                    error: e.to_string(),
                    state: "state_machine_init_error".to_string(),
                    is_terminal: Some(true),
                };
                let event_json = serde_json::to_string(&error_event).unwrap();
                let sse_event = Event::default().data(event_json);
                let _ = event_sender.send(Ok(sse_event)).await;
                return;
            }
        };

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

async fn answer_v1(
    collection_id: CollectionId,
    analytics_metadata: AnalyticsMetadataFromRequest,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<AnswerQueryParams>,
    Json(interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (event_sender, event_receiver) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(event_receiver);

    tokio::spawn(async move {
        let analytics_holder = AnalyticsHolder::new(
            read_side.clone(),
            collection_id,
            &interaction,
            analytics_metadata,
        );

        let analytics_holder = Arc::new(Mutex::new(analytics_holder));

        let llm_service = read_side.get_llm_service();

        let state_machine = AnswerStateMachine::new(
            AnswerConfig::default(),
            llm_service,
            read_side.clone(),
            collection_id,
            query_params.api_key,
            Some(analytics_holder.clone()),
        );

        let log_sender = read_side.get_hook_logs().get_sender(&collection_id);

        let event_stream = match state_machine
            .run_stream(interaction, collection_id, query_params.api_key, log_sender)
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                let error_event = AnswerEvent::Error {
                    error: e.to_string(),
                    state: "state_machine_init_error".to_string(),
                    is_terminal: Some(true),
                };
                let event_json = serde_json::to_string(&error_event).unwrap();
                let sse_event = Event::default().data(event_json);
                let _ = event_sender.send(Ok(sse_event)).await;
                return;
            }
        };

        let mut event_stream = event_stream;
        while let Some(event) = event_stream.next().await {
            match &event {
                AnswerEvent::RelatedQueries { queries } => {
                    if let Ok(rq) = serde_json::to_string(queries) {
                        let mut lock = analytics_holder.lock().await;
                        lock.set_generated_related_queries(rq);
                    }
                }
                AnswerEvent::SelectedLLM { provider, model } => {
                    let mut lock = analytics_holder.lock().await;
                    lock.set_llm_info(provider.to_string(), model.to_string());
                }
                AnswerEvent::SearchResults { results } => {
                    match serde_json::to_string(results) {
                        Ok(r) => {
                            let mut lock = analytics_holder.lock().await;
                            lock.set_full_context(r);
                        }
                        Err(err) => {
                            tracing::error!(error = ?err, "Cannot set full context");
                        }
                    };
                }
                _ => {}
            }

            let event_json = serde_json::to_string(&event).unwrap();
            let sse_event = Event::default().data(event_json);

            if event_sender.send(Ok(sse_event)).await.is_err() {
                break;
            }
        }

        drop(analytics_holder);
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}

async fn get_default_system_prompt_v1(
    collection_id: CollectionId,
    State(read_side): State<Arc<ReadSide>>,
    Query(query_params): Query<GetSystemPromptQueryParams>,
) -> Result<Json<serde_json::Value>, ReadError> {
    let read_api_key = query_params.api_key;
    let system_prompt_id = query_params.system_prompt_id;

    match read_side
        .get_default_system_prompt(collection_id, read_api_key, system_prompt_id)
        .await
    {
        Ok(prompt) => Ok(Json(json!({ "system_prompt": prompt }))),
        Err(e) => Err(e),
    }
}
