// Make conversions public for testing
pub mod conversions;

use crate::ai::answer::{Answer, AnswerError};
use crate::collection_manager::sides::read::ReadSide;
use crate::types::{CollectionId, ReadApiKey};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::post;
use axum::{Json, Router};
use conversions::{
    answer_event_to_openai_chunk, openai_request_to_interaction, OpenAIChatRequest,
    OpenAIStreamEvent, ResponseAccumulator,
};
use futures::Stream;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info};

async fn openai_chat_completion_stream(
    collection_id: CollectionId,
    api_key: ReadApiKey,
    read_side: Arc<ReadSide>,
    request: OpenAIChatRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AnswerError> {
    info!(
        "OpenAI chat completion request (streaming) for collection {}",
        collection_id
    );

    let interaction =
        openai_request_to_interaction(request, collection_id).map_err(AnswerError::Generic)?;
    let answer = Answer::try_new(read_side.clone(), collection_id, api_key).await?;
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();
    let logs = read_side.get_hook_logs();
    let log_sender = logs.get_sender(&collection_id);

    tokio::spawn(async move {
        let r = answer.answer(interaction, answer_sender, log_sender).await;
        if let Err(e) = r {
            error!(error = ?e, "Failed to run answer");
        }
    });

    let (http_sender, http_receiver) = mpsc::channel(10);
    let stream_id = format!("chatcmpl-{}", cuid2::create_id());

    tokio::spawn(async move {
        while let Some(event) = answer_receiver.recv().await {
            debug!(event = ?event, "Received answer event for OpenAI streaming");

            if let Some(openai_event) = answer_event_to_openai_chunk(event, &stream_id) {
                let is_done = matches!(openai_event, OpenAIStreamEvent::Done);

                let ev = match openai_event {
                    OpenAIStreamEvent::Chunk(json) => Event::default().data(json),
                    OpenAIStreamEvent::Done => Event::default().data("[DONE]"),
                };

                if let Err(err) = http_sender.send(Ok(ev)).await {
                    error!(error = ?err, "Cannot stream to client");
                    break;
                }

                if is_done {
                    break;
                }
            }
        }
    });

    let rx_stream = ReceiverStream::new(http_receiver);
    Ok(Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text(""),
    ))
}

async fn openai_chat_completion(
    collection_id: CollectionId,
    api_key: ReadApiKey,
    read_side: Arc<ReadSide>,
    request: OpenAIChatRequest,
) -> Result<Response, AnswerError> {
    info!(
        "OpenAI chat completion request (non-streaming) for collection {}",
        collection_id
    );

    let interaction =
        openai_request_to_interaction(request, collection_id).map_err(AnswerError::Generic)?;
    let answer = Answer::try_new(read_side.clone(), collection_id, api_key).await?;
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();
    let logs = read_side.get_hook_logs();
    let log_sender = logs.get_sender(&collection_id);

    tokio::spawn(async move {
        let r = answer.answer(interaction, answer_sender, log_sender).await;
        if let Err(e) = r {
            error!(error = ?e, "Failed to run answer");
        }
    });

    let mut accumulator = ResponseAccumulator::new();
    while let Some(event) = answer_receiver.recv().await {
        debug!(event = ?event, "Received answer event for OpenAI non-streaming");
        accumulator.add_event(event);
    }

    info!("Completed accumulating OpenAI response");

    let request_id = format!("chatcmpl-{}", cuid2::create_id());
    let response_json = accumulator.to_openai_response(&request_id);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(response_json.into())
        .expect("Failed to build response"))
}

async fn openai_chat_completion_handler(
    Path(collection_id): Path<CollectionId>,
    api_key: ReadApiKey,
    State(read_side): State<Arc<ReadSide>>,
    Json(request): Json<OpenAIChatRequest>,
) -> Response {
    debug!(
        collection_id = %collection_id,
        stream = request.stream,
        "OpenAI chat completion handler"
    );

    if request.stream {
        match openai_chat_completion_stream(collection_id, api_key, read_side, request).await {
            Ok(sse) => sse.into_response(),
            Err(e) => {
                error!(error = ?e, "Failed to create streaming response");
                e.into_response()
            }
        }
    } else {
        match openai_chat_completion(collection_id, api_key, read_side, request).await {
            Ok(response) => response,
            Err(e) => {
                error!(error = ?e, "Failed to create response");
                e.into_response()
            }
        }
    }
}

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route(
            "/v1/{collection_id}/openai/chat/completions",
            post(openai_chat_completion_handler),
        )
        .with_state(read_side)
}
