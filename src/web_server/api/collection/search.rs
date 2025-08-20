use std::{convert::Infallible, sync::Arc};

use axum::{
    extract::{Path, Query, State},
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    collection_manager::sides::read::AnalyticSearchEventInvocationType, types::NLPSearchRequest,
};
use crate::{
    collection_manager::sides::read::ReadSide,
    types::{ApiKey, CollectionId, CollectionStatsRequest, SearchParams},
    web_server::api::util::print_error,
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{collection_id}/search", post(search))
        .route(
            "/v1/collections/{collection_id}/nlp_search",
            post(nlp_search),
        )
        .route(
            "/v1/collections/{collection_id}/nlp_search_stream",
            post(nlp_search_streamed),
        )
        .route("/v1/collections/{collection_id}/stats", get(stats))
        .with_state(read_side)
}

#[derive(Deserialize)]
struct SearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn search(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<SearchParams>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    read_side
        .search(
            read_api_key,
            collection_id,
            json,
            AnalyticSearchEventInvocationType::Direct,
        )
        .await
        .map(Json)
}

async fn nlp_search(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<NLPSearchRequest>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    let logs = read_side.get_hook_logs();
    let log_sender = logs.get_sender(&collection_id);

    read_side
        .nlp_search(
            read_side.clone(),
            read_api_key,
            collection_id,
            json,
            log_sender,
        )
        .await
        .map(Json)
}

async fn nlp_search_streamed(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<NLPSearchRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let read_api_key = query.api_key;
    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    let logs = read_side.get_hook_logs();
    let log_sender = logs.get_sender(&collection_id);

    tokio::spawn(async move {
        match read_side
            .nlp_search_stream(
                read_side.clone(),
                read_api_key,
                collection_id,
                json,
                log_sender,
            )
            .await
        {
            Ok(mut stream) => {
                while let Some(result) = tokio_stream::StreamExt::next(&mut stream).await {
                    match result {
                        Ok(data) => {
                            let event = Event::default().data(json!(data).to_string());
                            if tx.send(Ok(event)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            print_error(&e, "Error in NLP search stream");
                            let _ = tx
                                .send(Ok(Event::default()
                                    .data(json!({ "error": e.to_string() }).to_string())))
                                .await;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = tx
                    .send(Ok(
                        Event::default().data(json!({ "error": e.to_string() }).to_string())
                    ))
                    .await;
            }
        }
    });

    Sse::new(rx_stream).keep_alive(axum::response::sse::KeepAlive::default())
}

async fn stats(
    Path(collection_id): Path<CollectionId>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    // We don't want to expose the variants on HTTP API, so we force `with_keys: false`
    // Anyway, if requested, we could add it on this enpoint in the future.
    read_side
        .collection_stats(
            read_api_key,
            collection_id,
            CollectionStatsRequest { with_keys: false },
        )
        .await
        .map(Json)
}
