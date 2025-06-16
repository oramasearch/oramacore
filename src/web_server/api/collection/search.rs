use std::{convert::Infallible, sync::Arc};

use axum::{
    extract::{Query, State},
    response::{sse::Event, IntoResponse, Sse},
    Json, Router,
};
use axum_openapi3::{utoipa::ToSchema, *};
use futures::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::IntoParams;

use crate::{
    collection_manager::sides::read::ReadSide,
    types::{ApiKey, CollectionId, CollectionStatsRequest, NLPSearchRequest, SearchParams},
    web_server::api::util::print_error,
};

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .add(search())
        .add(nlp_search())
        .add(nlp_search_streamed())
        .add(stats())
        .with_state(read_side)
}

#[derive(Deserialize, IntoParams, ToSchema)]
struct SearchQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/search",
    description = "Search Endpoint"
)]
async fn search(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<SearchParams>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    read_side
        .search(read_api_key, collection_id, json)
        .await
        .map(Json)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/nlp_search",
    description = "Advanced NLP search endpoint powered by AI"
)]
async fn nlp_search(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<NLPSearchRequest>,
) -> impl IntoResponse {
    let read_api_key = query.api_key;

    read_side
        .nlp_search(read_side.clone(), read_api_key, collection_id, json)
        .await
        .map(Json)
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/nlp_search_stream",
    description = "Advanced NLP search endpoint powered by AI - streamed"
)]
async fn nlp_search_streamed(
    collection_id: CollectionId,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<SearchQueryParams>,
    Json(json): Json<NLPSearchRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let read_api_key = query.api_key;
    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        match read_side
            .nlp_search_stream(read_side.clone(), read_api_key, collection_id, json)
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
                            let _ = tx.send(Ok(Event::default()
                                .data(json!({ "error": e.to_string() }).to_string())));
                        }
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Ok(
                    Event::default().data(json!({ "error": e.to_string() }).to_string())
                ));
            }
        }
    });

    Sse::new(rx_stream).keep_alive(axum::response::sse::KeepAlive::default())
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/stats",
    description = "Stats Endpoint"
)]
async fn stats(
    collection_id: CollectionId,
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
