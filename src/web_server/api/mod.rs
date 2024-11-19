use std::sync::{atomic::AtomicUsize, Arc};

use axum::{extract::MatchedPath, response::IntoResponse, routing::get, Router};
use http::Request;
use tower_http::trace::TraceLayer;
use tracing::info_span;

use crate::collection_manager::CollectionManager;
mod collection;

pub fn api_config() -> Router<Arc<CollectionManager>> {
    // build our application with a route
    let router = Router::new()
        .route("/", get(index))
        .route("/health", get(health));
    let router = router.nest("/v0/collections", collection::apis());

    let counter = Arc::new(AtomicUsize::new(0));

    router.layer(
        TraceLayer::new_for_http().make_span_with(move |request: &Request<_>| {
            let req_id = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            info_span!(
                "http_request",
                req_id,
                method = ?request.method(),
                path = ?request.uri(),
            )
        }),
    )
}

static INDEX_MESSAGE: &str = "hi! welcome to Orama";
async fn index() -> impl IntoResponse {
    println!("index");
    INDEX_MESSAGE
}

static HEALTH_MESSAGE: &str = "up";
async fn health() -> impl IntoResponse {
    println!("health");
    HEALTH_MESSAGE
}
