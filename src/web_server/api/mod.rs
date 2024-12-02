use std::sync::{atomic::AtomicUsize, Arc};

use axum::{response::IntoResponse, Router, Json};
use axum_openapi3::{endpoint, build_openapi, reset_openapi, utoipa::openapi::{InfoBuilder, OpenApiBuilder}, AddRoute};
use http::Request;
use tower_http::trace::TraceLayer;
use tracing::info_span;

use crate::collection_manager::sides::{read::CollectionsReader, write::CollectionsWriter};
mod collection;

pub fn api_config(
    writers: Option<Arc<CollectionsWriter>>,
    readers: Option<Arc<CollectionsReader>>,
) -> Router {
    reset_openapi();

    // build our application with a route
    let router = Router::new()
        .add(index())
        .add(health())
        .add(openapi());
    let router = router.nest("/", collection::apis(writers, readers));

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

#[endpoint(method = "GET", path = "/", description = "Welcome to Orama")]
async fn index() -> Json<&'static str> {
    println!("index");
    Json(INDEX_MESSAGE)
}

static HEALTH_MESSAGE: &str = "up";
#[endpoint(method = "GET", path = "/health", description = "Health check")]
async fn health() -> Json<&'static str> {
    println!("health");
    Json(HEALTH_MESSAGE)
}

#[endpoint(method = "GET", path = "/openapi.json", description = "OpenAPI spec")]
async fn openapi() -> impl IntoResponse {
    let openapi = build_openapi(|| {
        OpenApiBuilder::new()
            .info(InfoBuilder::new().title("Orama").version("0.1.0"))
    });

    Json(openapi)
}
