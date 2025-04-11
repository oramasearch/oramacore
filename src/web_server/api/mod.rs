use std::sync::{atomic::AtomicUsize, Arc};

use axum::{
    extract::{DefaultBodyLimit, State},
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{
    build_openapi, endpoint, reset_openapi,
    utoipa::openapi::{InfoBuilder, OpenApiBuilder},
    AddRoute,
};
use http::Request;
use metrics_exporter_prometheus::PrometheusHandle;
use tower_http::trace::TraceLayer;
use tracing::{info, info_span};

use crate::{
    build_info::get_build_version,
    collection_manager::sides::{ReadSide, WriteSide},
};
mod collection;
mod util;

pub fn api_config(
    write_side: Option<Arc<WriteSide>>,
    read_side: Option<Arc<ReadSide>>,
    prometheus_handle: Option<PrometheusHandle>,
) -> Router {
    reset_openapi();

    // build our application with a route
    let router = Router::new().add(index()).add(health()).add(openapi());

    let router = if let Some(prometheus_handle) = prometheus_handle {
        let metric = Router::new()
            .add(prometheus_handler())
            .with_state(Arc::new(prometheus_handle));
        info!("Prometheus metrics enabled");
        router.merge(metric)
    } else {
        router
    };

    let router = router.merge(collection::apis(write_side, read_side));

    let counter = Arc::new(AtomicUsize::new(0));

    const MAX_BODY_SIZE: usize = 1024 * 1024 * 100; // 100 MB

    router
        .layer(
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
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
}

#[endpoint(method = "GET", path = "/metrics", description = "Prometheus Metric")]
async fn prometheus_handler(prometheus_handle: State<Arc<PrometheusHandle>>) -> impl IntoResponse {
    prometheus_handle.render()
}

static INDEX_MESSAGE: &str = "OramaCore is up and running.";

#[endpoint(method = "GET", path = "/", description = "Welcome to Orama")]
async fn index() -> Json<String> {
    Json(format!("{} ({})", INDEX_MESSAGE, get_build_version()))
}

static HEALTH_MESSAGE: &str = "up";
#[endpoint(method = "GET", path = "/health", description = "Health check")]
async fn health() -> Json<&'static str> {
    Json(HEALTH_MESSAGE)
}

#[endpoint(method = "GET", path = "/openapi.json", description = "OpenAPI spec")]
async fn openapi() -> impl IntoResponse {
    let openapi = build_openapi(|| {
        OpenApiBuilder::new().info(InfoBuilder::new().title("Orama").version("0.1.0"))
    });

    Json(openapi)
}
