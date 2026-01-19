use std::sync::{atomic::AtomicUsize, Arc};

use axum::{
    extract::{DefaultBodyLimit, State},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use http::Request;
use metrics_exporter_prometheus::PrometheusHandle;
use tower_http::trace::TraceLayer;
use tracing::{info, info_span};

use crate::{
    build_info::get_build_version,
    collection_manager::sides::{read::ReadSide, write::WriteSide},
};
pub mod collection;
mod util;

pub fn api_config(
    write_side: Option<Arc<WriteSide>>,
    read_side: Option<Arc<ReadSide>>,
    prometheus_handle: Option<PrometheusHandle>,
) -> Router {
    // build our application with a route
    let router = Router::new()
        .route("/", get(index))
        .route("/health", get(health));

    let router = if let Some(prometheus_handle) = prometheus_handle {
        let metric = Router::new()
            .route("/metrics", get(prometheus_handler))
            .with_state(Arc::new(prometheus_handle));
        info!("Prometheus metrics enabled");
        router.merge(metric)
    } else {
        router
    };

    let router = router.merge(collection::apis(write_side, read_side));

    let counter = Arc::new(AtomicUsize::new(0));

    const MAX_BODY_SIZE: usize = 1024 * 1024 * 400; // 400 MB

    router
        .layer(
            TraceLayer::new_for_http().make_span_with(move |request: &Request<_>| {
                let req_id = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                info_span!(
                    "http_request",
                    req_id,
                    method = ?request.method(),
                    path = ?request.uri().path_and_query().map(|pq| pq.path()),
                )
            }),
        )
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
}

async fn prometheus_handler(prometheus_handle: State<Arc<PrometheusHandle>>) -> impl IntoResponse {
    prometheus_handle.render()
}

static INDEX_MESSAGE: &str = "OramaCore is up and running.";

async fn index() -> Json<String> {
    Json(format!("{} ({})", INDEX_MESSAGE, get_build_version()))
}

static HEALTH_MESSAGE: &str = "up";
async fn health() -> Json<&'static str> {
    Json(HEALTH_MESSAGE)
}
