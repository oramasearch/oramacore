use std::sync::{atomic::AtomicUsize, Arc};

use axum::{
    extract::{DefaultBodyLimit, State},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use http::{Request, StatusCode};
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
        .route("/health", get(health))
        .merge(
            Router::new()
                .route("/health/ready", get(health_ready))
                .with_state(read_side.clone()),
        );

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
/// Liveness: the process is up and serving HTTP.
async fn health() -> Json<&'static str> {
    Json(HEALTH_MESSAGE)
}

/// Readiness: 200 while the read side keeps up with the operation stream and commits on
/// schedule, 503 otherwise. The body is the same in both cases:
/// `{applied_offset, last_applied_offset, last_received_offset, last_applied_at, last_received_at, last_commit_at, ready}`.
/// A node without a read side has nothing to replicate and is always ready.
pub(crate) async fn health_ready(State(read_side): State<Option<Arc<ReadSide>>>) -> Response {
    let Some(read_side) = read_side else {
        return Json(serde_json::json!({
            "applied_offset": null,
            "last_applied_offset": null,
            "last_received_offset": null,
            "last_applied_at": null,
            "last_received_at": null,
            "last_commit_at": null,
            "ready": true,
        }))
        .into_response();
    };

    let report = read_side.readiness();
    let status = if report.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report)).into_response()
}
