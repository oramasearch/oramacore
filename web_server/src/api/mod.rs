use std::sync::Arc;

use axum::{response::IntoResponse, routing::get, Router};
use collection_manager::CollectionManager;
mod collection;

pub fn api_config() -> Router<Arc<CollectionManager>> {
    // build our application with a route
    let router = Router::new()
        .route("/", get(index))
        .route("/health", get(health));
    router.nest("/v0/collections", collection::apis())
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
