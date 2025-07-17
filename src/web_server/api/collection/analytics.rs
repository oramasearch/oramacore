use std::sync::Arc;

use axum::{body::Body, extract::State, response::IntoResponse, routing::get, Router};

use crate::collection_manager::sides::read::ReadSide;

pub fn read_apis(write_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/analytics", get(analytics))
        .with_state(write_side)
}

async fn analytics(State(read_side): State<Arc<ReadSide>>) -> impl IntoResponse {
    let analytics = read_side.get_analytics_logs();

    let analytics_data = analytics.get_and_erase().await.unwrap();

    Body::from_stream(analytics_data)
}
