use std::sync::Arc;

use axum::{body::Body, extract::State, response::IntoResponse, routing::get, Router};

use crate::{collection_manager::sides::read::ReadSide, types::ApiKey};

pub fn read_apis(write_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/analytics", get(analytics))
        .with_state(write_side)
}

async fn analytics(
    State(read_side): State<Arc<ReadSide>>,
    analytics_api_key: ApiKey,
) -> impl IntoResponse {
    if let Some(analytics) = read_side.get_analytics_logs() {
        let analytics_data = match analytics.get_and_erase(analytics_api_key).await {
            Ok(data) => data,
            Err(e) => {
                return Err((
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to retrieve analytics data: {e:?}"),
                ));
            }
        };

        Ok(Body::from_stream(analytics_data))
    } else {
        Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Analytics logs are not available. Please check if the analytics feature is enabled."
                .to_string(),
        ))
    }
}
