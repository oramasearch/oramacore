use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use anyhow::Result;
use api::api_config;
use metrics_exporter_prometheus::PrometheusHandle;
use serde::Deserialize;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::collection_manager::sides::{read::CollectionsReader, WriteSide};

mod api;

#[derive(Debug, Deserialize, Clone)]
pub struct HttpConfig {
    pub host: IpAddr,
    pub port: u16,
    pub allow_cors: bool,
    pub with_prometheus: bool,
}

pub struct WebServer {
    write_side: Option<Arc<WriteSide>>,
    collections_reader: Option<Arc<CollectionsReader>>,
    prometheus_handler: Option<PrometheusHandle>,
}

impl WebServer {
    pub fn new(
        write_side: Option<Arc<WriteSide>>,
        collections_reader: Option<Arc<CollectionsReader>>,
        prometheus_handler: Option<PrometheusHandle>,
    ) -> Self {
        Self {
            write_side,
            collections_reader,
            prometheus_handler,
        }
    }

    pub async fn start(self, config: HttpConfig) -> Result<()> {
        let addr = SocketAddr::new(config.host, config.port);

        let router = api_config(
            self.write_side,
            self.collections_reader,
            self.prometheus_handler,
        );

        let router = if config.allow_cors {
            info!("Enabling CORS");
            let cors_layer = CorsLayer::new()
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any)
                .allow_origin(tower_http::cors::Any);

            router.layer(cors_layer)
        } else {
            router
        };

        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

        info!("Address binded. Starting web server on http://{}", addr);
        let output = axum::serve(listener, router).await;

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}
