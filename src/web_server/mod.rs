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

use crate::collection_manager::sides::{ReadSide, WriteSide};

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
    read_side: Option<Arc<ReadSide>>,
    prometheus_handler: Option<PrometheusHandle>,
}

impl WebServer {
    pub fn new(
        write_side: Option<Arc<WriteSide>>,
        read_side: Option<Arc<ReadSide>>,
        prometheus_handler: Option<PrometheusHandle>,
    ) -> Self {
        Self {
            write_side,
            read_side,
            prometheus_handler,
        }
    }

    pub async fn start(self, config: HttpConfig) -> Result<()> {
        let addr = SocketAddr::new(config.host, config.port);

        let router = api_config(
            self.write_side,
            self.read_side.clone(),
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
        let output = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal(self.read_side))
            .await;

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

async fn shutdown_signal(reader_side: Option<Arc<ReadSide>>) {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    // wait for a CTRL+C signal
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    };

    if let Some(reader_side) = reader_side {
        info!("Stopping reader side");
        reader_side.stop().await;
    }

    info!("Shutting down web server");
}
