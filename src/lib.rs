use std::{path::PathBuf, sync::Arc};

use ai::{AIService, AIServiceConfig};
use anyhow::{Context, Result};
use collection_manager::sides::{
    channel_creator, hooks::HooksRuntime, InputSideChannelType, OutputSideChannelType, ReadSide,
    ReadSideConfig, WriteSide, WriteSideConfig,
};
use metrics_exporter_prometheus::PrometheusBuilder;
use nlp::NLPService;
use serde::Deserialize;
#[allow(unused_imports)]
use tracing::{info, warn};
use web_server::{HttpConfig, WebServer};

pub mod indexes;
pub mod types;

pub mod code_parser;
pub mod nlp;

pub mod collection_manager;

pub mod web_server;

mod capped_heap;
pub mod js;

mod metrics;

mod file_utils;

mod merger;
mod offset_storage;

pub mod ai;

#[cfg(test)]
mod tests;

#[derive(Debug, Deserialize, Clone, Default)]
pub struct LogConfig {
    pub file_path: Option<PathBuf>,
}

#[derive(Deserialize, Clone)]
pub struct OramacoreConfig {
    pub log: LogConfig,
    pub http: HttpConfig,
    pub ai_server: AIServiceConfig,
    #[cfg(any(test, feature = "writer"))]
    pub writer_side: WriteSideConfig,
    #[cfg(any(test, feature = "reader"))]
    pub reader_side: ReadSideConfig,
}

pub async fn start(config: OramacoreConfig) -> Result<()> {
    info!("Starting oramacore");

    let prometheus_hadler = if config.http.with_prometheus {
        Some(
            PrometheusBuilder::new()
                .install_recorder()
                .context("failed to install recorder")?,
        )
    } else {
        None
    };

    let (write_side, read_side) = build_orama(config.clone()).await?;

    info!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );

    let web_server = WebServer::new(write_side, read_side, prometheus_hadler);
    web_server.start(config.http).await?;

    Ok(())
}

pub async fn build_orama(
    config: OramacoreConfig,
) -> Result<(Option<Arc<WriteSide>>, Option<Arc<ReadSide>>)> {
    info!("Building ai_service");
    let ai_service = AIService::new(config.ai_server);
    let ai_service = Arc::new(ai_service);

    info!("Building hooks_runtime");
    let hooks_runtime = HooksRuntime::new(50).await;
    let hooks_runtime = Arc::new(hooks_runtime);

    #[cfg(feature = "writer")]
    let writer_sender_config: Option<OutputSideChannelType> =
        Some(config.writer_side.output.clone());
    #[cfg(not(feature = "writer"))]
    let writer_sender_config = None;
    #[cfg(feature = "reader")]
    let reader_sender_config: Option<InputSideChannelType> = Some(config.reader_side.input.clone());
    #[cfg(not(feature = "reader"))]
    let reader_sender_config = None;

    let (sender_creator, receiver_creator) =
        channel_creator(writer_sender_config, reader_sender_config).await?;

    info!("Building nlp_service");
    let nlp_service = Arc::new(NLPService::new());

    #[cfg(feature = "writer")]
    let write_side = {
        info!("Building write_side");
        let sender_creator = sender_creator.expect("Sender is not created");

        let write_side = WriteSide::try_load(
            sender_creator,
            config.writer_side,
            ai_service.clone(),
            hooks_runtime,
            nlp_service.clone(),
        )
        .await
        .context("Cannot create write side")?;
        Some(write_side)
    };
    #[cfg(not(feature = "writer"))]
    let write_side = {
        warn!("Building write_side skipped due to compilation flag");
        None
    };

    #[cfg(feature = "reader")]
    let read_side = {
        info!("Building read_side");

        let receiver_creator = receiver_creator.expect("Receiver is not created");
        let read_side = ReadSide::try_load(
            receiver_creator,
            ai_service,
            nlp_service,
            config.reader_side,
        )
        .await
        .context("Cannot create read side")?;
        Some(read_side)
    };
    #[cfg(not(feature = "reader"))]
    let read_side = {
        warn!("Building read_side skipped due to compilation flag");
        None
    };

    Ok((write_side, read_side))
}
