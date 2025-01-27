use std::{path::PathBuf, sync::Arc};

use ai::{AIService, AIServiceConfig};
use anyhow::{Context, Result};
use collection_manager::sides::{
    channel, hooks::HooksRuntime, CollectionsWriterConfig, IndexesConfig, OperationReceiver,
    ReadSide, WriteSide,
};
use metrics_exporter_prometheus::PrometheusBuilder;
use nlp::NLPService;
use serde::Deserialize;
use tracing::info;
use web_server::{HttpConfig, WebServer};

pub mod indexes;
pub mod types;

pub mod code_parser;
pub mod nlp;

pub mod collection_manager;

pub mod web_server;

// pub mod embeddings;

mod capped_heap;
pub mod js;

mod metrics;

mod file_utils;

mod merger;
mod offset_storage;

mod field_id_hashmap;

pub mod ai;

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "benchmarking"))]
pub mod test_utils;

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub enum SideChannelType {
    #[serde(rename = "in-memory")]
    InMemory,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WriteSideConfig {
    pub output: SideChannelType,
    pub config: CollectionsWriterConfig,
}
#[derive(Debug, Deserialize, Clone)]
pub struct ReadSideConfig {
    pub input: SideChannelType,
    pub config: IndexesConfig,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct LogConfig {
    pub file_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OramacoreConfig {
    pub log: LogConfig,
    pub http: HttpConfig,
    pub ai_server: AIServiceConfig,
    pub writer_side: WriteSideConfig,
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

    let (write_side, read_side, receiver) = build_orama(config.clone()).await?;

    connect_write_and_read_side(receiver, read_side.clone().unwrap());

    info!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );

    let web_server = WebServer::new(write_side, read_side, prometheus_hadler);
    web_server.start(config.http).await?;

    Ok(())
}

pub fn connect_write_and_read_side(mut receiver: OperationReceiver, read_side: Arc<ReadSide>) {
    tokio::spawn(async move {
        while let Some(op) = receiver.recv().await {
            read_side.update(op).await.expect("OUCH!");
        }
    });
}

pub async fn build_orama(
    config: OramacoreConfig,
) -> Result<(
    Option<Arc<WriteSide>>,
    Option<Arc<ReadSide>>,
    OperationReceiver,
)> {
    let OramacoreConfig {
        ai_server,
        writer_side,
        reader_side,
        ..
    } = config;

    info!("Building ai_service");
    let ai_service = AIService::new(ai_server);
    let ai_service = Arc::new(ai_service);

    info!("Building hooks_runtime");
    let hooks_runtime = HooksRuntime::new(50).await;
    let hooks_runtime = Arc::new(hooks_runtime);

    let (sender, receiver) = channel(10_000);

    assert_eq!(
        writer_side.output,
        SideChannelType::InMemory,
        "Only in-memory is supported"
    );
    assert_eq!(
        reader_side.input,
        SideChannelType::InMemory,
        "Only in-memory is supported"
    );

    info!("Building nlp_service");
    let nlp_service = Arc::new(NLPService::new());

    info!("Building write_side");
    let mut write_side = WriteSide::new(
        sender.clone(),
        writer_side.config,
        ai_service.clone(),
        hooks_runtime,
        nlp_service.clone(),
    );
    write_side.load().await.context("Cannot load write side")?;

    info!("Building read_side");
    let mut read_side = ReadSide::try_new(ai_service, nlp_service, reader_side.config)
        .context("Cannot create read side")?;
    read_side
        .load()
        .await
        .context("Cannot load collection reader")?;

    let write_side = Some(Arc::new(write_side));
    let collections_reader = Some(Arc::new(read_side));

    Ok((write_side, collections_reader, receiver))
}
