use std::sync::Arc;

use ai::{
    grpc::{GrpcRepo, GrpcRepoConfig},
    AiService,
};
use anyhow::{Context, Result};
use collection_manager::sides::{
    channel, CollectionsWriterConfig, IndexesConfig, OperationReceiver, ReadSide, WriteSide,
};
use embeddings::{EmbeddingConfig, EmbeddingService, ModelConfig};
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

pub mod embeddings;

mod capped_heap;
pub mod js;

mod metrics;

mod file_utils;

mod merger;
mod offset_storage;

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

#[derive(Debug, Deserialize, Clone)]
pub struct OramacoreConfig {
    pub http: HttpConfig,
    pub embeddings: EmbeddingConfig,
    pub writer_side: WriteSideConfig,
    pub reader_side: ReadSideConfig,
}

pub async fn start(config: OramacoreConfig) -> Result<()> {
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
        while let Ok(op) = receiver.recv().await {
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
        embeddings: embedding_config,
        writer_side,
        reader_side,
        ..
    } = config;

    let grpc_repo = GrpcRepo::new(
        GrpcRepoConfig {
            host: "0.0.0.0".parse().unwrap(),
            port: 50051,
            api_key: None,
        },
        embedding_config
            .models
            .iter()
            .filter_map(|(name, model)| {
                if let ModelConfig::Grpc(model) = model {
                    Some((name.clone(), model.clone()))
                } else {
                    None
                }
            })
            .collect(),
    );

    let ai_service = AiService::new(grpc_repo);

    let embedding_service = EmbeddingService::try_new(embedding_config, Some(Arc::new(ai_service)))
        .await
        .with_context(|| "Failed to initialize the EmbeddingService")?;
    let embedding_service = Arc::new(embedding_service);

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

    let mut write_side = WriteSide::new(
        sender.clone(),
        writer_side.config,
        embedding_service.clone(),
    );

    write_side.load().await.context("Cannot load write side")?;

    let nlp_service = Arc::new(NLPService::new());
    let mut collections_reader =
        ReadSide::try_new(embedding_service, nlp_service, reader_side.config)
            .context("Cannot create read side")?;

    collections_reader
        .load()
        .await
        .context("Cannot load collection reader")?;

    let write_side = Some(Arc::new(write_side));
    let collections_reader = Some(Arc::new(collections_reader));

    Ok((write_side, collections_reader, receiver))
}
