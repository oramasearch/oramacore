use std::sync::{atomic::AtomicU32, Arc};

use anyhow::{Context, Result};
use collection_manager::sides::{
    document_storage::{DocumentStorage, InMemoryDocumentStorage},
    read::{CollectionsReader, IndexesConfig},
    write::{CollectionsWriter, WriteOperation},
};
use embeddings::{EmbeddingConfig, EmbeddingService};
use metrics_exporter_prometheus::PrometheusBuilder;
use serde::Deserialize;
use tokio::sync::broadcast::Receiver;
use tracing::info;
use web_server::{HttpConfig, WebServer};

pub mod indexes;
pub mod types;

pub mod code_parser;
pub mod nlp;

pub mod collection_manager;
pub mod document_storage;

pub mod web_server;

pub mod embeddings;

mod capped_heap;
pub mod js;

mod metrics;

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
}
#[derive(Debug, Deserialize, Clone)]
pub struct ReadSideConfig {
    pub input: SideChannelType,
    pub data: IndexesConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RustoramaConfig {
    http: HttpConfig,
    embeddings: EmbeddingConfig,
    writer_side: WriteSideConfig,
    reader_side: ReadSideConfig,
}

pub async fn start(config: RustoramaConfig) -> Result<()> {
    let prometheus_hadler = if config.http.with_prometheus {
        Some(
            PrometheusBuilder::new()
                .install_recorder()
                .context("failed to install recorder")?,
        )
    } else {
        None
    };

    let (writer, reader, mut receiver) =
        build_orama(config.embeddings, config.writer_side, config.reader_side).await?;

    let collections_reader = reader.clone().unwrap();
    tokio::spawn(async move {
        while let Ok(op) = receiver.recv().await {
            collections_reader.update(op).await.expect("OUCH!");
        }
    });

    info!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );

    let web_server = WebServer::new(writer, reader, prometheus_hadler);
    web_server.start(config.http).await?;

    Ok(())
}

pub async fn build_orama(
    embedding_config: EmbeddingConfig,
    writer_side: WriteSideConfig,
    reader_side: ReadSideConfig,
) -> Result<(
    Option<Arc<CollectionsWriter>>,
    Option<Arc<CollectionsReader>>,
    Receiver<WriteOperation>,
)> {
    let embedding_service = EmbeddingService::try_new(embedding_config)
        .await
        .with_context(|| "Failed to initialize the EmbeddingService")?;
    let embedding_service = Arc::new(embedding_service);

    let (sender, receiver) = tokio::sync::broadcast::channel(10_000);

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

    let document_id_generator = Arc::new(AtomicU32::new(0));
    let collections_writer =
        CollectionsWriter::new(document_id_generator, sender, embedding_service.clone());

    let document_storage: Arc<dyn DocumentStorage> = Arc::new(InMemoryDocumentStorage::new());
    let collections_reader =
        CollectionsReader::new(embedding_service, document_storage, reader_side.data);

    let collections_writer = Some(Arc::new(collections_writer));
    let collections_reader = Some(Arc::new(collections_reader));

    Ok((collections_writer, collections_reader, receiver))
}
