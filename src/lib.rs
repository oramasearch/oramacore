use std::sync::{atomic::AtomicU32, Arc};

use anyhow::{Context, Result};
use collection_manager::sides::{
    document_storage::{DocumentStorage, InMemoryDocumentStorage},
    read::CollectionsReader,
    write::{CollectionsWriter, WriteOperation},
};
use embeddings::{EmbeddingConfig, EmbeddingService};
use serde::Deserialize;
use tokio::sync::broadcast::Receiver;

pub mod indexes;
pub mod types;

pub mod code_parser;
pub mod nlp;

pub mod collection_manager;
pub mod document_storage;

pub mod web_server;

pub mod embeddings;

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
    pub output: SideChannelType,
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
        reader_side.output,
        SideChannelType::InMemory,
        "Only in-memory is supported"
    );

    let document_id_generator = Arc::new(AtomicU32::new(0));
    let collections_writer =
        CollectionsWriter::new(document_id_generator, sender, embedding_service.clone());

    let document_storage: Arc<dyn DocumentStorage> = Arc::new(InMemoryDocumentStorage::new());
    let collections_reader = CollectionsReader::new(embedding_service, document_storage);

    let collections_writer = Some(Arc::new(collections_writer));
    let collections_reader = Some(Arc::new(collections_reader));

    Ok((collections_writer, collections_reader, receiver))
}
