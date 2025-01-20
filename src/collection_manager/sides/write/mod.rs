mod collection;
mod collections;
mod embedding;
mod fields;
mod hooks;
mod operation;

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use anyhow::{Context, Result};
use collections::CollectionsWriter;
pub use collections::CollectionsWriterConfig;
use embedding::{start_calculate_embedding_loop, EmbeddingCalculationRequest};
pub use operation::*;

#[cfg(any(test, feature = "benchmarking"))]
pub use fields::*;
use tokio::sync::broadcast::Sender;
use tracing::{info, warn};

use crate::{
    collection_manager::dto::{CollectionDTO, CreateCollectionOptionDTO},
    embeddings::EmbeddingService,
    metrics::{AddedDocumentsLabels, ADDED_DOCUMENTS_COUNTER},
    types::{CollectionId, DocumentId, DocumentList},
};

pub struct WriteSide {
    sender: Sender<WriteOperation>,
    collections: CollectionsWriter,
    document_count: AtomicU64,
}

impl WriteSide {
    pub fn new(
        sender: Sender<WriteOperation>,
        config: CollectionsWriterConfig,
        embedding_service: Arc<EmbeddingService>,
    ) -> WriteSide {
        let (sx, rx) =
            tokio::sync::mpsc::channel::<EmbeddingCalculationRequest>(config.embedding_queue_limit);

        start_calculate_embedding_loop(embedding_service.clone(), rx, config.embedding_queue_limit);

        WriteSide {
            sender,
            collections: CollectionsWriter::new(config, sx),
            document_count: AtomicU64::new(0),
        }
    }

    pub async fn load(&mut self) -> Result<()> {
        self.collections.load().await
    }

    pub async fn commit(&self) -> Result<()> {
        self.collections.commit().await
    }

    pub async fn create_collection(&self, option: CreateCollectionOptionDTO) -> Result<()> {
        self.collections
            .create_collection(option, self.sender.clone())
            .await?;

        Ok(())
    }

    pub async fn write(
        &self,
        collection_id: CollectionId,
        document_list: DocumentList,
    ) -> Result<()> {
        info!("Inserting batch of {} documents", document_list.len());

        ADDED_DOCUMENTS_COUNTER
            .create(AddedDocumentsLabels {
                collection: collection_id.0.clone(),
            })
            .increment_by(document_list.len());

        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        let sender = self.sender.clone();

        for mut doc in document_list {
            let doc_id = self.document_count.fetch_add(1, Ordering::Relaxed);

            let doc_id_value = doc.get("id");
            // Forces the id to be set, if not set
            if doc_id_value.is_none() {
                doc.inner.insert(
                    "id".to_string(),
                    serde_json::Value::String(cuid2::create_id()),
                );
            } else if let Some(doc_id_value) = doc_id_value {
                if !doc_id_value.is_string() {
                    // The search result contains the document id and it is defined as a string.
                    // So, if the original document id is not a string, we should overwrite it with a new one
                    // Anyway, this implies the loss of the original document id. For instance we could support number as well
                    // TODO: think better
                    warn!("Document id is not a string, overwriting it with new one");
                    doc.inner.insert(
                        "id".to_string(),
                        serde_json::Value::String(cuid2::create_id()),
                    );
                }
            }

            let doc_id = DocumentId(doc_id);
            collection
                .process_new_document(doc_id, doc, sender.clone())
                .await
                .context("Cannot process document")?;
        }

        Ok(())
    }

    pub async fn list_collections(&self) -> Vec<CollectionDTO> {
        self.collections.list().await
    }

    pub async fn get_collection_dto(&self, collection_id: CollectionId) -> Option<CollectionDTO> {
        let collection = self.collections.get_collection(collection_id).await?;
        Some(collection.as_dto())
    }
}
