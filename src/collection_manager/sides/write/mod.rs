mod collection;
mod collections;
mod embedding;
mod fields;
mod operation;

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use collections::CollectionsWriter;
pub use collections::CollectionsWriterConfig;
use embedding::{start_calculate_embedding_loop, EmbeddingCalculationRequest};
pub use operation::*;

#[cfg(any(test, feature = "benchmarking"))]
pub use fields::*;

use crate::{
    collection_manager::dto::{CollectionDTO, CreateCollectionOptionDTO},
    embeddings::EmbeddingService,
    file_utils::BufferedFile,
    metrics::{AddedDocumentsLabels, ADDED_DOCUMENTS_COUNTER},
    types::{CollectionId, DocumentId, DocumentList},
};

pub struct WriteSide {
    sender: OperationSender,
    collections: CollectionsWriter,
    document_count: AtomicU64,
    data_dir: PathBuf,
}

impl WriteSide {
    pub fn new(
        sender: OperationSender,
        config: CollectionsWriterConfig,
        embedding_service: Arc<EmbeddingService>,
    ) -> WriteSide {
        let data_dir = config.data_dir.clone();

        let (sx, rx) =
            tokio::sync::mpsc::channel::<EmbeddingCalculationRequest>(config.embedding_queue_limit);

        start_calculate_embedding_loop(embedding_service.clone(), rx, config.embedding_queue_limit);

        WriteSide {
            sender,
            collections: CollectionsWriter::new(config, sx),
            document_count: AtomicU64::new(0),
            data_dir,
        }
    }

    pub async fn load(&mut self) -> Result<()> {
        self.collections.load().await?;

        let info: WriteSideInfo = match BufferedFile::open(self.data_dir.join("info.json"))
            .and_then(|f| f.read_json_data())
            .context("Cannot read info file")
        {
            Ok(info) => info,
            Err(err) => {
                warn!("Cannot read info file: {}. Skip loading", err);
                return Ok(());
            }
        };

        self.document_count
            .store(info.document_count, Ordering::Relaxed);
        self.sender.set_offset(info.offset);

        Ok(())
    }

    pub async fn commit(&self) -> Result<()> {
        let offset = self.sender.offset();

        self.collections.commit().await?;

        // This load is not atomic with the commit.
        // This means, we save a document count possible higher.
        // Anyway it is not a problem, because the document count is only used for the document id generation
        // So, if something goes wrong, we save an higher number, and this is ok.
        let document_count = self.document_count.load(Ordering::Relaxed);
        let info = WriteSideInfo {
            document_count,
            offset,
        };
        BufferedFile::create(self.data_dir.join("info.json"))
            .context("Cannot create info file")?
            .write_json_data(&info)
            .context("Cannot write info file")?;

        Ok(())
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

#[derive(Serialize, Deserialize)]
struct WriteSideInfo {
    document_count: u64,
    offset: Offset,
}
