mod collection;
mod collections;
mod embedding;
mod fields;
mod operation;

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use super::hooks::{HookName, HookValue};
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
    js::deno::JavaScript,
    js::deno::Operation,
    metrics::{AddedDocumentsLabels, ADDED_DOCUMENTS_COUNTER},
    types::{CollectionId, DocumentId, DocumentList},
};

pub struct WriteSide {
    sender: Sender<WriteOperation>,
    collections: CollectionsWriter,
    document_count: AtomicU64,
    javascript_runtime: Arc<JavaScript>,
    hook_storage: super::hooks::HookStorage,
}

impl WriteSide {
    pub fn new(
        sender: Sender<WriteOperation>,
        config: CollectionsWriterConfig,
        embedding_service: Arc<EmbeddingService>,
        javascript_runtime: Arc<JavaScript>,
    ) -> WriteSide {
        let (sx, rx) =
            tokio::sync::mpsc::channel::<EmbeddingCalculationRequest>(config.embedding_queue_limit);

        start_calculate_embedding_loop(embedding_service.clone(), rx, config.embedding_queue_limit);

        WriteSide {
            sender,
            collections: CollectionsWriter::new(config, sx),
            document_count: AtomicU64::new(0),
            javascript_runtime,
            hook_storage: super::hooks::HookStorage::new(),
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
            .get_collection(collection_id.clone())
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        let sender = self.sender.clone();

        let select_embedding_properties_hook =
            self.get_javascript_hook(collection_id, HookName::SelectEmbeddingsProperties);

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

            if let Some(before_index_hook) = &select_embedding_properties_hook {
                let code = before_index_hook.code.clone();
                let properties: Vec<String> = self
                    .javascript_runtime
                    .eval(
                        Operation::SelectEmbeddingsProperties,
                        code,
                        doc.inner.clone(),
                    )
                    .await?;
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

    pub fn insert_javascript_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
        code: String,
    ) -> Result<()> {
        self.hook_storage.insert_hook(collection_id, name, code)
    }

    pub fn get_javascript_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<HookValue> {
        self.hook_storage.get_hook(collection_id, name)
    }

    pub fn delete_javascript_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<(String, HookValue)> {
        self.hook_storage.delete_hook(collection_id, name)
    }

    pub fn list_javascript_hooks(
        &self,
        collection_id: CollectionId,
    ) -> HashMap<HookName, HookValue> {
        self.hook_storage.list_hooks(collection_id)
    }
}
