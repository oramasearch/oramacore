use std::{path::PathBuf, sync::{atomic::AtomicU64, Arc}};

use anyhow::{anyhow, Context, Ok, Result};
use dashmap::DashMap;
use tokio::sync::broadcast::Sender;
use tracing::{info, warn};

use crate::{
    collection_manager::dto::CollectionDTO,
    embeddings::EmbeddingService,
    metrics::{AddedDocumentsLabels, ADDED_DOCUMENTS_COUNTER},
    types::{CollectionId, DocumentId, DocumentList},
};

use crate::collection_manager::dto::{CreateCollectionOptionDTO, LanguageDTO};

use super::{collection::CollectionWriter, GenericWriteOperation, WriteOperation};

pub struct CollectionsWriter {
    document_id_generator: Arc<AtomicU64>,
    sender: Sender<WriteOperation>,
    embedding_service: Arc<EmbeddingService>,
    collections: DashMap<CollectionId, CollectionWriter>,
}

impl CollectionsWriter {
    pub fn new(
        document_id_generator: Arc<AtomicU64>,
        sender: Sender<WriteOperation>,
        embedding_service: Arc<EmbeddingService>,
    ) -> CollectionsWriter {
        CollectionsWriter {
            document_id_generator,
            sender,
            embedding_service,
            collections: DashMap::new(),
        }
    }

    fn generate_document_id(&self) -> DocumentId {
        let id = self
            .document_id_generator
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        DocumentId(id)
    }

    pub async fn create_collection(
        &self,
        collection_option: CreateCollectionOptionDTO,
    ) -> Result<CollectionId> {
        let CreateCollectionOptionDTO {
            id,
            description,
            language,
            typed_fields,
        } = collection_option;

        let id = CollectionId(id);

        self.sender
            .send(WriteOperation::Generic(
                GenericWriteOperation::CreateCollection { id: id.clone() },
            ))
            .context("Cannot send create collection")?;

        let collection = CollectionWriter::new(
            id.clone(),
            description,
            language.unwrap_or(LanguageDTO::English),
        );

        for (field_name, field_type) in typed_fields {
            let field_id = collection.get_field_id_by_name(&field_name);

            collection
                .create_field(
                    field_id,
                    field_name,
                    field_type,
                    self.embedding_service.clone(),
                    &self.sender,
                )
                .await
                .context("Cannot create field")?;
        }

        // This substitute the previous value and this is wrong.
        // We should *NOT* allow to overwrite a collection
        // We should return an error if the collection already exists
        // NB: the check of the existence of the collection and the insertion should be done atomically
        // TODO: do it.
        self.collections.insert(id.clone(), collection);

        Ok(id)
    }

    pub fn list(&self) -> Vec<CollectionDTO> {
        self.collections
            .iter()
            .map(|e| {
                let coll = e.value();

                coll.as_dto()
            })
            .collect()
    }

    pub fn get_collection_dto(&self, collection_id: CollectionId) -> Option<CollectionDTO> {
        let collection = self.collections.get(&collection_id);
        collection.map(|c| c.as_dto())
    }

    pub fn commit(&self, data_dir: PathBuf) -> Result<()> {

        Ok(())
    }

    pub fn load(&self, data_dir: PathBuf) -> Result<()> {

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
            .get(&collection_id)
            .ok_or_else(|| anyhow!("Collection not found"))?;

        for mut doc in document_list {
            let doc_id = self.generate_document_id();

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

            collection
                .process_new_document(
                    doc_id,
                    doc,
                    self.embedding_service.clone(),
                    &self.sender.clone(),
                )
                .await
                .context("Cannot process document")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_writer_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<CollectionsWriter>();
    }
}
