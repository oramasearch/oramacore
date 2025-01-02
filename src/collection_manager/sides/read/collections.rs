use std::{
    collections::HashMap,
    ops::Deref,
    path::PathBuf,
    sync::{atomic::AtomicU64, Arc},
};

use crate::{
    collection_manager::{
        sides::{
            document_storage::DocumentStorage,
            write::{
                CollectionWriteOperation, DocumentFieldIndexOperation, GenericWriteOperation,
                WriteOperation,
            },
        },
        CollectionId,
    },
    embeddings::EmbeddingService,
    metrics::{
        CollectionAddedLabels, CollectionOperationLabels, COLLECTION_ADDED_COUNTER,
        COLLECTION_OPERATION_COUNTER,
    },
};

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{error, info};

use super::collection::CollectionReader;

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {
    pub data_dir: PathBuf,
    pub max_size_per_chunk: usize,
}

pub struct CollectionsReader {
    embedding_service: Arc<EmbeddingService>,
    collections: RwLock<HashMap<CollectionId, CollectionReader>>,
    document_storage: Arc<dyn DocumentStorage>,
    posting_id_generator: Arc<AtomicU64>,
    indexes_config: IndexesConfig,
}
impl CollectionsReader {
    pub fn new(
        embedding_service: Arc<EmbeddingService>,
        document_storage: Arc<dyn DocumentStorage>,
        indexes_config: IndexesConfig,
    ) -> Self {
        Self {
            embedding_service,
            collections: Default::default(),
            document_storage,
            posting_id_generator: Arc::new(AtomicU64::new(0)),
            indexes_config,
        }
    }

    pub async fn update(&self, op: WriteOperation) -> Result<()> {
        match op {
            WriteOperation::Generic(GenericWriteOperation::CreateCollection { id }) => {
                info!("CreateCollection {:?}", id);
                COLLECTION_ADDED_COUNTER
                    .create(CollectionAddedLabels {
                        collection: id.0.to_string(),
                    })
                    .increment_by_one();

                let collection_reader = CollectionReader::try_new(
                    id.clone(),
                    self.embedding_service.clone(),
                    Arc::clone(&self.document_storage),
                    self.posting_id_generator.clone(),
                    self.indexes_config.clone(),
                )?;

                self.collections.write().await.insert(id, collection_reader);
            }
            WriteOperation::Collection(collection_id, coll_op) => {
                let collections = self.collections.read().await;

                COLLECTION_OPERATION_COUNTER
                    .create(CollectionOperationLabels {
                        collection: collection_id.0.to_string(),
                    })
                    .increment_by_one();

                let collection_reader = match collections.get(&collection_id) {
                    Some(collection_reader) => collection_reader,
                    None => {
                        error!(target: "Collection not found", ?collection_id);
                        return Err(anyhow::anyhow!("Collection not found"));
                    }
                };

                match coll_op {
                    CollectionWriteOperation::CreateField {
                        field_id,
                        field_name,
                        field,
                    } => {
                        collection_reader
                            .create_field(field_id, field_name, field)
                            .await
                            .context("Cannot create field")?;
                    }
                    CollectionWriteOperation::InsertDocument { doc_id, doc } => {
                        collection_reader
                            .insert_document(doc_id, doc)
                            .await
                            .context("cannot insert document")?;
                    }
                    CollectionWriteOperation::Index(doc_id, field_id, field_op) => match field_op {
                        DocumentFieldIndexOperation::IndexBoolean { value } => {
                            collection_reader
                                .index_boolean(doc_id, field_id, value)
                                .context("cannot index boolean")?;
                        }
                        DocumentFieldIndexOperation::IndexNumber { value } => {
                            collection_reader
                                .index_number(doc_id, field_id, value)
                                .context("cannot index number")?;
                        }
                        DocumentFieldIndexOperation::IndexString {
                            field_length,
                            terms,
                        } => {
                            collection_reader
                                .index_string(doc_id, field_id, field_length, terms)
                                .context("cannot index string")?;
                        }
                        DocumentFieldIndexOperation::IndexEmbedding { value } => {
                            collection_reader
                                .index_embedding(doc_id, field_id, value)
                                .context("cannot index embedding")?;
                        }
                    },
                }
            }
        };

        Ok(())
    }

    pub async fn get_collection<'s, 'coll>(
        &'s self,
        id: CollectionId,
    ) -> Option<CollectionReadLock<'coll>>
    where
        's: 'coll,
    {
        let r = self.collections.read().await;
        CollectionReadLock::try_new(r, id)
    }
}

pub struct CollectionReadLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionReader>>,
    id: CollectionId,
}

impl<'guard> CollectionReadLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionReader>>,
        id: CollectionId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(CollectionReadLock { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for CollectionReadLock<'_> {
    type Target = CollectionReader;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
}
