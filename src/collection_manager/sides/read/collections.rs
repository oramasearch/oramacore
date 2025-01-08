use std::{collections::HashMap, ops::Deref, path::PathBuf, sync::Arc};

use crate::{
    collection_manager::sides::{
        document_storage::DocumentStorage, read::collection::CommitConfig,
        CollectionWriteOperation, DocumentFieldIndexOperation, GenericWriteOperation,
        WriteOperation,
    },
    embeddings::EmbeddingService,
    metrics::{
        CollectionAddedLabels, CollectionOperationLabels, COLLECTION_ADDED_COUNTER,
        COLLECTION_OPERATION_COUNTER,
    },
    types::CollectionId,
};

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{debug, error, info, instrument, warn};

use super::collection::CollectionReader;

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {}

#[derive(Debug)]
pub struct CollectionsReader {
    embedding_service: Arc<EmbeddingService>,
    collections: RwLock<HashMap<CollectionId, CollectionReader>>,
    document_storage: Arc<dyn DocumentStorage>,
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

    #[instrument(skip(self, data_dir))]
    pub async fn load_from_disk(&mut self, data_dir: PathBuf) -> Result<()> {
        info!("Loading collections from disk '{:?}'.", data_dir);

        match std::fs::exists(&data_dir) {
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Error while checking if the data directory exists: {:?}",
                    e
                ));
            }
            Ok(true) => {
                debug!("Data directory exists.");
            }
            Ok(false) => {
                info!("Data directory does not exist. Creating it.");
                std::fs::create_dir_all(&data_dir).context("Cannot create data directory")?;
            }
        }

        let base_dir_for_collections = data_dir.join("collections");

        let files = std::fs::read_dir(&base_dir_for_collections).with_context(|| {
            format!(
                "Cannot read collections directory {:?}",
                base_dir_for_collections
            )
        })?;
        for f in files {
            let f = f.context("Cannot read file")?;

            let metadata = f.metadata().context("Cannot get file metadata")?;

            if !metadata.is_dir() {
                warn!("File {:?} is not a directory. Skipping.", f.path());
                continue;
            }

            info!("Loading collection {:?}", f.path());

            let collection_id = CollectionId(f.file_name().to_string_lossy().to_string());

            let mut collection = CollectionReader::try_new(
                collection_id.clone(),
                self.embedding_service.clone(),
                self.document_storage.clone(),
                self.indexes_config.clone(),
            )?;

            collection.load(base_dir_for_collections.join(&collection.id.0))?;

            let mut guard = self.collections.write().await;
            guard.insert(collection_id, collection);
        }

        info!("Collections loaded from disk.");

        Ok(())
    }

    #[instrument(skip(self, data_dir))]
    pub async fn commit(&self, data_dir: PathBuf) -> Result<()> {
        match std::fs::exists(&data_dir) {
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Error while checking if the data directory exists: {:?}",
                    e
                ));
            }
            Ok(true) => {
                debug!("Data directory exists.");
            }
            Ok(false) => {
                info!("Data directory does not exist. Creating it.");
                std::fs::create_dir_all(&data_dir).context("Cannot create data directory")?;
            }
        };

        let col = self.collections.read().await;
        let col = &*col;

        let collections_dir = data_dir.join("collections");
        std::fs::create_dir_all(&collections_dir)
            .context("Cannot create 'collections' directory")?;

        for (id, reader) in col {
            info!("Committing collection {:?}", id);

            reader.commit(CommitConfig {
                folder_to_commit: collections_dir.join(&id.0),
                epoch: 0,
            })?;
        }

        Ok(())
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
