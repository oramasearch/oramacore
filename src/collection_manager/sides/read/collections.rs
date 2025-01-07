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
            read::CommitConfig,
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
use tracing::{debug, error, info, instrument};

use super::collection::CollectionReader;
use super::commit::CollectionDescriptorDump;

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {
    pub data_dir: PathBuf,
}

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

    #[instrument(skip(self))]
    pub fn load_from_disk(&mut self) -> Result<()> {
        info!(
            "Loading collections from disk '{:?}'.",
            self.indexes_config.data_dir
        );

        match std::fs::exists(&self.indexes_config.data_dir) {
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
                std::fs::create_dir_all(&self.indexes_config.data_dir)
                    .context("Cannot create data directory")?;
            }
        }

        let base_dir_for_collections = self.indexes_config.data_dir.join("collections");

        let files = std::fs::read_dir(&base_dir_for_collections).with_context(|| {
            format!(
                "Cannot read collections directory '{:?}'",
                base_dir_for_collections
            )
        })?;
        for f in files {
            let f = f.context("Cannot read file")?;

            let f_name = f
                .file_name()
                .into_string()
                .map_err(|e| anyhow::anyhow!("Cannot convert file name: {:?}", e))?;
            if !f_name.ends_with(".json") {
                continue;
            }

            let collection_path = f.path();
            let collection_metadata_file = std::fs::File::open(&collection_path)
                .with_context(|| format!("Cannot open file '{:?}'", f.path()))?;
            let collection_descriptor: CollectionDescriptorDump =
                serde_json::from_reader(collection_metadata_file).with_context(|| {
                    format!(
                        "Cannot deserialize collection descriptor from file '{:?}'",
                        collection_path
                    )
                })?;
        }

        info!("Collections loaded from disk.");
        todo!()
    }

    pub async fn commit(&self) -> Result<()> {
        match std::fs::exists(&self.indexes_config.data_dir) {
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
                std::fs::create_dir_all(&self.indexes_config.data_dir)
                    .context("Cannot create data directory")?;
            }
        };

        let col = self.collections.read().await;
        let col = &*col;

        let collections_dir = self.indexes_config.data_dir.join("collections");
        std::fs::create_dir_all(&collections_dir)
            .context("Cannot create 'collections' directory")?;

        for (id, reader) in col {
            info!("Committing collection {:?}", id);
            let desc = reader
                .get_collection_descriptor_dump()
                .context("Cannot create collection description dump")?;

            let coll_desc_file_path = collections_dir.join(format!("{}.json", id.0));
            let coll_desc_file =
                std::fs::File::create(&coll_desc_file_path).with_context(|| {
                    format!(
                        "Cannot create file for collection {:?} at {:?}",
                        id, coll_desc_file_path
                    )
                })?;
            serde_json::to_writer(coll_desc_file, &desc).with_context(|| {
                format!(
                    "Cannot serialize collection descriptor for {:?} to file {:?}",
                    id, coll_desc_file_path
                )
            })?;

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
