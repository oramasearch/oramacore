use std::{collections::HashMap, ops::Deref, path::PathBuf, sync::Arc};

use crate::{
    collection_manager::sides::{read::collection::CommitConfig, Offset},
    embeddings::EmbeddingService,
    file_utils::list_directory_in_path,
    nlp::NLPService,
    offset_storage::OffsetStorage,
    types::CollectionId,
};

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{debug, info, instrument};

use super::collection::CollectionReader;

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {
    pub data_dir: PathBuf,
}

#[derive(Debug)]
pub struct CollectionsReader {
    embedding_service: Arc<EmbeddingService>,
    nlp_service: Arc<NLPService>,
    collections: RwLock<HashMap<CollectionId, CollectionReader>>,
    indexes_config: IndexesConfig,

    offset_storage: OffsetStorage,
}
impl CollectionsReader {
    pub fn try_new(
        embedding_service: Arc<EmbeddingService>,
        nlp_service: Arc<NLPService>,
        indexes_config: IndexesConfig,
    ) -> Result<Self> {
        Ok(Self {
            embedding_service,
            nlp_service,

            collections: Default::default(),
            indexes_config,

            offset_storage: OffsetStorage::new(),
        })
    }

    pub fn get_embedding_service(&self) -> Arc<EmbeddingService> {
        self.embedding_service.clone()
    }

    pub(super) async fn get_collection<'s, 'coll>(
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
    pub(super) async fn load(&mut self) -> Result<()> {
        let data_dir = &self.indexes_config.data_dir;
        info!("Loading collections from disk '{:?}'.", data_dir);

        match std::fs::exists(data_dir) {
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
                std::fs::create_dir_all(data_dir).context("Cannot create data directory")?;
            }
        }

        let base_dir_for_collections = data_dir.join("collections");

        let collection_dirs = list_directory_in_path(&base_dir_for_collections)
            .context("Cannot read collection list from disk")?;

        let collection_dirs = match collection_dirs {
            Some(collection_dirs) => collection_dirs,
            None => {
                info!(
                    "No collections found in data directory {:?}. Skipping load.",
                    data_dir
                );
                return Ok(());
            }
        };

        for collection_dir in collection_dirs {
            info!("Loading collection {:?}", collection_dir);

            let file_name = collection_dir
                .file_name()
                .expect("File name is always given at this point");
            let file_name: String = file_name.to_string_lossy().into();

            let collection_id = CollectionId(file_name);

            let mut collection = CollectionReader::try_new(
                collection_id.clone(),
                self.embedding_service.clone(),
                self.nlp_service.clone(),
                self.indexes_config.clone(),
            )?;

            collection
                .load(base_dir_for_collections.join(&collection.id.0))
                .await
                .with_context(|| format!("Cannot load {:?} collection", collection_id))?;

            let mut guard = self.collections.write().await;
            guard.insert(collection_id, collection);
        }

        info!("Collections loaded from disk.");

        Ok(())
    }

    #[instrument(skip(self))]
    pub(super) async fn commit(&self) -> Result<()> {
        let data_dir = &self.indexes_config.data_dir;

        match std::fs::exists(data_dir) {
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
                std::fs::create_dir_all(data_dir).context("Cannot create data directory")?;
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

    pub(super) async fn create_collection(&self, offset: Offset, id: CollectionId) -> Result<()> {
        info!("Creating collection {:?}", id);

        let collection = CollectionReader::try_new(
            id.clone(),
            self.embedding_service.clone(),
            self.nlp_service.clone(),
            self.indexes_config.clone(),
        )?;

        let mut guard = self.collections.write().await;
        guard.insert(id, collection);

        self.offset_storage.set_offset(offset);

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
