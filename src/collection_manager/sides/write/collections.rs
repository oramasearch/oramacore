use std::collections::HashMap;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Ok, Result};
use serde::Deserialize;
use tokio::sync::{broadcast::Sender, RwLock, RwLockReadGuard};
use tracing::{info, instrument};

use crate::collection_manager::sides::hooks::HooksRuntime;
use crate::{
    collection_manager::dto::CollectionDTO, file_utils::list_directory_in_path, types::CollectionId,
};

use crate::collection_manager::dto::{CreateCollectionOptionDTO, LanguageDTO};

use super::{collection::CollectionWriter, embedding::EmbeddingCalculationRequest, WriteOperation};

pub struct CollectionsWriter {
    collections: RwLock<HashMap<CollectionId, CollectionWriter>>,
    config: CollectionsWriterConfig,
    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CollectionsWriterConfig {
    pub data_dir: PathBuf,
    #[serde(default = "embedding_queue_limit_default")]
    pub embedding_queue_limit: usize,
}

fn embedding_queue_limit_default() -> usize {
    50
}

impl CollectionsWriter {
    pub fn new(
        config: CollectionsWriterConfig,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
    ) -> CollectionsWriter {
        CollectionsWriter {
            collections: Default::default(),
            config,
            embedding_sender,
        }
    }

    pub async fn get_collection<'s, 'coll>(
        &'s self,
        id: CollectionId,
    ) -> Option<CollectionWriteLock<'coll>>
    where
        's: 'coll,
    {
        let r = self.collections.read().await;
        CollectionWriteLock::try_new(r, id)
    }

    pub async fn create_collection(
        &self,
        collection_option: CreateCollectionOptionDTO,
        sender: Sender<WriteOperation>,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<()> {
        let CreateCollectionOptionDTO {
            id,
            description,
            language,
            typed_fields,
        } = collection_option;

        info!("Creating collection {:?}", id);

        let collection = CollectionWriter::new(
            id.clone(),
            description,
            language.unwrap_or(LanguageDTO::English),
            self.embedding_sender.clone(),
        );

        sender
            .send(WriteOperation::CreateCollection { id: id.clone() })
            .context("Cannot send create collection")?;

        collection
            .register_fields(typed_fields, sender.clone(), hooks_runtime)
            .await
            .context("Cannot register fields")?;

        let mut collections = self.collections.write().await;
        if collections.contains_key(&id) {
            // This error should be typed.
            // todo: create a custom error type
            return Err(anyhow!(format!("Collection \"{}\" already exists", id.0)));
        }
        collections.insert(id, collection);
        drop(collections);

        Ok(())
    }

    pub async fn list(&self) -> Vec<CollectionDTO> {
        let collections = self.collections.read().await;

        collections.iter().map(|(_, coll)| coll.as_dto()).collect()
    }

    pub async fn commit(&self) -> Result<()> {
        let data_dir = &self.config.data_dir;

        // This `write lock` will not change the content of the collections
        // But it is requered to ensure that the collections are not being modified
        // while we are saving them to disk
        let mut collections = self.collections.write().await;

        std::fs::create_dir_all(data_dir).context("Cannot create data directory")?;

        for (collection_id, collection) in collections.iter_mut() {
            let collection_dir = data_dir.join(collection_id.0.clone());
            collection.commit(collection_dir)?;
        }

        // Now it is safe to drop the lock
        // because we safe everything to disk
        drop(collections);

        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn load(&mut self, hooks_runtime: Arc<HooksRuntime>) -> Result<()> {
        // `&mut self` isn't needed here
        // but we need to ensure that the method is not called concurrently
        let data_dir = &self.config.data_dir;

        info!("Loading collections from disk from {:?}", data_dir);

        let collection_dirs =
            list_directory_in_path(data_dir).context("Cannot read collection list from disk")?;

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
            let file_name = collection_dir
                .file_name()
                .expect("File name is always given at this point");
            let file_name: String = file_name.to_string_lossy().into();

            let collection_id = CollectionId(file_name);

            let mut collection = CollectionWriter::new(
                collection_id.clone(),
                None,
                LanguageDTO::English,
                self.embedding_sender.clone(),
            );
            collection
                .load(collection_dir, hooks_runtime.clone())
                .await?;

            self.collections
                .write()
                .await
                .insert(collection_id, collection);
        }

        Ok(())
    }
}

pub struct CollectionWriteLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
    id: CollectionId,
}

impl<'guard> CollectionWriteLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
        id: CollectionId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(CollectionWriteLock { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for CollectionWriteLock<'_> {
    type Target = CollectionWriter;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
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
