use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::Arc,
};

use crate::{
    ai::AIService,
    collection_manager::sides::{read::context::ReadSideContext, Offset},
    lock::{OramaAsyncLock, OramaAsyncLockReadGuard},
    metrics::{commit::COMMIT_CALCULATION_TIME, Empty},
    types::{ApiKey, CollectionId},
};

use oramacore_lib::fs::{create_if_not_exists, create_if_not_exists_async, BufferedFile};

use oramacore_lib::nlp::locales::Locale;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{error, info, instrument, warn};

use super::{collection::CollectionReader, IndexesConfig};

/// Lifecicle of a collection
///
/// # Collection creation
/// When a collection is created, it is added the the `collections` map.
///
/// # Collection deletion
/// When a collection is deleted, it is marked as deleted.
/// The collection still exists in the `collections` map.
/// NB: `get_collection` will return `None`.
///
pub struct CollectionsReader {
    context: ReadSideContext,

    collections: OramaAsyncLock<HashMap<CollectionId, CollectionReader>>,
    indexes_config: IndexesConfig,
    last_reindexed_collections: OramaAsyncLock<Vec<(CollectionId, CollectionId)>>,
}

impl CollectionsReader {
    pub async fn try_load(context: ReadSideContext, indexes_config: IndexesConfig) -> Result<Self> {
        let data_dir = &indexes_config.data_dir;
        info!("Loading collections from disk '{:?}'.", data_dir);

        create_if_not_exists(data_dir).context("Cannot create data directory")?;

        let collections_info: CollectionsInfo =
            match BufferedFile::open(data_dir.join("info.json")).and_then(|f| f.read_json_data()) {
                Ok(info) => info,
                Err(_) => {
                    warn!("Cannot read info.json file. Skip loading collections",);
                    return Ok(Self {
                        context,
                        collections: OramaAsyncLock::new("collections", Default::default()),
                        indexes_config,
                        last_reindexed_collections: OramaAsyncLock::new(
                            "last_reindexed_collections",
                            Default::default(),
                        ),
                    });
                }
            };

        let CollectionsInfo::V1(collections_info) = collections_info;

        let base_dir_for_collections = data_dir.join("collections");
        let mut collections: HashMap<CollectionId, CollectionReader> = HashMap::with_capacity(
            collections_info.collection_ids.len() - collections_info.deleted_collection_ids.len(),
        );
        for collection_id in collections_info.collection_ids {
            if collections_info
                .deleted_collection_ids
                .contains(&collection_id)
            {
                info!("Collection {:?} is deleted. Skip loading", collection_id);
                continue;
            }
            let collection_dir = base_dir_for_collections.join(collection_id.as_str());
            info!("Loading collection {:?}", collection_dir);

            let collection = CollectionReader::try_load(
                context.clone(),
                collection_dir.clone(),
                indexes_config.offload_field,
            )
            .with_context(|| format!("Cannot load {collection_id:?} collection"))?;

            info!("Collection {:?} loaded", collection_dir);

            collections.insert(collection_id, collection);
        }

        info!("Collections loaded from disk.");

        Ok(Self {
            context,

            collections: OramaAsyncLock::new("collections", collections),
            indexes_config,
            last_reindexed_collections: OramaAsyncLock::new(
                "last_reindexed_collections",
                collections_info
                    .last_reindexed_collections
                    .into_iter()
                    .collect(),
            ),
        })
    }

    pub fn get_ai_service(&self) -> Arc<AIService> {
        self.context.ai_service.clone()
    }

    pub async fn get_collection<'s, 'coll>(
        &'s self,
        id: CollectionId,
    ) -> Option<CollectionReadLock<'coll>>
    where
        's: 'coll,
    {
        let collections_lock = self.collections.read("get_collection").await;
        let last_reindexed_collections_lock =
            self.last_reindexed_collections.read("get_collection").await;

        if let Some(collection) = collections_lock.get(&id) {
            if collection.is_deleted() {
                return None;
            }
        }

        CollectionReadLock::try_new(collections_lock, last_reindexed_collections_lock, id)
    }

    #[instrument(skip(self, offset))]
    pub async fn commit(&self, offset: Offset) -> Result<()> {
        let data_dir = &self.indexes_config.data_dir;
        let collections_dir = data_dir.join("collections");

        let col = self.collections.read("commit").await;

        let mut collection_ids: Vec<_> = vec![];
        info!("Committing collections: {:?}", collection_ids);
        for (id, collection) in col.iter() {
            let collection_dir = collections_dir.join(id.as_str());

            if collection.is_deleted() {
                // TODO: should we delete the collection from the disk?
                continue;
            }

            collection_ids.push(*id);

            create_if_not_exists_async(&collection_dir)
                .await
                .with_context(|| {
                    format!("Cannot create directory for collection '{}'", id.as_str())
                })?;

            let m = COMMIT_CALCULATION_TIME.create(Empty);

            match collection.commit(offset).await {
                Ok(_) => {}
                Err(error) => {
                    error!(error = ?error, collection_id=?id, "Cannot commit collection {:?}: {:?}", id, error);
                }
            }

            drop(m);
        }

        let guard = self.last_reindexed_collections.read("commit").await;
        let collections_info = CollectionsInfo::V1(CollectionsInfoV1 {
            collection_ids: collection_ids.into_iter().collect(),
            deleted_collection_ids: Default::default(),
            last_reindexed_collections: guard.iter().cloned().collect(),
        });
        drop(guard);

        BufferedFile::create_or_overwrite(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&collections_info)
            .context("Cannot write info.json file")?;

        info!("Collections committed");

        Ok(())
    }

    pub async fn create_collection(
        &self,
        id: CollectionId,
        description: Option<String>,
        mcp_description: Option<String>,
        default_locale: Locale,
        read_api_key: ApiKey,
        write_api_key: Option<ApiKey>,
    ) -> Result<()> {
        info!(collection_id=?id, "ReadSide: Creating collection {:?}", id);

        let collection_dir = self
            .indexes_config
            .data_dir
            .join("collections")
            .join(id.as_str());

        let collection = CollectionReader::empty(
            collection_dir,
            id,
            description,
            mcp_description,
            default_locale,
            read_api_key,
            write_api_key,
            self.context.clone(),
            self.indexes_config.offload_field,
        )?;

        let mut guard = self.collections.write("create_collection").await;

        if let Some(collection) = guard.get(&id) {
            if !collection.is_deleted() {
                warn!(collection_id=?id, "Collection already exists");
                return Err(anyhow::anyhow!("Collection already exists"));
            }

            // If the collection is previously deleted, we ignore the previous data
            // and clean the FS
            self.clean_fs_for_collection(collection).await.ok();
        }

        guard.insert(id, collection);
        drop(guard);

        info!(collection_id=?id, "Collection created {:?}", id);

        Ok(())
    }

    /*
    pub async fn remove_collection(&self, id: CollectionId) -> Result<()> {
        let mut guard = self.collections.write().await;
        if let Some(collection) = guard.get_mut(&id) {
            collection.mark_as_deleted();
            info!(collection_id=?id, "Collection marked as deleted {:?}", id);
        }

        Ok(())
    }
    */

    pub async fn clean_fs_for_collection(&self, collection: &CollectionReader) -> Result<()> {
        info!(collection_id=?collection.id(), "Cleaning FS collection");

        let data_dir = &self.indexes_config.data_dir;
        let collections_dir = data_dir.join("collections");

        let collection_dir = collections_dir.join(collection.id().as_str());
        if collection_dir.exists() {
            tokio::fs::remove_dir_all(&collection_dir)
                .await
                .with_context(|| {
                    format!(
                        "Cannot remove directory for collection '{}'",
                        collection.id().as_str()
                    )
                })?;
        }

        info!("FS cleaned");

        Ok(())
    }

    pub async fn delete_collection(&self, collection_id: CollectionId) -> Result<()> {
        info!(collection_id=?collection_id, "ReadSide: Deleting collection {:?}", collection_id);

        let mut guard = self.collections.write("delete_collection").await;
        if let Some(collection) = guard.get_mut(&collection_id) {
            collection.mark_as_deleted();
            info!(collection_id=?collection_id, "Collection marked as deleted {:?}", collection_id);
        }

        Ok(())
    }
}

pub struct CollectionReadLock<'guard> {
    lock: OramaAsyncLockReadGuard<'guard, HashMap<CollectionId, CollectionReader>>,
    pub id: CollectionId,
}

impl<'guard> CollectionReadLock<'guard> {
    pub fn try_new(
        collections_lock: OramaAsyncLockReadGuard<'guard, HashMap<CollectionId, CollectionReader>>,
        last_reindexed_collections_lock: OramaAsyncLockReadGuard<
            'guard,
            Vec<(CollectionId, CollectionId)>,
        >,
        id: CollectionId,
    ) -> Option<Self> {
        let guard = collections_lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(CollectionReadLock {
                    lock: collections_lock,
                    id,
                })
            }
            None => {
                let target_collection_id = last_reindexed_collections_lock
                    .iter()
                    .find(|(source, _)| source == &id)
                    .map(|(_, target)| target);
                match target_collection_id {
                    None => None,
                    Some(id) => {
                        let guard = collections_lock.get(id);
                        match &guard {
                            Some(_) => {
                                let _ = guard;
                                Some(CollectionReadLock {
                                    lock: collections_lock,
                                    id: *id,
                                })
                            }
                            None => None,
                        }
                    }
                }
            }
        }
    }
}

impl Deref for CollectionReadLock<'_> {
    type Target = CollectionReader;

    fn deref(&self) -> &Self::Target {
        // Safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).expect("THe colleciton alwaus contains id because we changed it before and we hold a read lock")
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "version")]
enum CollectionsInfo {
    #[serde(rename = "1")]
    V1(CollectionsInfoV1),
}
#[derive(Debug, Deserialize, Serialize)]
struct CollectionsInfoV1 {
    collection_ids: HashSet<CollectionId>,
    #[serde(default)]
    deleted_collection_ids: HashSet<CollectionId>,
    last_reindexed_collections: Vec<(CollectionId, CollectionId)>,
}
