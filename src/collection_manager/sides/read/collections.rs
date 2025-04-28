use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::Arc,
};

use crate::{
    ai::{llms::LLMService, AIService},
    collection_manager::sides::{read::notify::Notifier, Offset},
    file_utils::{create_if_not_exists, create_if_not_exists_async, BufferedFile},
    metrics::{commit::COMMIT_CALCULATION_TIME, Empty},
    nlp::{locales::Locale, NLPService},
    types::{ApiKey, CollectionId, DocumentId, SearchOffset},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{error, info, instrument, warn};

use super::{collection::CollectionReader, IndexesConfig};

const LIMIT: usize = 100;

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
/// # Collection substitution
/// When a collection is substituted, the old collection is removed from the `collections` map.
/// The old collection id is added to the `last_reindexed_collections`
///

pub struct CollectionsReader {
    ai_service: Arc<AIService>,
    nlp_service: Arc<NLPService>,
    llm_service: Arc<LLMService>,
    notifier: Option<Arc<Notifier>>,

    collections: RwLock<HashMap<CollectionId, CollectionReader>>,
    indexes_config: IndexesConfig,
    last_reindexed_collections: RwLock<Vec<(CollectionId, CollectionId)>>,
}

impl CollectionsReader {
    pub async fn try_load(
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        indexes_config: IndexesConfig,
    ) -> Result<Self> {
        let data_dir = &indexes_config.data_dir;
        info!("Loading collections from disk '{:?}'.", data_dir);

        create_if_not_exists(data_dir).context("Cannot create data directory")?;

        let mut notifier = None;
        if let Some(notifier_config) = &indexes_config.notifier {
            let n = Notifier::try_new(notifier_config).context("Cannot create notifier")?;
            notifier = Some(n);
        }

        let collections_info: CollectionsInfo =
            match BufferedFile::open(data_dir.join("info.json")).and_then(|f| f.read_json_data()) {
                Ok(info) => info,
                Err(_) => {
                    warn!("Cannot read info.json file. Skip loading collections",);
                    return Ok(Self {
                        ai_service,
                        nlp_service,
                        llm_service,
                        notifier,

                        collections: Default::default(),
                        indexes_config,
                        last_reindexed_collections: Default::default(),
                    });
                }
            };

        let CollectionsInfo::V1(collections_info) = collections_info;

        let base_dir_for_collections = data_dir.join("collections");
        let mut collections: HashMap<CollectionId, CollectionReader> = Default::default();
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
                ai_service.clone(),
                nlp_service.clone(),
                llm_service.clone(),
                notifier.clone(),
                collection_dir,
            )
            .with_context(|| format!("Cannot load {:?} collection", collection_id))?;

            collections.insert(collection_id, collection);
        }

        info!("Collections loaded from disk.");

        Ok(Self {
            ai_service,
            nlp_service,
            llm_service,
            notifier,

            collections: RwLock::new(collections),
            indexes_config,
            last_reindexed_collections: RwLock::new(
                collections_info
                    .last_reindexed_collections
                    .into_iter()
                    .collect(),
            ),
        })
    }

    pub fn get_ai_service(&self) -> Arc<AIService> {
        self.ai_service.clone()
    }

    pub async fn get_collection<'s, 'coll>(
        &'s self,
        id: CollectionId,
    ) -> Option<CollectionReadLock<'coll>>
    where
        's: 'coll,
    {
        let collections_lock = self.collections.read().await;
        let last_reindexed_collections_lock = self.last_reindexed_collections.read().await;

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

        let col = self.collections.read().await;

        let collection_ids: Vec<_> = col.keys().cloned().collect();
        info!("Committing collections: {:?}", collection_ids);
        for (id, collection) in col.iter() {
            let collection_dir = collections_dir.join(id.as_str());

            if collection.is_deleted() {
                // TODO: should we delete the collection from the disk?
                continue;
            }

            create_if_not_exists_async(&collection_dir)
                .await
                .with_context(|| {
                    format!("Cannot create directory for collection '{}'", id.as_str())
                })?;

            let m = COMMIT_CALCULATION_TIME.create(Empty);

            match collection.commit(collection_dir, offset).await {
                Ok(_) => {}
                Err(error) => {
                    error!(error = ?error, collection_id=?id, "Cannot commit collection {:?}: {:?}", id, error);
                }
            }

            drop(m);
        }

        let guard = self.last_reindexed_collections.read().await;
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
        default_locale: Locale,
        read_api_key: ApiKey,
    ) -> Result<()> {
        info!(collection_id=?id, "ReadSide: Creating collection {:?}", id);

        let collection = CollectionReader::empty(
            id,
            description,
            default_locale,
            read_api_key,
            self.ai_service.clone(),
            self.nlp_service.clone(),
            self.llm_service.clone(),
            self.notifier.clone(),
        );

        let mut guard = self.collections.write().await;

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

    pub async fn remove_collection(&self, id: CollectionId) -> Result<()> {
        let mut guard = self.collections.write().await;
        if let Some(collection) = guard.get_mut(&id) {
            collection.mark_as_deleted();
            info!(collection_id=?id, "Collection marked as deleted {:?}", id);
        }

        Ok(())
    }

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

    /*
    pub async fn substitute_collection(
        &self,
        _offset: SearchOffset,
        target_collection_id: CollectionId,
        source_collection_id: CollectionId,
        reference: Option<String>,
    ) -> Result<()> {
        info!(
            target_collection_id=?target_collection_id,
            source_collection_id=?source_collection_id,
            "ReadSide: Substitute collection {:?} with {:?}", target_collection_id, source_collection_id
        );

        let mut guard = self.collections.write().await;
        let source = guard
            .remove(&source_collection_id)
            .ok_or_else(|| anyhow::anyhow!("Source collection not found"))?;
        // source.id = target_collection_id;
        // Ignore return value: we don't care if the target collection was there or not.
        let _ = guard.remove(&target_collection_id);
        guard.insert(target_collection_id, source);
        drop(guard);

        let mut guard = self.last_reindexed_collections.write().await;
        guard.push((source_collection_id, target_collection_id));
        if guard.len() > LIMIT {
            guard.remove(0);
        }
        drop(guard);

        let data_dir = &self.indexes_config.data_dir;
        let collections_dir = data_dir.join("collections");

        let collection_dir = collections_dir.join(target_collection_id.as_str());
        create_if_not_exists_async(&collection_dir)
            .await
            .with_context(|| {
                format!(
                    "Cannot create directory for collection '{}'",
                    target_collection_id.as_str()
                )
            })?;

        let m = COMMIT_CALCULATION_TIME.create(Empty);
        let collection = self
            .get_collection(target_collection_id)
            .await
            .context("Cannot get collection")?;
        /*
        collection
            .commit(collection_dir, true)
            .await
            .with_context(|| format!("Cannot commit collection {:?}", collection.get_id()))?;
        */
        drop(m);

        if let Some(notifier) = &self.notifier {
            if let Err(error) = notifier
                .notify_collection_substitution(
                    target_collection_id,
                    source_collection_id,
                    reference,
                )
                .await
                .context("Cannot notify collection substitution")
            {
                error!(
                    error = ?error,
                    target_collection_id=?target_collection_id,
                    source_collection_id=?source_collection_id,
                    "Cannot notify collection substitution. Skip it"
                );
            };
        }

        info!(
            target_collection_id=?target_collection_id,
            source_collection_id=?source_collection_id,
            "Collection substituted {:?} with {:?}", target_collection_id, source_collection_id
        );

        Ok(())
    }
    */
    pub async fn delete_collection(&self, collection_id: CollectionId) -> Result<()> {
        info!(collection_id=?collection_id, "ReadSide: Deleting collection {:?}", collection_id);

        let mut guard = self.collections.write().await;
        if let Some(collection) = guard.get_mut(&collection_id) {
            collection.mark_as_deleted();
            info!(collection_id=?collection_id, "Collection marked as deleted {:?}", collection_id);
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
        collections_lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionReader>>,
        last_reindexed_collections_lock: RwLockReadGuard<'guard, Vec<(CollectionId, CollectionId)>>,
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

/*

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use http::uri::Scheme;

    use crate::{ai::AIServiceConfig, tests::utils::generate_new_path};

    use super::*;

    #[tokio::test]
    async fn test_last_reindex_collections_simple() {
        let ai_service = Arc::new(AIService::new(AIServiceConfig {
            api_key: None,
            host: "127.0.0.0".parse().unwrap(),
            scheme: Scheme::HTTP,
            port: 80,
            max_connections: 0,
            llm: crate::ai::AIServiceLLMConfig {
                host: "localhost".to_string(),
                port: 8000,
                model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
            },
            remote_llms: None,
        }));

        let llm_service = Arc::new(
            LLMService::try_new(
                crate::ai::AIServiceLLMConfig {
                    host: "localhost".to_string(),
                    port: 8000,
                    model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
                },
                None,
            )
            .unwrap(),
        );

        let nlp_service = Arc::new(NLPService::new());
        let collections = CollectionsReader::try_load(
            ai_service,
            nlp_service,
            llm_service,
            IndexesConfig {
                commit_interval: Duration::from_secs(1_000),
                data_dir: generate_new_path(),
                insert_batch_commit_size: 100_000,
                notifier: None,
            },
        )
        .await
        .unwrap();

        let target_collection_id = CollectionId::from("target".to_string());
        let tmp_collection_id = CollectionId::from("tmp".to_string());

        // Target collection
        collections
            .create_collection(
                Offset(1),
                target_collection_id,
                None,
                Locale::EN,
                ApiKey::try_from("read").unwrap(),
            )
            .await
            .unwrap();

        // TMP collection
        collections
            .create_collection(
                Offset(1),
                tmp_collection_id,
                None,
                Locale::EN,
                ApiKey::try_from("read").unwrap(),
            )
            .await
            .unwrap();

        // Substitute target collection with tmp collection
        collections
            .substitute_collection(Offset(2), target_collection_id, tmp_collection_id, None)
            .await
            .unwrap();

        let collection = collections
            .get_collection(target_collection_id)
            .await
            .unwrap();
        let collection = &*collection;
        assert_eq!(collection.get_id(), target_collection_id);

        let collection = collections.get_collection(tmp_collection_id).await.unwrap();
        let collection = &*collection;
        assert_eq!(collection.get_id(), target_collection_id);
    }

    #[tokio::test]
    async fn test_last_reindex_collections_limit() {
        let ai_service = Arc::new(AIService::new(AIServiceConfig {
            api_key: None,
            host: "127.0.0.0".parse().unwrap(),
            scheme: Scheme::HTTP,
            port: 80,
            max_connections: 0,
            llm: crate::ai::AIServiceLLMConfig {
                host: "localhost".to_string(),
                port: 8000,
                model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
            },
            remote_llms: None,
        }));
        let nlp_service = Arc::new(NLPService::new());
        let llm_service = Arc::new(
            LLMService::try_new(
                crate::ai::AIServiceLLMConfig {
                    host: "localhost".to_string(),
                    port: 8000,
                    model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
                },
                None,
            )
            .unwrap(),
        );
        let collections = CollectionsReader::try_load(
            ai_service,
            nlp_service,
            llm_service,
            IndexesConfig {
                commit_interval: Duration::from_secs(1_000),
                data_dir: generate_new_path(),
                insert_batch_commit_size: 100_000,
                notifier: None,
            },
        )
        .await
        .unwrap();

        let couples: Vec<_> = (0..(LIMIT + 1))
            .map(|i| {
                let target_collection_id = CollectionId::from(format!("target-{}", i));
                let tmp_collection_id = CollectionId::from(format!("tmp-{}", i));
                (target_collection_id, tmp_collection_id)
            })
            .collect();

        for (target_collection_id, tmp_collection_id) in &couples {
            // Target collection
            collections
                .create_collection(
                    Offset(1),
                    *target_collection_id,
                    None,
                    Locale::EN,
                    ApiKey::try_from("read").unwrap(),
                )
                .await
                .unwrap();

            // TMP collection
            collections
                .create_collection(
                    Offset(1),
                    *tmp_collection_id,
                    None,
                    Locale::EN,
                    ApiKey::try_from("read").unwrap(),
                )
                .await
                .unwrap();

            // Substitute target collection with tmp collection
            collections
                .substitute_collection(Offset(2), *target_collection_id, *tmp_collection_id, None)
                .await
                .unwrap();

            let collection = collections
                .get_collection(*target_collection_id)
                .await
                .unwrap();
            let collection = &*collection;
            assert_eq!(collection.get_id(), target_collection_id.clone());

            let collection = collections
                .get_collection(*tmp_collection_id)
                .await
                .unwrap();
            let collection = &*collection;
            assert_eq!(collection.get_id(), target_collection_id.clone());
        }

        // All collections are there
        for (target, _) in &couples {
            let collection = collections.get_collection(*target).await;
            assert!(collection.is_some());
        }

        let collection = collections.get_collection(couples[0].0).await;
        assert!(collection.is_some());
        // But the first one tmp collection is not there
        let collection = collections.get_collection(couples[0].1).await;
        assert!(collection.is_none());
    }
}

*/
