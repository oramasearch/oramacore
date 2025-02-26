use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::Arc,
};

use crate::{
    ai::AIService,
    collection_manager::{
        dto::{ApiKey, LanguageDTO},
        sides::Offset,
    },
    file_utils::{
        create_if_not_exists, create_if_not_exists_async, create_or_overwrite, BufferedFile,
    },
    metrics::{commit::COMMIT_CALCULATION_TIME, CollectionCommitLabels},
    nlp::NLPService,
    types::CollectionId,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{info, instrument, warn};

use super::{collection::CollectionReader, IndexesConfig};

const LIMIT: usize = 100;

#[derive(Debug)]
pub struct CollectionsReader {
    ai_service: Arc<AIService>,
    nlp_service: Arc<NLPService>,
    collections: RwLock<HashMap<CollectionId, CollectionReader>>,
    indexes_config: IndexesConfig,
    last_reindexed_collections: RwLock<Vec<(CollectionId, CollectionId)>>,
}
impl CollectionsReader {
    pub async fn try_load(
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        indexes_config: IndexesConfig,
    ) -> Result<Self> {
        let data_dir = &indexes_config.data_dir;
        info!("Loading collections from disk '{:?}'.", data_dir);

        create_if_not_exists(data_dir).context("Cannot create data directory")?;

        let collections_info: CollectionsInfo = match BufferedFile::open(data_dir.join("info.json"))
            .and_then(|f| f.read_json_data())
            .context("Cannot deserialize info.json file")
        {
            Ok(info) => info,
            Err(e) => {
                warn!(
                    "Cannot read info.json file: {:?}. Skip loading collections",
                    e
                );
                return Ok(Self {
                    ai_service,
                    nlp_service,

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
            let collection_dir = base_dir_for_collections.join(&collection_id.0);
            info!("Loading collection {:?}", collection_dir);

            let collection =
                CollectionReader::try_load(ai_service.clone(), nlp_service.clone(), collection_dir)
                    .with_context(|| format!("Cannot load {:?} collection", collection_id))?;

            collections.insert(collection_id, collection);
        }

        info!("Collections loaded from disk.");

        Ok(Self {
            ai_service,
            nlp_service,
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
        CollectionReadLock::try_new(collections_lock, last_reindexed_collections_lock, id)
    }

    #[instrument(skip(self))]
    pub async fn commit(&self) -> Result<()> {
        info!("Committing collections");

        let data_dir = &self.indexes_config.data_dir;
        let collections_dir = data_dir.join("collections");

        let col = self.collections.read().await;
        let col = &*col;
        let collection_ids: Vec<_> = col.keys().cloned().collect();
        for (id, collection) in col {
            let collection_dir = collections_dir.join(&id.0);

            create_if_not_exists_async(&collection_dir)
                .await
                .with_context(|| format!("Cannot create directory for collection '{}'", id.0))?;

            let m = COMMIT_CALCULATION_TIME.create(CollectionCommitLabels {
                collection: id.0.clone(),
                side: "read",
            });
            collection
                .commit(collection_dir, false)
                .await
                .with_context(|| format!("Cannot commit collection {:?}", collection.get_id()))?;
            drop(m);
        }

        let guard = self.last_reindexed_collections.read().await;
        let collections_info = CollectionsInfo::V1(CollectionsInfoV1 {
            collection_ids: collection_ids.into_iter().collect(),
            last_reindexed_collections: guard.iter().cloned().collect(),
        });
        drop(guard);

        create_or_overwrite(data_dir.join("info.json"), &collections_info)
            .await
            .context("Cannot create info.json file")?;

        info!("Collections committed");

        Ok(())
    }

    pub async fn create_collection(
        &self,
        _offset: Offset,
        id: CollectionId,
        description: Option<String>,
        default_language: LanguageDTO,
        read_api_key: ApiKey,
    ) -> Result<()> {
        info!(collection_id=?id, "ReadSide: Creating collection {:?}", id);

        let collection = CollectionReader::empty(
            id.clone(),
            description,
            default_language,
            read_api_key,
            self.ai_service.clone(),
            self.nlp_service.clone(),
        );

        let mut guard = self.collections.write().await;
        if guard.contains_key(&id) {
            warn!(collection_id=?id, "Collection already exists");
            return Err(anyhow::anyhow!("Collection already exists"));
        }
        guard.insert(id.clone(), collection);
        drop(guard);

        info!(collection_id=?id, "Collection created {:?}", id);

        Ok(())
    }

    pub async fn substitute_collection(
        &self,
        _offset: Offset,
        target_collection_id: CollectionId,
        source_collection_id: CollectionId,
    ) -> Result<()> {
        info!(
            target_collection_id=?target_collection_id,
            source_collection_id=?source_collection_id,
            "ReadSide: Substitute collection {:?} with {:?}", target_collection_id, source_collection_id
        );

        let mut guard = self.collections.write().await;
        let mut source = guard
            .remove(&source_collection_id)
            .ok_or_else(|| anyhow::anyhow!("Source collection not found"))?;
        source.id = target_collection_id.clone();
        // ignore return value
        let _ = guard.remove(&target_collection_id);
        guard.insert(target_collection_id.clone(), source);
        drop(guard);

        let mut guard = self.last_reindexed_collections.write().await;
        guard.push((source_collection_id.clone(), target_collection_id.clone()));
        if guard.len() > LIMIT {
            guard.remove(0);
        }
        drop(guard);

        let data_dir = &self.indexes_config.data_dir;
        let collections_dir = data_dir.join("collections");

        let collection_dir = collections_dir.join(&target_collection_id.0);
        create_if_not_exists_async(&collection_dir)
            .await
            .with_context(|| {
                format!(
                    "Cannot create directory for collection '{}'",
                    target_collection_id.0
                )
            })?;

        let m = COMMIT_CALCULATION_TIME.create(CollectionCommitLabels {
            collection: target_collection_id.0.clone(),
            side: "read",
        });
        let collection = self
            .get_collection(target_collection_id.clone())
            .await
            .context("Cannot get collection")?;
        collection
            .commit(collection_dir, true)
            .await
            .with_context(|| format!("Cannot commit collection {:?}", collection.get_id()))?;
        drop(m);

        info!(
            target_collection_id=?target_collection_id,
            source_collection_id=?source_collection_id,
            "Collection substituted {:?} with {:?}", target_collection_id, source_collection_id
        );

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
                                    id: id.clone(),
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
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "version")]
enum CollectionsInfo {
    #[serde(rename = "1")]
    V1(CollectionsInfoV1),
}
#[derive(Deserialize, Serialize)]
struct CollectionsInfoV1 {
    collection_ids: HashSet<CollectionId>,
    last_reindexed_collections: Vec<(CollectionId, CollectionId)>,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use http::uri::Scheme;
    use redact::Secret;

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
        }));
        let nlp_service = Arc::new(NLPService::new());
        let collections = CollectionsReader::try_load(
            ai_service,
            nlp_service,
            IndexesConfig {
                commit_interval: Duration::from_secs(1_000),
                data_dir: generate_new_path(),
                insert_batch_commit_size: 100_000,
            },
        )
        .await
        .unwrap();

        let target_collection_id = CollectionId("target".to_string());
        let tmp_collection_id = CollectionId("tmp".to_string());

        // Target collection
        collections
            .create_collection(
                Offset(1),
                target_collection_id.clone(),
                None,
                LanguageDTO::English,
                ApiKey(Secret::from("read".to_string())),
            )
            .await
            .unwrap();

        // TMP collection
        collections
            .create_collection(
                Offset(1),
                tmp_collection_id.clone(),
                None,
                LanguageDTO::English,
                ApiKey(Secret::from("read".to_string())),
            )
            .await
            .unwrap();

        // Substitute target collection with tmp collection
        collections
            .substitute_collection(
                Offset(2),
                target_collection_id.clone(),
                tmp_collection_id.clone(),
            )
            .await
            .unwrap();

        let collection = collections
            .get_collection(target_collection_id.clone())
            .await
            .unwrap();
        let collection = &*collection;
        assert_eq!(collection.get_id(), target_collection_id);

        let collection = collections
            .get_collection(tmp_collection_id.clone())
            .await
            .unwrap();
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
        }));
        let nlp_service = Arc::new(NLPService::new());
        let collections = CollectionsReader::try_load(
            ai_service,
            nlp_service,
            IndexesConfig {
                commit_interval: Duration::from_secs(1_000),
                data_dir: generate_new_path(),
                insert_batch_commit_size: 100_000,
            },
        )
        .await
        .unwrap();

        let couples: Vec<_> = (0..(LIMIT + 1))
            .map(|i| {
                let target_collection_id = CollectionId(format!("target-{}", i));
                let tmp_collection_id = CollectionId(format!("tmp-{}", i));
                (target_collection_id, tmp_collection_id)
            })
            .collect();

        for (target_collection_id, tmp_collection_id) in &couples {
            // Target collection
            collections
                .create_collection(
                    Offset(1),
                    target_collection_id.clone(),
                    None,
                    LanguageDTO::English,
                    ApiKey(Secret::from("read".to_string())),
                )
                .await
                .unwrap();

            // TMP collection
            collections
                .create_collection(
                    Offset(1),
                    tmp_collection_id.clone(),
                    None,
                    LanguageDTO::English,
                    ApiKey(Secret::from("read".to_string())),
                )
                .await
                .unwrap();

            // Substitute target collection with tmp collection
            collections
                .substitute_collection(
                    Offset(2),
                    target_collection_id.clone(),
                    tmp_collection_id.clone(),
                )
                .await
                .unwrap();

            let collection = collections
                .get_collection(target_collection_id.clone())
                .await
                .unwrap();
            let collection = &*collection;
            assert_eq!(collection.get_id(), target_collection_id.clone());

            let collection = collections
                .get_collection(tmp_collection_id.clone())
                .await
                .unwrap();
            let collection = &*collection;
            assert_eq!(collection.get_id(), target_collection_id.clone());
        }

        // All collections are there
        for (target, _) in &couples {
            let collection = collections.get_collection(target.clone()).await;
            assert!(collection.is_some());
        }

        let collection = collections.get_collection(couples[0].0.clone()).await;
        assert!(collection.is_some());
        // But the first one tmp collection is not there
        let collection = collections.get_collection(couples[0].1.clone()).await;
        assert!(collection.is_none());
    }
}
