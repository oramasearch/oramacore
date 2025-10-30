use crate::lock::OramaAsyncLock;
use crate::types::{CollectionId, IndexId};
use anyhow::{Context, Result};
use oramacore_lib::fs::create_if_not_exists;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::s3::{self, S3Fetcher};

struct DatasourceWrapper {
    in_use: AtomicBool,
    fetcher: Fetcher,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum Fetcher {
    S3(S3Fetcher),
}

impl Fetcher {
    pub async fn validate(&self) -> Result<()> {
        match self {
            Fetcher::S3(s3_fetcher) => s3::validate_credentials(s3_fetcher).await,
        }
    }
}

pub type CollectionDatasources = HashMap<(CollectionId, IndexId), Arc<DatasourceWrapper>>;
type SerializableDatasourceMap = HashMap<(CollectionId, IndexId), Fetcher>;

#[derive(Clone)]
pub struct DatasourceStorage {
    base_dir: PathBuf,
    file_path: PathBuf,
    map: Arc<OramaAsyncLock<CollectionDatasources>>,
}

impl DatasourceStorage {
    pub fn try_new(base_dir: &PathBuf) -> Result<Self> {
        create_if_not_exists(base_dir).context("Cannot create datasource directory")?;

        let file_path = base_dir.join("datasources.json");
        let map = Self::load_from_disk(&file_path)?;

        Ok(Self {
            base_dir: base_dir.clone(),
            file_path,
            map: Arc::new(OramaAsyncLock::new("datasource_storage", map)),
        })
    }

    pub fn base_dir(&self) -> &PathBuf {
        &self.base_dir
    }

    pub async fn insert(
        &self,
        collection_id: CollectionId,
        index_id: IndexId,
        fetcher: Fetcher,
    ) -> Result<()> {
        fetcher.validate().await?;

        let mut m = self.map.write("insert").await;
        match m.entry((collection_id, index_id)) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Arc::new(DatasourceWrapper {
                    in_use: AtomicBool::new(false),
                    fetcher,
                }));
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                unreachable!("nop")
            }
        }

        Ok(())
    }

    pub async fn remove_collection(&self, collection_id: CollectionId) {
        let mut m = self.map.write("remove_collection").await;
        let keys_to_remove: Vec<_> = m
            .iter()
            .filter_map(|(k, _)| {
                if k.0 == collection_id {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();

        for key in keys_to_remove {
            m.remove(&key);
        }
    }

    pub async fn remove_index(&self, collection_id: CollectionId, index_id: IndexId) {
        let mut m = self.map.write("remove_index").await;
        m.remove(&(collection_id, index_id));
    }

    pub async fn get(&self) -> CollectionDatasources {
        self.map.read("get").await.clone()
    }

    pub async fn exists(&self, collection_id: CollectionId, index_id: IndexId) -> bool {
        let m = self.map.read("exists").await;
        m.contains_key(&(collection_id, index_id))
    }

    fn load_from_disk(file_path: &PathBuf) -> Result<CollectionDatasources> {
        if file_path.exists() {
            let file = std::fs::File::open(file_path)
                .with_context(|| format!("Cannot open datasources.json at {:?}", file_path))?;
            let serializable_map: SerializableDatasourceMap = serde_json::from_reader(file)
                .with_context(|| {
                    format!("Cannot deserialize datasources.json at {:?}", file_path)
                })?;

            Ok(serializable_map
                .into_iter()
                .map(|(key, fetcher)| {
                    (
                        key,
                        Arc::new(DatasourceWrapper {
                            in_use: AtomicBool::new(false),
                            fetcher,
                        }),
                    )
                })
                .collect())
        } else {
            Ok(HashMap::new())
        }
    }

    pub async fn commit(&self) -> Result<()> {
        let file_path_tmp = self.file_path.with_extension("tmp.json");
        let map_guard = self.map.read("commit").await;

        let serializable_map: SerializableDatasourceMap = map_guard
            .iter()
            .map(|(key, value)| (key.clone(), value.fetcher.clone()))
            .collect();

        let file = std::fs::File::create(&file_path_tmp)
            .with_context(|| format!("Cannot create datasources.json at {:?}", file_path_tmp))?;

        serde_json::to_writer_pretty(file, &serializable_map)
            .with_context(|| format!("Cannot serialize datasources to {:?}", file_path_tmp))?;

        std::fs::rename(&file_path_tmp, &self.file_path).with_context(|| {
            format!("Cannot rename {:?} to {:?}", file_path_tmp, self.file_path)
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::utils::setup_s3_container;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_insert_fail_invalid_credentials() {
        let temp_dir = tempdir().unwrap();
        let storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();

        let coll_id = CollectionId::try_new("coll1".to_string()).unwrap();
        let index_id = IndexId::try_new("idx1".to_string()).unwrap();

        let fetcher = Fetcher::S3(S3Fetcher {
            bucket: "bucket1".to_string(),
            region: "us-east-1".to_string(),
            access_key_id: "invaid".to_string(),
            secret_access_key: "invalid".to_string(),
            endpoint_url: None,
        });

        let result = storage
            .insert(coll_id.clone(), index_id.clone(), fetcher.clone())
            .await;
        assert!(result.is_err());

        let all_ds = storage.get().await;
        assert!(all_ds.is_empty());
    }

    #[tokio::test]
    async fn test_datasource_persistence() {
        let (_s3_client, bucket_name, _container, endpoint_url) = setup_s3_container().await;
        let temp_dir = tempdir().unwrap();
        let storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();

        let coll_id = CollectionId::try_new("coll1".to_string()).unwrap();
        let index_id = IndexId::try_new("idx1".to_string()).unwrap();

        let fetcher = Fetcher::S3(S3Fetcher {
            bucket: bucket_name,
            region: "us-east-1".to_string(),
            access_key_id: "test".to_string(),
            secret_access_key: "test".to_string(),
            endpoint_url: Some(endpoint_url),
        });

        storage
            .insert(coll_id.clone(), index_id.clone(), fetcher.clone())
            .await
            .unwrap();
        storage.commit().await.unwrap();

        let new_storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();
        let all_ds = new_storage.get().await;
        let key = (coll_id, index_id);
        assert!(all_ds.contains_key(&key));
        assert_eq!(all_ds[&key].fetcher, fetcher);
        assert_eq!(
            all_ds[&key]
                .in_use
                .load(std::sync::atomic::Ordering::SeqCst),
            false
        );
    }
}
