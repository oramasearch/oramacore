use crate::lock::OramaAsyncLock;
use crate::types::{CollectionId, IndexId};
use anyhow::{Context, Result};
use oramacore_lib::fs::create_if_not_exists;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

type IndexDatasources = HashMap<IndexId, Vec<DatasourceEntry>>;
type CollectionDatasources = HashMap<CollectionId, IndexDatasources>;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct S3 {
    pub bucket: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub endpoint_url: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum DatasourceKind {
    S3(S3),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DatasourceEntry {
    pub id: IndexId,
    #[serde(flatten)]
    pub datasource: DatasourceKind,
}

#[derive(Clone)]
pub struct DatasourceStorage {
    base_dir: PathBuf,
    map: Arc<OramaAsyncLock<CollectionDatasources>>,
}

impl DatasourceStorage {
    pub fn try_new(base_dir: &PathBuf) -> Result<Self> {
        create_if_not_exists(base_dir).context("Cannot create data directory")?;

        let file_path = base_dir.join("datasources.json");
        let map = if file_path.exists() {
            let file = std::fs::File::open(&file_path)
                .with_context(|| format!("Cannot open datasources.json at {:?}", file_path))?;
            serde_json::from_reader(file).with_context(|| {
                format!("Cannot deserialize datasources.json at {:?}", file_path)
            })?
        } else {
            HashMap::new()
        };

        Ok(Self {
            base_dir: base_dir.clone(),
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
        datasource: DatasourceEntry,
    ) {
        let mut m = self.map.write("insert").await;
        m.entry(collection_id)
            .or_default()
            .entry(index_id)
            .or_default()
            .push(datasource);
    }

    pub async fn remove_collection(&self, collection_id: CollectionId) {
        let mut m = self.map.write("remove_collection").await;
        m.remove(&collection_id);
    }

    pub async fn remove_index(&self, collection_id: CollectionId, index_id: IndexId) {
        let mut m = self.map.write("remove_index").await;
        if let Some(collection_map) = m.get_mut(&collection_id) {
            collection_map.remove(&index_id);
            if collection_map.is_empty() {
                m.remove(&collection_id);
            }
        }
    }

    pub async fn remove_datasource(
        &self,
        collection_id: CollectionId,
        index_id: IndexId,
        datasource_id: IndexId,
    ) {
        let mut m = self.map.write("remove_datasource").await;
        if let Some(collection_map) = m.get_mut(&collection_id) {
            if let Some(entries) = collection_map.get_mut(&index_id) {
                entries.retain(|e| e.id != datasource_id);

                if entries.is_empty() {
                    collection_map.remove(&index_id);
                }
            }
            if collection_map.is_empty() {
                m.remove(&collection_id);
            }
        }
    }

    pub async fn get(&self) -> CollectionDatasources {
        self.map.read("get").await.clone()
    }

    pub async fn commit(&self) -> Result<()> {
        let file_path = self.base_dir.join("datasources.json");
        let file_path_tmp = self.base_dir.join("datasources.tmp.json");
        let map_guard = self.map.read("commit").await;
        let map: &CollectionDatasources = &*map_guard;

        let file = std::fs::File::create(&file_path_tmp)
            .with_context(|| format!("Cannot create datasources.json at {:?}", file_path_tmp))?;

        serde_json::to_writer_pretty(file, map)
            .with_context(|| format!("Cannot serialize datasources to {:?}", file_path_tmp))?;

        std::fs::rename(&file_path_tmp, &file_path)
            .with_context(|| format!("Cannot rename {:?} to {:?}", file_path_tmp, file_path))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_datasoruce_persistence() {
        let temp_dir = tempdir().unwrap();
        let storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();

        let coll_id = CollectionId::try_new("coll1".to_string()).unwrap();
        let index_id = IndexId::try_new("idx1".to_string()).unwrap();
        let ds_id = IndexId::try_new("ds1".to_string()).unwrap();

        let ds = DatasourceEntry {
            id: ds_id,
            datasource: DatasourceKind::S3(S3 {
                bucket: "bucket1".to_string(),
                region: "us-east-1".to_string(),
                access_key_id: "key_id".to_string(),
                secret_access_key: "secret".to_string(),
                endpoint_url: None,
            }),
        };

        storage.insert(coll_id, index_id, ds.clone()).await;
        storage.commit().await.unwrap();

        let new_storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();
        let all_ds = new_storage.get().await;
        assert_eq!(all_ds[&coll_id][&index_id][0], ds);

        new_storage
            .remove_datasource(coll_id, index_id, ds_id)
            .await;
        new_storage.commit().await.unwrap();

        let final_storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();
        assert!(final_storage.get().await.is_empty());
    }
}
