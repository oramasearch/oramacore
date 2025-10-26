use crate::lock::OramaAsyncLock;
use crate::types::{CollectionId, IndexId};
use anyhow::{Context, Result};
use oramacore_lib::fs::create_if_not_exists;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

type IndexDatasources = HashMap<IndexId, Vec<DatasourceEntry>>;
type CollectionDatasources = HashMap<CollectionId, IndexDatasources>;

#[derive(Clone, Debug, Deserialize)]
pub struct S3 {
    pub bucket: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub endpoint_url: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DatasourceKind {
    S3(S3),
}

#[derive(Clone, Debug, Deserialize)]
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

        Ok(Self {
            base_dir: base_dir.clone(),
            map: Arc::new(OramaAsyncLock::new("datasource_storage", HashMap::new())),
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
}
