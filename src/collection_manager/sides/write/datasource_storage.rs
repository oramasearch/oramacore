use crate::lock::OramaAsyncLock;
use crate::types::{CollectionId, IndexId};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct S3 {
    pub bucket: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub endpoint_url: Option<String>,
}

#[derive(Clone, Debug)]
pub enum DatasourceKind {
    S3(S3),
}

#[derive(Clone, Debug)]
pub struct DatasourceEntry {
    pub id: IndexId,
    pub datasource: DatasourceKind,
}

pub struct DatasourceStorage {
    map: Arc<OramaAsyncLock<HashMap<CollectionId, HashMap<IndexId, Vec<DatasourceEntry>>>>>,
}

impl DatasourceStorage {
    pub fn new() -> Self {
        Self {
            map: Arc::new(OramaAsyncLock::new("datasource_storge", HashMap::new())),
        }
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

    pub async fn get(&self) -> HashMap<CollectionId, HashMap<IndexId, Vec<DatasourceEntry>>> {
        let m = self.map.read("get_by_type").await;
        let mut result: HashMap<CollectionId, HashMap<IndexId, Vec<DatasourceEntry>>> =
            HashMap::new();

        for (collection_id, indexes) in m.iter() {
            for (index_id, datasources) in indexes.iter() {
                let filtered_datasources: Vec<DatasourceEntry> =
                    datasources.iter().cloned().collect();

                if !filtered_datasources.is_empty() {
                    result
                        .entry(*collection_id)
                        .or_default()
                        .insert(*index_id, filtered_datasources);
                }
            }
        }

        result
    }
}
