use crate::lock::OramaAsyncLock;
use crate::types::IndexId;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct S3 {
    pub id: IndexId,
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

pub struct DatasourceStorage {
    map: Arc<OramaAsyncLock<HashMap<IndexId, Vec<DatasourceKind>>>>,
}

impl DatasourceStorage {
    pub fn new() -> Self {
        Self {
            map: Arc::new(OramaAsyncLock::new("datasource_storge", HashMap::new())),
        }
    }

    pub async fn insert(&self, index_id: IndexId, datasource: DatasourceKind) {
        let mut m = self.map.write("insert").await;
        m.entry(index_id).or_default().push(datasource);
    }

    pub async fn remove_single(&self, index_id: IndexId, datasource_id: IndexId) {
        let mut m = self.map.write("remove_single").await;
        if let Some(entries) = m.get_mut(&index_id) {
            entries.retain(|e| {
                let id_matches = match e {
                    DatasourceKind::S3(s3) => s3.id == datasource_id,
                };
                !id_matches
            });

            if entries.is_empty() {
                m.remove(&index_id);
            }
        }
    }

    pub async fn remove(&self, index_id: IndexId) {
        let mut m = self.map.write("remove").await;
        m.remove(&index_id);
    }

    // pub async fn get_datasources(&self, index_id: IndexId) -> Option<&Vec<Datasource>> {
    //     let m = self.map.write("get_datasources").await;
    //     m.get(&index_id)
    // }
}
