use crate::lock::OramaAsyncLock;
use crate::types::IndexId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct S3 {
    pub bucket: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub endpoint_url: Option<String>,
    pub pull_interval: Duration,
}

#[derive(Clone, Debug)]
pub enum Datasource {
    S3(S3),
}

pub struct DatasourceStorage {
    map: Arc<OramaAsyncLock<HashMap<IndexId, Vec<Datasource>>>>,
}

impl DatasourceStorage {
    pub fn new() -> Self {
        Self {
            map: Arc::new(OramaAsyncLock::new("datasource_storge", HashMap::new())),
        }
    }

    pub async fn insert(&mut self, index_id: IndexId, datasource: Datasource) {
        let mut m = self.map.write("insert").await;
        m.entry(index_id).or_default().push(datasource);
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

