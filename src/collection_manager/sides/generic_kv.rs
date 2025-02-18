use crate::{file_utils::BufferedFile, types::CollectionId};
use anyhow::{Context, Result};
use ptrie::Trie;
use serde::{de::DeserializeOwned, Serialize};
use tokio::sync::RwLock;

#[derive(Debug)]
pub struct KV<V = String> {
    data: RwLock<Trie<u8, V>>,
}

impl<V: Clone + Serialize + DeserializeOwned> KV<V> {
    pub fn new() -> Self {
        KV {
            data: RwLock::new(Trie::new()),
        }
    }

    pub async fn insert(&self, key: String, value: V) {
        self.data
            .write()
            .await
            .insert(key.as_bytes().iter().cloned(), value)
    }

    pub async fn get(&self, key: &str) -> Option<V> {
        let read_ref = self.data.read().await;
        read_ref.get(key.as_bytes().iter().cloned()).cloned()
    }

    pub async fn remove(&self, key: &str) -> Option<V> {
        self.data
            .write()
            .await
            .remove(key.as_bytes().iter().cloned())
    }

    pub async fn prefix_scan(&self, prefix: &str) -> Vec<V> {
        let read_ref = self.data.read().await;

        read_ref
            .find_prefixes(prefix.as_bytes().iter().cloned())
            .into_iter()
            .map(|v| v.clone())
            .collect()
    }

    pub async fn commit(&self) -> Result<()> {
        let data = self.data.read().await;
        let path = "data/kv".to_string();

        BufferedFile::create_or_overwrite(path)
            .context("Cannot create previous kv info")?
            .write_json_data(&*data)
            .context("Cannot write previous kv info")
    }
}

pub fn format_key(collection_id: CollectionId, key: &str) -> String {
    format!("{}:{}", collection_id.0, key)
}
