use crate::types::CollectionId;
use ptrie::Trie;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KV<V = String> {
    data: Trie<u8, V>,
}

impl<V: Clone> KV<V> {
    pub fn new() -> Self {
        KV { data: Trie::new() }
    }

    pub fn insert(&mut self, key: String, value: V) {
        self.data.insert(key.as_bytes().iter().cloned(), value);
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.data.get(key.as_bytes().iter().cloned())
    }

    pub fn remove(&mut self, key: &str) -> Option<V> {
        self.data.remove(key.as_bytes().iter().cloned())
    }

    pub fn prefix_scan<'a>(&'a self, prefix: &str) -> impl Iterator<Item = (Vec<u8>, V)> + 'a
    where
        V: Clone,
    {
        let prefix_bytes = prefix.as_bytes().to_vec();
        self.data
            .iter()
            .filter(move |(key, _)| key.starts_with(&prefix_bytes))
    }
}

pub fn format_key(collection_id: CollectionId, key: &str) -> String {
    format!("{}:{}", collection_id.0, key)
}
