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
}
