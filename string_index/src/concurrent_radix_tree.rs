use std::{fmt::Debug, sync::{Arc, RwLock}};

use dashmap::DashMap;
use itertools::cloned;
use radix_trie::Trie;

#[derive(Debug)]
pub struct ConcurrentRadixTree<V: Debug> {
    inner: Arc<RwLock<Trie<String, V>>>,
}

impl<V: Clone + Debug> ConcurrentRadixTree<V> {
    pub fn new() -> ConcurrentRadixTree<V> {
        ConcurrentRadixTree {
            inner: Arc::new(RwLock::new(Trie::new())),
        }
    }

    pub fn get(&self, key: &str) -> Option<V> {
        let lock = self.inner.read().unwrap();
        let r = lock.get(key);
        r.cloned()
    }

    pub fn insert_or_update(
        &self,
        key: String,
        update_or_update: impl FnOnce(Option<&mut V>) -> InsertedOrUpdated<V>,
    ) {
        let mut guard = self.inner.write().unwrap();

        match guard.get_mut(&key) {
            None => {
                match update_or_update(None) {
                    InsertedOrUpdated::Inserted(v) => {
                        guard.insert(key, v);
                    },
                    InsertedOrUpdated::Updated => unreachable!("Wrong usage of insert_or_update"),
                }
            },
            Some(v) => {
                update_or_update(Some(v));
            },
        };
    }

    pub fn insert(&self, key: String, value: V) {
        let mut lock = self.inner.write().unwrap();
        lock.insert(key, value);
    }
}

pub enum InsertedOrUpdated<T> {
    Inserted(T),
    Updated,
}


#[derive(Debug)]
pub struct ConcurrentRadixTree2<V: Debug> {
    inner: DashMap<String, V>,
}


impl<V: Clone + Debug> ConcurrentRadixTree2<V> {
    pub fn new() -> Self {
        Self {
            inner: DashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<V> {
        let v = self.inner.get(key)
            .map(|r| r.value().clone());
        v
    }

    pub fn insert_or_update(
        &self,
        key: String,
        update_or_update: impl FnOnce(Option<&mut V>) -> InsertedOrUpdated<V>,
    ) {

        let entry = self.inner.entry(key);

        match entry {
            dashmap::Entry::Occupied(mut e) => {
                update_or_update(Some(e.get_mut()));
            },
            dashmap::Entry::Vacant(e) => {
                match update_or_update(None) {
                    InsertedOrUpdated::Inserted(v) => {
                        e.insert(v);
                    },
                    InsertedOrUpdated::Updated => unreachable!("Wrong usage of insert_or_update"),
                }
            }
        };
    }

    pub fn insert(&self, key: String, value: V) {
        self.inner.insert(key, value);
    }
}
