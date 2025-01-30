use std::{collections::HashMap, fmt::Debug, hash::Hash, path::PathBuf};

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};

use crate::file_utils::{create_if_not_exists, BufferedFile};

pub struct Map<Key, Value> {
    // Probably hashmap isn't a good choice here
    // We should use a datastructure that allows us to store the data in disk
    // For instance, https://crates.io/crates/odht
    // TODO: think about this
    inner: HashMap<Key, Value>,

    file_path: PathBuf,
}

impl<Key: Debug, Value: Debug> Debug for Map<Key, Value> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map").field("items", &self.inner).finish()
    }
}

impl<Key: Eq + Hash + Serialize + DeserializeOwned, Value: Serialize + DeserializeOwned>
    Map<Key, Value>
{
    pub fn from_hash_map(hash_map: HashMap<Key, Value>, file_path: PathBuf) -> Result<Self> {
        let s = Self {
            inner: hash_map,
            file_path,
        };

        s.commit()?;

        Ok(s)
    }

    pub fn from_iter<I>(iter: I, file_path: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (Key, Value)>,
    {
        let map: HashMap<_, _> = iter.collect();
        Self::from_hash_map(map, file_path)
    }

    pub fn commit(&self) -> Result<()> {
        create_if_not_exists(self.file_path.parent().expect("file_path has a parent"))
            .context("Cannot create the base directory for the committed index")?;
        BufferedFile::create_or_overwrite(self.file_path.clone())
            .context("Cannot create file")?
            .write_bincode_data(&self.inner)
            .context("Cannot write map to file")?;

        Ok(())
    }

    pub fn load(file_path: PathBuf) -> Result<Self> {
        let map: HashMap<Key, Value> = BufferedFile::open(file_path.clone())
            .context("Cannot open file")?
            .read_bincode_data()
            .context("Cannot read map from file")?;

        Ok(Self {
            inner: map,
            file_path,
        })
    }

    pub fn get(&self, key: &Key) -> Option<&Value> {
        self.inner.get(key)
    }
    pub fn file_path(&self) -> PathBuf {
        self.file_path.clone()
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.inner.values()
    }
    pub fn insert(&mut self, key: Key, value: Value) {
        self.inner.insert(key, value);
    }
}

impl<Key: Ord, Value> Map<Key, Value> {
    pub fn get_max_key(&self) -> Option<&Key> {
        self.inner.keys().max()
    }
}

impl<Key: Debug + Eq + Hash, Value: Debug> Map<Key, Vec<Value>> {
    pub fn merge(&mut self, key: Key, iter: impl Iterator<Item = Value>) {
        let entry = self.inner.entry(key).or_default();
        entry.extend(iter);
    }
}
