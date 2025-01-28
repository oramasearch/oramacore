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
}

impl<Key: Debug, Value: Debug> Debug for Map<Key, Value> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map").field("items", &self.inner).finish()
    }
}

impl<Key: Eq + Hash + Serialize + DeserializeOwned, Value: Serialize + DeserializeOwned>
    Map<Key, Value>
{
    pub fn from_hash_map(
        hash_map: HashMap<Key, Value>
    ) -> Self {
        Self { inner: hash_map }
    }

    pub fn from_iter<I>(iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (Key, Value)>,
    {
        create_if_not_exists(&data_dir)
            .context("Cannot create the base directory for the committed index")?;
        let path_to_commit = data_dir.join("index.map");

        let map: HashMap<_, _> = iter.collect();

        BufferedFile::create(path_to_commit.clone())
            .context("Cannot create file")?
            .write_json_data(&map)
            .context("Cannot write map to file")?;

        Ok(Self { inner: map })
    }

    pub fn load(data_dir: PathBuf) -> Result<Self> {
        let path_to_commit = data_dir.join("index.map");

        let map: HashMap<Key, Value> = BufferedFile::open(path_to_commit.clone())
            .context("Cannot open file")?
            .read_json_data()
            .context("Cannot read map from file")?;

        Ok(Self { inner: map })
    }

    pub fn get(&self, key: &Key) -> Option<&Value> {
        self.inner.get(key)
    }
}
