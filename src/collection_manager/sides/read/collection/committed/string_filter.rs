use std::{collections::{HashMap, HashSet}, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::dto::{Number, NumberFilter, SerializableNumber}, file_utils::{create_if_not_exists, BufferedFile}, indexes::ordered_key::{BoundedValue, OrderedKeyIndex}, types::DocumentId
};

#[derive(Debug)]
pub struct StringFilterField {
    inner: HashMap<String, HashSet<DocumentId>>,
    data_dir: PathBuf,
}

impl StringFilterField {
    pub fn from_iter<I>(iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (String, HashSet<DocumentId>)>,
    {
        create_if_not_exists(&data_dir).context("Cannot create data directory")?;

        let mut inner: HashMap<String, HashSet<DocumentId>> = HashMap::new();
        for (key, doc_ids) in iter {
            if let Some(v) = inner.get_mut(&key) {
                v.extend(doc_ids);
                continue;
            }
            inner.insert(key, doc_ids);
        }

        BufferedFile::create(data_dir.join("data.bin"))
            .context("Cannot create hnsw file")?
            .write_bincode_data(&inner)
            .context("Cannot write hnsw file")?;

        Ok(Self { inner, data_dir })
    }

    pub fn load(info: StringFilterFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let dump_file_path = data_dir.join("data.bin");

        let inner: StringFilterFieldDump = BufferedFile::open(dump_file_path)
            .context("Cannot open hnsw file")?
            .read_bincode_data()
            .context("Cannot read hnsw file")?;
        let inner = match inner {
            StringFilterFieldDump::V1(inner) => inner.data,
        };
        Ok(Self { inner, data_dir })
    }

    pub fn get_field_info(&self) -> StringFilterFieldInfo {
        StringFilterFieldInfo {
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn get_stats(&self) -> Result<StringFilterCommittedFieldStats> {
        Ok(StringFilterCommittedFieldStats {
            variant_count: self.inner.len(),
        })
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter: &String,
    ) -> impl Iterator<Item = DocumentId> + 'iter
    where
        's: 'iter,
    {
        self.inner.get(filter).map(|doc_ids| doc_ids.iter().cloned()).unwrap_or_default()
    }

    pub fn iter(&self) -> impl Iterator<Item = (String, HashSet<DocumentId>)> + '_ {
        self.inner.iter()
            .map(|(k, doc_ids)| (k.clone(), doc_ids.clone()))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StringFilterFieldInfo {
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct StringFilterCommittedFieldStats {
    pub variant_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
enum StringFilterFieldDump {
    V1(StringFilterFieldDumpV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct StringFilterFieldDumpV1 {
    data: HashMap<String, HashSet<DocumentId>>,
}
