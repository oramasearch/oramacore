use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    file_utils::{create_if_not_exists, BufferedFile},
    types::DocumentId,
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

        let data = StringFilterFieldDump::V1(StringFilterFieldDumpV1 { data: inner });
        BufferedFile::create_or_overwrite(data_dir.join("data.bin"))
            .context("Cannot create data.bin")?
            .write_bincode_data(&data)
            .context("Cannot write data.bin")?;
        let StringFilterFieldDump::V1(inner) = data;

        Ok(Self {
            inner: inner.data,
            data_dir,
        })
    }

    pub fn load(info: StringFilterFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let dump_file_path = data_dir.join("data.bin");

        let inner: StringFilterFieldDump = BufferedFile::open(dump_file_path)
            .context("Cannot open data.bin")?
            .read_bincode_data()
            .context("Cannot read data.bin")?;
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

    pub fn get_stats(&self) -> StringFilterCommittedFieldStats {
        let doc_count = self.inner.values().map(|v| v.len()).sum();
        StringFilterCommittedFieldStats {
            variant_count: self.inner.len(),
            doc_count,
        }
    }

    pub fn filter<'s, 'iter>(&'s self, filter: &str) -> impl Iterator<Item = DocumentId> + 'iter
    where
        's: 'iter,
    {
        self.inner
            .get(filter)
            .map(|doc_ids| doc_ids.iter().cloned())
            .unwrap_or_default()
    }

    pub fn get_string_value<'s, 'iter>(&'s self) -> impl Iterator<Item = &'s str> + 'iter
    where
        's: 'iter,
    {
        self.inner.keys().map(|k| k.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (String, HashSet<DocumentId>)> + '_ {
        self.inner
            .iter()
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
    pub doc_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
enum StringFilterFieldDump {
    V1(StringFilterFieldDumpV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct StringFilterFieldDumpV1 {
    data: HashMap<String, HashSet<DocumentId>>,
}
