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
pub struct CommittedStringFilterField {
    field_path: Box<[String]>,
    inner: HashMap<String, HashSet<DocumentId>>,
    data_dir: PathBuf,
}

impl CommittedStringFilterField {
    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
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
            field_path,
            inner: inner.data,
            data_dir,
        })
    }

    pub fn try_load(info: StringFilterFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let dump_file_path = data_dir.join("data.bin");

        let inner: StringFilterFieldDump = BufferedFile::open(dump_file_path)
            .context("Cannot open data.bin")?
            .read_bincode_data()
            .context("Cannot read data.bin")?;
        let inner = match inner {
            StringFilterFieldDump::V1(inner) => inner.data,
        };
        Ok(Self {
            inner,
            data_dir,
            field_path: info.field_path,
        })
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn get_field_info(&self) -> StringFilterFieldInfo {
        StringFilterFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn stats(&self, with_keys: bool) -> CommittedStringFilterFieldStats {
        let doc_count = self.inner.values().map(|v| v.len()).sum();

        let keys = if with_keys {
            Some(self.inner.keys().cloned().collect())
        } else {
            None
        };

        CommittedStringFilterFieldStats {
            key_count: self.inner.len(),
            document_count: doc_count,
            keys,
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

    pub fn get_string_values<'s, 'iter>(&'s self) -> impl Iterator<Item = &'s str> + 'iter
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
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct CommittedStringFilterFieldStats {
    pub key_count: usize,
    pub document_count: usize,
    pub keys: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
enum StringFilterFieldDump {
    V1(StringFilterFieldDumpV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct StringFilterFieldDumpV1 {
    data: HashMap<String, HashSet<DocumentId>>,
}
