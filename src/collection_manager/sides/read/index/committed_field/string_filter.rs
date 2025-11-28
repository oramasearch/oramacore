use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::read::{
        index::{
            merge::{CommittedField, CommittedFieldMetadata, Field},
            uncommitted_field::UncommittedStringFilterField,
        },
        OffloadFieldConfig,
    },
    merger::MergedIterator,
    types::DocumentId,
};
use oramacore_lib::fs::BufferedFile;

#[derive(Debug)]
pub struct CommittedStringFilterField {
    field_path: Box<[String]>,
    inner: HashMap<String, HashSet<DocumentId>>,
    data_dir: PathBuf,
}

impl CommittedStringFilterField {
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

impl CommittedField for CommittedStringFilterField {
    type FieldMetadata = StringFilterFieldInfo;
    type Uncommitted = UncommittedStringFilterField;

    fn from_uncommitted(
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let iter = uncommitted.iter().map(|(k, mut d)| {
            d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
            (k, d)
        });

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
            field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
            inner: inner.data,
            data_dir,
        })
    }

    fn try_load(
        metadata: Self::FieldMetadata,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let data_dir = metadata.data_dir;
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
            field_path: metadata.field_path,
        })
    }

    fn add_uncommitted(
        &self,
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let uncommitted_iter = uncommitted.iter();
        let committed_iter = self.iter();

        let iter = MergedIterator::new(
            committed_iter,
            uncommitted_iter,
            |_, v| v,
            |_, mut v1, v2| {
                v1.extend(v2);
                v1
            },
        )
        .map(|(k, mut d)| {
            d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
            (k, d)
        });

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
            field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
            inner: inner.data,
            data_dir,
        })
    }

    fn metadata(&self) -> Self::FieldMetadata {
        StringFilterFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

impl Field for CommittedStringFilterField {
    type FieldStats = CommittedStringFilterFieldStats;

    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }

    fn stats(&self) -> CommittedStringFilterFieldStats {
        let doc_count = self.inner.values().map(|v| v.len()).sum();

        let keys = Some(self.inner.keys().cloned().collect());

        CommittedStringFilterFieldStats {
            key_count: self.inner.len(),
            document_count: doc_count,
            keys,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StringFilterFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

impl CommittedFieldMetadata for StringFilterFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }
    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }
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
