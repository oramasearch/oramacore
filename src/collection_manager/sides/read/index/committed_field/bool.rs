use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    collection_manager::sides::read::{
        index::{
            filter::Filterable,
            merge::{CommittedField, CommittedFieldMetadata, Field},
            uncommitted_field::UncommittedBoolField,
        },
        OffloadFieldConfig,
    },
    types::DocumentId,
};
use oramacore_lib::{data_structures::ordered_key::BoundedValue, fs::BufferedFile};

#[derive(Debug)]
pub struct CommittedBoolField {
    field_path: Box<[String]>,
    map: HashMap<bool, HashSet<DocumentId>>,
    data_dir: PathBuf,
}

impl CommittedBoolField {
    fn commit(&self) -> Result<()> {
        let file_path = self.data_dir.join("bool_map.bin");
        BufferedFile::create_or_overwrite(file_path)
            .context("Failed to create bool_map.bin")?
            .write_bincode_data(&self.map)
            .context("Failed to serialize bool map")?;

        Ok(())
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        value: bool,
    ) -> Result<impl Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        let docs = self.map.get(&value).unwrap();

        Ok(docs.iter().copied())
    }

    pub fn clone_inner(&self) -> Result<(HashSet<DocumentId>, HashSet<DocumentId>)> {
        let false_docs: HashSet<_> = self.map.get(&false).unwrap().clone();
        let true_docs: HashSet<_> = self.map.get(&true).unwrap().clone();

        Ok((true_docs, false_docs))
    }

    /// Returns references to the inner HashSets without cloning.
    /// Returns (true_docs, false_docs).
    pub fn inner_ref(&self) -> (&HashSet<DocumentId>, &HashSet<DocumentId>) {
        (
            self.map.get(&true).expect("true entry must exist"),
            self.map.get(&false).expect("false entry must exist"),
        )
    }
}

impl CommittedField for CommittedBoolField {
    type FieldMetadata = BoolFieldInfo;
    type Uncommitted = UncommittedBoolField;

    fn from_uncommitted(
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let (mut true_docs, mut false_docs) = uncommitted.clone_inner();
        true_docs.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
        false_docs.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));

        let mut map = HashMap::with_capacity(2);
        map.insert(false, false_docs);
        map.insert(true, true_docs);

        let field = Self {
            field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
            map,
            data_dir: data_dir.clone(),
        };

        field.commit().context("Failed to commit bool field")?;

        Ok(field)
    }

    #[allow(deprecated)]
    fn try_load(
        metadata: Self::FieldMetadata,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let data_dir: PathBuf = metadata.data_dir;

        let (map, needs_commit) = match std::fs::File::open(data_dir.join("bool_map.bin")) {
            Ok(file) => (
                bincode::deserialize_from::<_, HashMap<bool, HashSet<DocumentId>>>(file)
                    .context("Failed to deserialize bool_map.bin")?,
                false,
            ),
            Err(_) => {
                info!("bool_map.bin not found, reconstructing from OrderedKeyIndex");
                use oramacore_lib::data_structures::ordered_key::OrderedKeyIndex;
                let inner = match OrderedKeyIndex::<BoolWrapper, DocumentId>::load(data_dir.clone())
                {
                    Ok(index) => index,
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to load OrderedKeyIndex for bool field: {e}"
                        ));
                    }
                };

                let false_count = inner
                    .count(BoolWrapper::False)
                    .context("Cannot count false values")?;
                let true_count = inner
                    .count(BoolWrapper::True)
                    .context("Cannot count true values")?;

                let items = inner
                    .get_items((true, BoolWrapper::Min), (true, BoolWrapper::Max))
                    .context("Cannot get items for bool field")?;
                let mut map: HashMap<bool, HashSet<DocumentId>> = HashMap::with_capacity(2);
                for item in items {
                    for doc_id in item.values {
                        let key = match item.key {
                            BoolWrapper::False => false,
                            BoolWrapper::True => true,
                            _ => continue, // Ignore Min and Max
                        };
                        map.entry(key)
                            .or_insert_with(|| {
                                let capacity = if key { true_count } else { false_count };
                                HashSet::with_capacity(capacity)
                            })
                            .insert(doc_id);
                    }
                }

                (map, true)
            }
        };

        let s = Self {
            field_path: metadata.field_path,
            map,
            data_dir,
        };

        if needs_commit {
            s.commit().context("Failed to commit bool field")?;
        }

        Ok(s)
    }

    fn add_uncommitted(
        &self,
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let (uncommitted_true_docs, uncommitted_false_docs) = uncommitted.clone_inner();
        let (mut committed_true_docs, mut committed_false_docs) = self.clone_inner()?;

        committed_true_docs.extend(uncommitted_true_docs);
        committed_false_docs.extend(uncommitted_false_docs);
        committed_true_docs.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
        committed_false_docs.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));

        let mut map = HashMap::with_capacity(2);
        map.insert(false, committed_false_docs);
        map.insert(true, committed_true_docs);

        let field = Self {
            field_path: self.field_path.clone(),
            map,
            data_dir,
        };

        field.commit().context("Failed to commit bool field")?;

        Ok(field)
    }

    fn metadata(&self) -> Self::FieldMetadata {
        BoolFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

impl Field for CommittedBoolField {
    type FieldStats = CommittedBoolFieldStats;

    fn field_path(&self) -> &[String] {
        self.field_path.as_ref()
    }

    fn stats(&self) -> Self::FieldStats {
        CommittedBoolFieldStats {
            false_count: self.map.get(&false).map_or(0, |set| set.len()),
            true_count: self.map.get(&true).map_or(0, |set| set.len()),
        }
    }
}

impl Filterable for CommittedBoolField {
    type FilterParam = bool;

    fn filter<'s, 'iter>(
        &'s self,
        filter_param: &Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let docs = self
            .map
            .get(filter_param)
            .ok_or_else(|| anyhow::anyhow!("Boolean value {filter_param} not found in index"))?;

        Ok(Box::new(docs.iter().copied()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolWrapper {
    Min,
    False,
    True,
    Max,
}

impl PartialOrd for BoolWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BoolWrapper {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (BoolWrapper::Min, BoolWrapper::Min) => std::cmp::Ordering::Equal,
            (BoolWrapper::Min, _) => std::cmp::Ordering::Less,
            (_, BoolWrapper::Min) => std::cmp::Ordering::Greater,
            (BoolWrapper::False, BoolWrapper::False) => std::cmp::Ordering::Equal,
            (BoolWrapper::False, _) => std::cmp::Ordering::Less,
            (_, BoolWrapper::False) => std::cmp::Ordering::Greater,
            (BoolWrapper::True, BoolWrapper::True) => std::cmp::Ordering::Equal,
            (BoolWrapper::True, _) => std::cmp::Ordering::Less,
            (_, BoolWrapper::True) => std::cmp::Ordering::Greater,
            (BoolWrapper::Max, BoolWrapper::Max) => std::cmp::Ordering::Equal,
        }
    }
}

impl From<bool> for BoolWrapper {
    fn from(value: bool) -> Self {
        match value {
            false => BoolWrapper::False,
            true => BoolWrapper::True,
        }
    }
}
impl Serialize for BoolWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        match self {
            BoolWrapper::Min => serializer.serialize_i32(0),
            BoolWrapper::False => serializer.serialize_i32(1),
            BoolWrapper::True => serializer.serialize_i32(2),
            BoolWrapper::Max => serializer.serialize_i32(3),
        }
    }
}
impl<'a> Deserialize<'a> for BoolWrapper {
    fn deserialize<D>(deserializer: D) -> Result<BoolWrapper, D::Error>
    where
        D: serde::de::Deserializer<'a>,
    {
        let value = i32::deserialize(deserializer)?;
        match value {
            0 => Ok(BoolWrapper::Min),
            1 => Ok(BoolWrapper::False),
            2 => Ok(BoolWrapper::True),
            3 => Ok(BoolWrapper::Max),
            _ => Err(serde::de::Error::custom("Invalid value for BoolWrapper")),
        }
    }
}

impl BoundedValue for BoolWrapper {
    fn max_value() -> Self {
        BoolWrapper::Max
    }

    fn min_value() -> Self {
        BoolWrapper::Min
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BoolFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct CommittedBoolFieldStats {
    pub false_count: usize,
    pub true_count: usize,
}

impl CommittedFieldMetadata for BoolFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }

    fn field_path(&self) -> &[String] {
        self.field_path.as_ref()
    }
}
