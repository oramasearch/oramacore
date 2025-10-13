use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::types::DocumentId;
use oramacore_lib::data_structures::ordered_key::BoundedValue;
use oramacore_lib::fs::create_if_not_exists;

#[derive(Debug)]
pub struct CommittedBoolField {
    field_path: Box<[String]>,
    map: HashMap<bool, HashSet<DocumentId>>,
    data_dir: PathBuf,
}

impl CommittedBoolField {
    pub fn from_data(
        field_path: Box<[String]>,
        true_docs: HashSet<DocumentId>,
        false_docs: HashSet<DocumentId>,
        data_dir: PathBuf,
    ) -> Result<Self> {
        let mut map = HashMap::with_capacity(2);
        map.insert(false, false_docs);
        map.insert(true, true_docs);

        let field = Self {
            field_path,
            map,
            data_dir,
        };

        field.commit().context("Failed to commit bool field")?;

        Ok(field)
    }

    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (BoolWrapper, HashSet<DocumentId>)>,
    {
        let mut map = HashMap::with_capacity(2);
        map.insert(false, HashSet::new());
        map.insert(true, HashSet::new());

        for (key, values) in iter {
            match key {
                BoolWrapper::False => {
                    map.insert(false, values);
                }
                BoolWrapper::True => {
                    map.insert(true, values);
                }
                _ => continue, // Ignore Min and Max
            }
        }

        let field = Self {
            field_path,
            map,
            data_dir: data_dir.clone(),
        };

        field.commit().context("Failed to commit bool field")?;

        Ok(field)
    }

    #[allow(deprecated)]
    pub fn try_load(info: BoolFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;

        let map = match std::fs::File::open(data_dir.join("bool_map.bin")) {
            Ok(file) => bincode::deserialize_from::<_, HashMap<bool, HashSet<DocumentId>>>(file)
                .context("Failed to deserialize bool_map.bin")?,
            Err(_) => {
                info!("bool_map.bin not found, reconstructing from OrderedKeyIndex");
                use oramacore_lib::data_structures::ordered_key::OrderedKeyIndex;
                let inner = match OrderedKeyIndex::<BoolWrapper, DocumentId>::load(data_dir.clone())
                {
                    Ok(index) => index,
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to load OrderedKeyIndex for bool field: {}",
                            e
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

                map
            }
        };

        let s = Self {
            field_path: info.field_path,
            map,
            data_dir,
        };

        s.commit().context("Failed to commit bool field")?;

        Ok(s)
    }

    fn commit(&self) -> Result<()> {
        // Ensure the data directory exists
        create_if_not_exists(&self.data_dir).context("Failed to create data directory")?;

        let file_path = self.data_dir.join("bool_map.bin");
        let file = std::fs::File::create(&file_path).context("Failed to create bool_map.bin")?;
        bincode::serialize_into(file, &self.map).context("Failed to serialize bool map")?;

        Ok(())
    }

    pub fn get_field_info(&self) -> BoolFieldInfo {
        BoolFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn stats(&self) -> Result<CommittedBoolFieldStats> {
        let false_count = self.map.get(&false).map_or(0, |set| set.len());
        let true_count = self.map.get(&true).map_or(0, |set| set.len());

        Ok(CommittedBoolFieldStats {
            false_count,
            true_count,
        })
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
