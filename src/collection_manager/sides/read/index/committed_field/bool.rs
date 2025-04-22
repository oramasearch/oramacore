use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    indexes::ordered_key::{BoundedValue, OrderedKeyIndex},
    types::DocumentId,
};

#[derive(Debug)]
pub struct CommittedBoolField {
    field_path: Box<[String]>,
    inner: OrderedKeyIndex<BoolWrapper, DocumentId>,
    data_dir: PathBuf,
}

impl CommittedBoolField {
    pub fn from_data(
        field_path: Box<[String]>,
        true_docs: HashSet<DocumentId>,
        false_docs: HashSet<DocumentId>,
        data_dir: PathBuf,
    ) -> Result<Self> {
        let inner = OrderedKeyIndex::from_iter(
            [
                (BoolWrapper::False, false_docs),
                (BoolWrapper::True, true_docs),
            ]
            .into_iter(),
            data_dir.clone(),
        )?;

        Ok(Self {
            field_path,
            inner,
            data_dir,
        })
    }

    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (BoolWrapper, HashSet<DocumentId>)>,
    {
        let inner = OrderedKeyIndex::from_iter(iter, data_dir.clone())?;
        Ok(Self {
            field_path,
            inner,
            data_dir,
        })
    }

    pub fn load(info: BoolFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let inner = OrderedKeyIndex::load(data_dir.clone())?;
        Ok(Self {
            field_path: info.field_path,
            inner,
            data_dir,
        })
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
        let false_count = self
            .inner
            .count(BoolWrapper::False)
            .context("Cannot count false values")?;
        let true_count = self
            .inner
            .count(BoolWrapper::True)
            .context("Cannot count true values")?;
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
        let w = value.into();
        Ok(self
            .inner
            .get_items(w, w)
            .context("Cannot get items for bool field")?
            .flat_map(|item| item.values))
    }

    pub fn clone_inner(&self) -> Result<(HashSet<DocumentId>, HashSet<DocumentId>)> {
        let false_docs: HashSet<_> = self
            .inner
            .get_items(BoolWrapper::False, BoolWrapper::False)?
            .flat_map(|item| item.values)
            .collect();
        let true_docs: HashSet<_> = self
            .inner
            .get_items(BoolWrapper::True, BoolWrapper::True)?
            .flat_map(|item| item.values)
            .collect();

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
