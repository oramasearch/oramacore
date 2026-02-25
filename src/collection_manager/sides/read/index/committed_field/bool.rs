use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::collection_manager::sides::read::index::merge::CommittedFieldMetadata;
use oramacore_lib::data_structures::ordered_key::BoundedValue;

/// Legacy wrapper type for bool values in OrderedKeyIndex.
/// Kept for backward-compatible deserialization of old index data.
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

/// Metadata for persisting and loading bool field data via DumpV1.
#[derive(Debug, Serialize, Deserialize)]
pub struct BoolFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
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
