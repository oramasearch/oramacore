use std::path::PathBuf;

use oramacore_lib::data_structures::ordered_key::BoundedValue;
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::read::index::merge::CommittedFieldMetadata,
    types::{Number, SerializableNumber},
};

/// Metadata for serializing/deserializing number field info in DumpV1.
/// This struct is kept for backward compatibility with the DumpV1 format
/// and for migration from old formats.
#[derive(Debug, Serialize, Deserialize)]
pub struct NumberFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

impl CommittedFieldMetadata for NumberFieldInfo {
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

/// BoundedValue impl for SerializableNumber, used by the migration path
/// in number_field.rs when loading from old OrderedKeyIndex format.
impl BoundedValue for SerializableNumber {
    fn max_value() -> Self {
        SerializableNumber(Number::F32(f32::INFINITY))
    }

    fn min_value() -> Self {
        SerializableNumber(Number::F32(f32::NEG_INFINITY))
    }
}
