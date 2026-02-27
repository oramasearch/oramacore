use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::collection_manager::sides::read::index::merge::CommittedFieldMetadata;

/// Metadata for a geopoint field, used in DumpV1 serialization.
/// Kept for backward compatibility and migration from old format.
#[derive(Debug, Serialize, Deserialize)]
pub struct GeoPointFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

impl CommittedFieldMetadata for GeoPointFieldInfo {
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
