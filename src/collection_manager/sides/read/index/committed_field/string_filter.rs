use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::collection_manager::sides::read::index::merge::CommittedFieldMetadata;

/// Metadata for a committed string filter field.
/// Retained for DumpV1 compatibility and legacy migration in StringFilterFieldStorage.
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

    fn field_path(&self) -> &[String] {
        self.field_path.as_ref()
    }
}
