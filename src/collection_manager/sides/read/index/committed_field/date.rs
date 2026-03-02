use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::collection_manager::sides::read::index::merge::CommittedFieldMetadata;

/// Metadata for a date field stored in the DumpV1 format.
/// Kept for backwards compatibility with existing dumps and for use by DateFieldStorage.
#[derive(Debug, Serialize, Deserialize)]
pub struct DateFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

impl CommittedFieldMetadata for DateFieldInfo {
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
