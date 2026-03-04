use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Metadata for a committed string filter field.
/// Retained for DumpV1 compatibility and legacy migration in StringFilterFieldStorage.
#[derive(Debug, Serialize, Deserialize)]
pub struct StringFilterFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}
