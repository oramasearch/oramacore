use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Metadata for a date field stored in the DumpV1 format.
/// Kept for backwards compatibility with existing dumps and for use by DateFieldStorage.
#[derive(Debug, Serialize, Deserialize)]
pub struct DateFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}
