use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Metadata for persisting and loading bool field data via DumpV1.
#[derive(Debug, Serialize, Deserialize)]
pub struct BoolFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}
