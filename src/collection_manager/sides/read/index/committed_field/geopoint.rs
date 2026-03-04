use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Metadata for a geopoint field, used in DumpV1 serialization.
/// Kept for backward compatibility and migration from old format.
#[derive(Debug, Serialize, Deserialize)]
pub struct GeoPointFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}
