use std::path::PathBuf;

use oramacore_lib::filters::FilterResult;
use serde::{Deserialize, Serialize};

use crate::{python::embeddings::Model, types::DocumentId};

use crate::collection_manager::sides::read::index::merge::CommittedFieldMetadata;

/// Search parameters for vector similarity search.
/// Used by EmbeddingFieldStorage's search API.
pub struct VectorSearchParams<'search> {
    pub target: &'search [f32],
    pub similarity: f32,
    pub limit: usize,
    pub filtered_doc_ids: Option<&'search FilterResult<DocumentId>>,
}

/// Metadata for vector fields, used for DumpV1 serialization and migration.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
    pub model: Model,
}

impl CommittedFieldMetadata for VectorFieldInfo {
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
