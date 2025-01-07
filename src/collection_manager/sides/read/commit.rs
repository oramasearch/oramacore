use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::collection_manager::{
    dto::{FieldId, TypedField},
    CollectionId,
};

use super::CollectionReader;

pub struct CommitConfig {
    pub folder_to_commit: PathBuf,
    pub epoch: u64,
}

impl CollectionReader {
    pub fn commit(&self, commit_config: CommitConfig) -> Result<()> {
        let strings_path = commit_config.folder_to_commit.join("strings");
        self.string_index.commit(strings_path)?;

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Committed {
    pub epoch: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionDescriptorDump {
    pub id: CollectionId,
    pub fields: Vec<(String, (FieldId, TypedField))>,
    pub used_models: Vec<(String, Vec<FieldId>)>,
}
