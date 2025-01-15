use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use dashmap::DashMap;

use crate::{file_utils::BufferedFile, types::DocumentId};

pub struct LoadedPostingIdStorage {
    content: HashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
}
impl LoadedPostingIdStorage {
    pub fn get_posting(&self, posting_id: &u64) -> Result<Option<&Vec<(DocumentId, Vec<usize>)>>> {
        Ok(self.content.get(posting_id))
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PostingListId(pub u32);

#[derive(Default, Debug)]
pub struct PostingIdStorage {
    pub(super) path: PathBuf,
}
impl PostingIdStorage {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn create(
        content: HashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
        new_path: PathBuf,
    ) -> Result<()> {
        BufferedFile::create(new_path)
            .context("Cannot create posting id file")?
            .write_json_data(&content)
            .context("Cannot serialize posting id")?;

        Ok(())
    }

    pub fn load(&self) -> Result<LoadedPostingIdStorage> {
        let content: HashMap<u64, Vec<(DocumentId, Vec<usize>)>> = BufferedFile::open(&self.path)
            .context("Cannot open posting ids file")?
            .read_json_data()
            .context("Cannot deserialize posting ids")?;

        Ok(LoadedPostingIdStorage { content })
    }

    pub fn apply_delta(
        &self,
        delta: DashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
        new_path: PathBuf,
    ) -> Result<()> {
        let mut content: HashMap<u64, Vec<(DocumentId, Vec<usize>)>> =
            BufferedFile::open(&self.path)
                .context("Cannot open posting ids file")?
                .read_json_data()
                .context("Cannot deserialize posting ids")?;

        for (posting_id, posting) in delta {
            let entry = content.entry(posting_id).or_default();
            entry.extend(posting);
        }

        BufferedFile::create(new_path)
            .context("Cannot create posting id file")?
            .write_json_data(&content)
            .context("Cannot serialize posting id")?;

        Ok(())
    }
}
