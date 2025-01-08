use std::{collections::HashMap, io::Write, path::PathBuf};

use anyhow::{Context, Result};
use dashmap::DashMap;

use crate::types::DocumentId;

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
        let mut file = std::fs::File::create(new_path).context("Cannot create file")?;
        serde_json::to_writer(&mut file, &content).context("Cannot write to file")?;
        file.flush()?;
        file.sync_all()?;

        Ok(())
    }

    pub fn get_posting(&self, posting_id: u64) -> Result<Option<Vec<(DocumentId, Vec<usize>)>>> {
        let file = std::fs::File::open(&self.path).context("Cannot open file")?;
        let content: HashMap<u64, Vec<(DocumentId, Vec<usize>)>> =
            serde_json::from_reader(file).context("Cannot decode from file")?;

        Ok(content.get(&posting_id).cloned())
    }

    pub fn apply_delta(
        &self,
        delta: DashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
        new_path: PathBuf,
    ) -> Result<()> {
        let file = std::fs::File::open(&self.path).context("Cannot open file")?;
        let mut content: HashMap<u64, Vec<(DocumentId, Vec<usize>)>> =
            serde_json::from_reader(file).context("Cannot decode from file")?;

        for (posting_id, posting) in delta {
            let entry = content.entry(posting_id).or_default();
            entry.extend(posting);
        }

        let mut file = std::fs::File::create(new_path).context("Cannot create file")?;
        serde_json::to_writer(&mut file, &content).context("Cannot write to file")?;
        file.flush()?;
        file.sync_all()?;

        Ok(())
    }
}
