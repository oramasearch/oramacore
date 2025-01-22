use std::{collections::HashMap, path::PathBuf, sync::RwLock};

use anyhow::{Context, Result};
use dashmap::DashMap;

use crate::{file_utils::BufferedFile, types::DocumentId};

type Content = HashMap<u64, Vec<(DocumentId, Vec<usize>)>>;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PostingListId(pub u32);

#[derive(Default, Debug)]
pub struct PostingIdStorage {
    content: RwLock<Content>,
}
impl PostingIdStorage {
    pub fn try_new(path: PathBuf) -> Result<Self> {
        let content: Content = BufferedFile::open(&path)
            .context("Cannot open posting ids file")?
            .read_json_data()
            .context("Cannot deserialize posting ids")?;

        Ok(Self {
            content: RwLock::new(content),
        })
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

    #[allow(clippy::type_complexity)]
    pub fn get_posting(&self, posting_id: &u64) -> Result<Option<Vec<(DocumentId, Vec<usize>)>>> {
        let lock = match self.content.read() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        Ok(lock.get(posting_id).cloned())
    }

    pub fn apply_delta(
        &self,
        delta: DashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
        new_path: PathBuf,
    ) -> Result<()> {
        let mut lock = match self.content.write() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        for (posting_id, posting) in delta {
            let entry = lock.entry(posting_id).or_default();
            entry.extend(posting);
        }

        BufferedFile::create(new_path)
            .context("Cannot create posting id file")?
            .write_json_data(&*lock)
            .context("Cannot serialize posting id")?;

        Ok(())
    }
}
