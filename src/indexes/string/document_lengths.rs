use std::{collections::HashMap, path::PathBuf, sync::RwLock};

use anyhow::{Context, Result};
use tracing::{debug, warn};

use crate::{file_utils::BufferedFile, types::DocumentId};

type Content = HashMap<DocumentId, u32>;

#[derive(Default, Debug)]
pub struct DocumentLengthsPerDocument {
    path: PathBuf,
    content: RwLock<Content>,
}
impl DocumentLengthsPerDocument {
    pub fn try_new(path: PathBuf) -> Result<Self> {
        let content: Content = BufferedFile::open(&path)
            .context("Cannot open document length file")?
            .read_json_data()
            .context("Cannot deserialize document length")?;

        Ok(Self {
            path,
            content: RwLock::new(content),
        })
    }

    pub fn create(lengths: &HashMap<DocumentId, u32>, new_path: PathBuf) -> Result<()> {
        debug!("Creating new document lengths file {:?}", new_path);

        BufferedFile::create(new_path)
            .context("Cannot create file")?
            .write_json_data(lengths)
            .context("Cannot serialize to file")?;

        Ok(())
    }

    pub fn get_length(&self, doc_id: &DocumentId) -> Result<u32> {
        let lock = match self.content.read() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        Ok(*lock.get(doc_id).unwrap_or(&1))
    }

    pub fn merge(&self, lengths: &HashMap<DocumentId, u32>, new_path: PathBuf) -> Result<()> {
        let mut content: HashMap<DocumentId, u32> = BufferedFile::open(&self.path)
            .context("Cannot open document length file")?
            .read_json_data()
            .context("Cannot deserialize document length")?;

        for (doc_id, length) in lengths {
            if content.insert(*doc_id, *length).is_some() {
                warn!(
                    "Document length already exists for doc {:?}, overwrite",
                    doc_id
                );
            }
        }

        BufferedFile::create(new_path)
            .context("Cannot create file")?
            .write_json_data(&content)
            .context("Cannot serialize to file")?;

        Ok(())
    }
}
