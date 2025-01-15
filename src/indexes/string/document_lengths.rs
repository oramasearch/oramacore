use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use tracing::{debug, warn};

use crate::{file_utils::BufferedFile, types::DocumentId};

pub struct LoadedDocumentLengths {
    content: HashMap<DocumentId, u32>,
}
impl LoadedDocumentLengths {
    pub fn get_length(&self, doc_id: &DocumentId) -> Result<u32> {
        let length = self.content.get(doc_id).unwrap_or(&1);
        Ok(*length)
    }
}

#[derive(Default, Debug)]
pub struct DocumentLengthsPerDocument {
    pub(super) path: PathBuf,
}
impl DocumentLengthsPerDocument {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn create(lengths: &HashMap<DocumentId, u32>, new_path: PathBuf) -> Result<()> {
        debug!("Creating new document lengths file {:?}", new_path);

        BufferedFile::create(new_path)
            .context("Cannot create file")?
            .write_json_data(lengths)
            .context("Cannot serialize to file")?;

        Ok(())
    }

    pub fn load(&self) -> Result<LoadedDocumentLengths> {
        let content: HashMap<DocumentId, u32> = BufferedFile::open(&self.path)
            .context("Cannot open document length file")?
            .read_json_data()
            .context("Cannot deserialize document length")?;

        Ok(LoadedDocumentLengths { content })
    }

    /*
    pub fn get_length(&self, doc_id: &DocumentId) -> Result<u32> {
        let content: HashMap<DocumentId, u32> = BufferedFile::open(&self.path)
            .context("Cannot open document length file")?
            .read_json_data()
            .context("Cannot deserialize document length")?;

        let length = content.get(doc_id).unwrap_or(&1);
        Ok(*length)
    }
    */

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
