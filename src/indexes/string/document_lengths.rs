use std::{collections::HashMap, io::Write, path::PathBuf};

use anyhow::{Context, Result};
use tracing::{debug, warn};

use crate::document_storage::DocumentId;


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

        let mut file = std::fs::File::create(new_path)
            .context("Cannot create file")?;
        serde_json::to_writer(&mut file, &lengths)
            .context("Cannot write to file")?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    }

    pub fn get_length(&self, doc_id: &DocumentId) -> Result<u32> {
        let file = std::fs::File::open(&self.path)
            .context("Cannot oper file")?;
        let content: HashMap<DocumentId, u32> = serde_json::from_reader(file)
            .context("Cannot decode from file")?;

        let length = content.get(doc_id).unwrap_or(&1);
        Ok(*length)
    }

    pub fn merge(&self, lengths: &HashMap<DocumentId, u32>, new_path: PathBuf) -> Result<()> {
        let file = std::fs::File::open(&self.path)
            .context("Cannot oper file")?;
        let mut content: HashMap<DocumentId, u32> = serde_json::from_reader(file)
            .context("Cannot decode from file")?;

        for (doc_id, length) in lengths {
            if let Some(_) = content.insert(*doc_id, *length) {
                warn!("Document length already exists for doc {:?}, overwrite", doc_id);
            }
        }

        let mut file = std::fs::File::create(new_path)
            .context("Cannot create file")?;
        serde_json::to_writer(&mut file, &content)
            .context("Cannot write to file")?;
        file.flush()?;
        file.sync_all()?;

        Ok(())
    }
}
