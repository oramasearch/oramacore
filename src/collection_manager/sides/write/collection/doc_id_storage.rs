use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};

use crate::{
    file_utils::{create_if_not_exists, BufferedFile},
    types::DocumentId,
};

const DOC_ID_STORAGE_FILE_NAME: &str = "doc_id_storage.bin";

#[derive(Debug, Default)]
pub struct DocIdStorage {
    document_id: HashMap<String, DocumentId>,
}

impl DocIdStorage {
    pub fn remove_document_id(&mut self, doc_id: Vec<String>) -> Vec<DocumentId> {
        doc_id
            .into_iter()
            .filter_map(|doc_id| self.document_id.remove(&doc_id))
            .collect()
    }

    pub fn insert_document_id(&mut self, doc_id: String, document_id: DocumentId) {
        self.document_id.insert(doc_id, document_id);
    }

    pub fn commit(&self, data_dir: PathBuf) -> Result<()> {
        create_if_not_exists(&data_dir)
            .context("Cannot create the base directory for the doc id storage")?;

        let file_path = data_dir.join(DOC_ID_STORAGE_FILE_NAME);
        BufferedFile::create_or_overwrite(file_path)
            .context("Cannot create file")?
            .write_bincode_data(&self.document_id)
            .context("Cannot write map to file")?;

        Ok(())
    }

    pub fn load(data_dir: PathBuf) -> Result<Self> {
        let file_path = data_dir.join(DOC_ID_STORAGE_FILE_NAME);
        let document_id: HashMap<String, DocumentId> = BufferedFile::open(file_path)
            .context("Cannot open file")?
            .read_bincode_data()
            .context("Cannot read doc_id_storage from file")?;

        Ok(Self { document_id })
    }
}
