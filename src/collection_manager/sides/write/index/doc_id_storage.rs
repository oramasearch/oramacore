use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};

use crate::{
    file_utils::{create_if_not_exists, BufferedFile},
    types::DocumentId,
};

const DOC_ID_STORAGE_FILE_NAME: &str = "doc_id_storage.bin";

#[derive(Debug, Default)]
pub struct DocIdStorage {
    document_ids: HashMap<String, DocumentId>,
}

impl DocIdStorage {
    pub fn empty() -> Self {
        Self {
            document_ids: HashMap::new(),
        }
    }

    pub fn get(&self, doc_id: &str) -> Option<DocumentId> {
        self.document_ids.get(doc_id).copied()
    }

    pub fn remove_document_ids(&mut self, doc_ids: Vec<String>) -> Vec<DocumentId> {
        doc_ids
            .into_iter()
            .filter_map(|doc_id| self.document_ids.remove(&doc_id))
            .collect()
    }

    #[must_use]
    pub fn insert_document_id(
        &mut self,
        doc_id: String,
        document_id: DocumentId,
    ) -> Option<DocumentId> {
        self.document_ids.insert(doc_id, document_id)
    }

    pub fn get_document_ids(&self) -> impl Iterator<Item = DocumentId> + '_ {
        self.document_ids.values().copied()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.document_ids.len()
    }

    pub fn commit(&self, data_dir: PathBuf) -> Result<()> {
        create_if_not_exists(&data_dir)
            .context("Cannot create the base directory for the doc id storage")?;

        let file_path = data_dir.join(DOC_ID_STORAGE_FILE_NAME);
        BufferedFile::create_or_overwrite(file_path)
            .context("Cannot create file")?
            .write_bincode_data(&self.document_ids)
            .context("Cannot write map to file")?;

        Ok(())
    }

    pub fn load(data_dir: PathBuf) -> Result<Self> {
        let file_path = data_dir.join(DOC_ID_STORAGE_FILE_NAME);
        let document_id: HashMap<String, DocumentId> = BufferedFile::open(file_path)
            .context("Cannot open file")?
            .read_bincode_data()
            .context("Cannot read doc_id_storage from file")?;

        Ok(Self {
            document_ids: document_id,
        })
    }
}
