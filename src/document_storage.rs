use std::sync::{atomic::AtomicU64, Arc};

use anyhow::Result;
use dashmap::DashMap;

use crate::types::{Document, DocumentId};

#[derive(Debug, Default)]
pub struct DocumentStorage {
    storage: DashMap<DocumentId, Document>,
    id_generator: Arc<AtomicU64>,
}

impl DocumentStorage {
    pub fn new(id_generator: Arc<AtomicU64>) -> Self {
        Self {
            storage: Default::default(),
            id_generator,
        }
    }

    pub fn generate_document_id(&self) -> DocumentId {
        let id = self
            .id_generator
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        DocumentId(id)
    }

    pub fn add_documents(&self, documents: Vec<(DocumentId, Document)>) -> Result<()> {
        for (doc_id, doc) in documents {
            self.storage.insert(doc_id, doc);
        }

        Ok(())
    }

    pub fn get_all(&self, doc_ids: Vec<DocumentId>) -> Result<Vec<Option<Document>>> {
        let docs = doc_ids
            .into_iter()
            // TODO: Avoid this clone
            .map(|doc_id| self.storage.get(&doc_id).map(|x| x.value().clone()))
            .collect();

        Ok(docs)
    }

    pub fn get(&self, doc_id: DocumentId) -> Result<Option<Document>> {
        // TODO: Avoid this clone
        Ok(self.storage.get(&doc_id).map(|x| x.value().clone()))
    }
}
