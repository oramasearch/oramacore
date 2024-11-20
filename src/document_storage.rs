use std::{collections::HashMap, sync::{atomic::AtomicU64, Arc}};

use anyhow::Result;
use tokio::sync::RwLock;

use crate::types::{Document, DocumentId};

#[derive(Debug, Default)]
pub struct DocumentStorage {
    storage: RwLock<HashMap<DocumentId, Document>>,
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

    pub async fn add_documents(&self, documents: Vec<(DocumentId, Document)>) -> Result<()> {
        let mut storage = self.storage.write().await;
        for (doc_id, doc) in documents {
            storage.insert(doc_id, doc);
        }

        Ok(())
    }

    pub async fn get_all(&self, doc_ids: Vec<DocumentId>) -> Result<Vec<Option<Document>>> {
        let storage = self.storage.read().await;
        let docs = doc_ids
            .into_iter()
            // TODO: Avoid the `clone`
            .map(|doc_id| storage.get(&doc_id).cloned())
            .collect();

        Ok(docs)
    }

    pub async fn get(&self, doc_id: DocumentId) -> Result<Option<Document>> {
        let storage = self.storage.read().await;
        // TODO: Avoid the `clone`
        Ok(storage.get(&doc_id).cloned())
    }
}
