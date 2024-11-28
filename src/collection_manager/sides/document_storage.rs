use async_trait::async_trait;
use std::fmt::Debug;

use anyhow::{Ok, Result};
use dashmap::DashMap;

use crate::{document_storage::DocumentId, types::Document};

#[async_trait]
pub trait DocumentStorage: Sync + Send + Debug {
    async fn add_document(&self, doc_id: DocumentId, doc: Document) -> Result<()>;

    async fn get_documents_by_ids(&self, doc_ids: Vec<DocumentId>)
        -> Result<Vec<Option<Document>>>;
}

#[derive(Debug)]
pub struct InMemoryDocumentStorage {
    documents: DashMap<DocumentId, Document>,
}
impl InMemoryDocumentStorage {
    pub fn new() -> Self {
        Self {
            documents: Default::default(),
        }
    }
}

#[async_trait]
impl DocumentStorage for InMemoryDocumentStorage {
    async fn add_document(&self, doc_id: DocumentId, doc: Document) -> Result<()> {
        self.documents.insert(doc_id, doc);
        Ok(())
    }

    async fn get_documents_by_ids(
        &self,
        doc_ids: Vec<DocumentId>,
    ) -> Result<Vec<Option<Document>>> {
        let docs = doc_ids
            .into_iter()
            .map(|doc_id| self.documents.get(&doc_id).map(|doc| doc.value().clone()))
            .collect();
        Ok(docs)
    }
}
