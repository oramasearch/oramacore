use async_trait::async_trait;
use std::{collections::HashMap, fmt::Debug, fs::File, path::PathBuf, sync::RwLock};
use tracing::warn;

use anyhow::{anyhow, Context, Ok, Result};
use dashmap::DashMap;

use crate::{document_storage::DocumentId, types::Document};

#[async_trait]
pub trait DocumentStorage: Sync + Send + Debug {
    async fn add_document(&self, doc_id: DocumentId, doc: Document) -> Result<()>;

    async fn get_documents_by_ids(&self, doc_ids: Vec<DocumentId>)
        -> Result<Vec<Option<Document>>>;

    async fn get_total_documents(&self) -> Result<usize>;

    fn commit(&self, path: PathBuf) -> Result<()>;

    fn load(&mut self, path: PathBuf) -> Result<()>;
}

#[derive(Debug)]
pub struct InMemoryDocumentStorage {
    documents: DashMap<DocumentId, Document>,
}
impl Default for InMemoryDocumentStorage {
    fn default() -> Self {
        Self::new()
    }
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

    async fn get_total_documents(&self) -> Result<usize> {
        Ok(self.documents.len())
    }

    fn commit(&self, path: PathBuf) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, path: PathBuf) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct CommittedDiskDocumentStorage {
    path: PathBuf,
}
impl CommittedDiskDocumentStorage {
    fn get_documents_by_ids(&self, doc_ids: &[DocumentId]) -> Result<Vec<Option<Document>>> {
        let mut result = Vec::with_capacity(doc_ids.len());
        for id in doc_ids {
            let doc_path = self.path.join(format!("{}", id.0));
            match std::fs::exists(&doc_path) {
                Err(e) => {
                    return Err(anyhow!(
                        "Error while checking if the document exists: {:?}",
                        e
                    ));
                }
                std::result::Result::Ok(false) => {
                    result.push(None);
                    continue;
                }
                std::result::Result::Ok(true) => {}
            };
            let doc_file = File::open(doc_path)?;
            let doc: Document = serde_json::from_reader(doc_file)?;
            result.push(Some(doc));
        }

        Ok(result)
    }

    fn get_total_documents(&self) -> Result<usize> {
        let mut total = 0;
        for entry in std::fs::read_dir(&self.path)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                total += 1;
            }
        }
        Ok(total)
    }

    fn add(&self, docs: Vec<(DocumentId, Document)>) {
        for (doc_id, doc) in docs {
            let doc_path = self.path.join(format!("{}", doc_id.0));
            let doc_file = File::create(doc_path).unwrap();
            serde_json::to_writer(doc_file, &doc).unwrap();
        }
    }
}

#[derive(Debug)]
pub struct DiskDocumentStorage {
    uncommitted: RwLock<HashMap<DocumentId, Document>>,
    committed: CommittedDiskDocumentStorage,
}

impl DiskDocumentStorage {
    pub fn try_new(doc_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&doc_dir).context("Cannot create document directory")?;

        Ok(Self {
            uncommitted: Default::default(),
            committed: CommittedDiskDocumentStorage { path: doc_dir },
        })
    }
}

#[async_trait]
impl DocumentStorage for DiskDocumentStorage {
    async fn add_document(&self, doc_id: DocumentId, doc: Document) -> Result<()> {
        let mut uncommitted = match self.uncommitted.write() {
            std::result::Result::Ok(uncommitted) => uncommitted,
            std::result::Result::Err(e) => e.into_inner(),
        };
        if uncommitted.insert(doc_id, doc).is_some() {
            warn!("Document {:?} already exists. Overwritten.", doc_id);
        }
        Ok(())
    }

    async fn get_documents_by_ids(
        &self,
        doc_ids: Vec<DocumentId>,
    ) -> Result<Vec<Option<Document>>> {
        let committed = self.committed.get_documents_by_ids(&doc_ids)?;

        let uncommitted = match self.uncommitted.read() {
            std::result::Result::Ok(uncommitted) => uncommitted,
            std::result::Result::Err(e) => e.into_inner(),
        };
        let uncommitted: Vec<_> = doc_ids
            .into_iter()
            .map(|doc_id| uncommitted.get(&doc_id).cloned())
            .collect();

        let result = committed
            .into_iter()
            .zip(uncommitted)
            .map(|(committed, uncommitted)| {
                if let Some(doc) = uncommitted {
                    Some(doc)
                } else {
                    committed
                }
            })
            .collect();

        Ok(result)
    }

    async fn get_total_documents(&self) -> Result<usize> {
        let mut total = self.committed.get_total_documents()?;
        let uncommitted = match self.uncommitted.read() {
            std::result::Result::Ok(uncommitted) => uncommitted,
            std::result::Result::Err(e) => e.into_inner(),
        };
        total += uncommitted.len();
        Ok(total)
    }

    fn commit(&self, path: PathBuf) -> Result<()> {
        let mut uncommitted = match self.uncommitted.write() {
            std::result::Result::Ok(uncommitted) => uncommitted,
            std::result::Result::Err(e) => e.into_inner(),
        };
        let uncommitted: Vec<_> = uncommitted.drain().collect();

        // This implementation differs from the indexes one.
        // In fact, in index, we "merge" committed data with uncommitted data.
        // This is because the inner datastructure wants to have ordered data.
        // Here instead, we could avoid to perform a similar approach.
        // So, we can just "add a new file" to the committed storage.
        // TODO: verify if this is ok for consistency PoV.
        self.committed.add(uncommitted);

        Ok(())
    }

    fn load(&mut self, path: PathBuf) -> Result<()> {
        self.committed = CommittedDiskDocumentStorage { path };
        Ok(())
    }
}
