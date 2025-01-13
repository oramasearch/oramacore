use async_trait::async_trait;
use std::{collections::HashMap, fmt::Debug, path::PathBuf, sync::RwLock};
use tracing::{debug, warn};

use anyhow::{anyhow, Context, Ok, Result};

use crate::{
    file_utils::BufferedFile,
    types::{Document, DocumentId},
};

#[async_trait]
pub trait DocumentStorage: Sync + Send + Debug {
    async fn add_document(&self, doc_id: DocumentId, doc: Document) -> Result<()>;

    async fn get_documents_by_ids(&self, doc_ids: Vec<DocumentId>)
        -> Result<Vec<Option<Document>>>;

    async fn get_total_documents(&self) -> Result<usize>;

    fn commit(&self) -> Result<()>;

    fn load(&mut self) -> Result<()>;
}

// The `CommittedDiskDocumentStorage` implementation is not optimal.
// Defenitely, we cannot read every time from disk, it is too heavy.
// We should be backed by the disk, but we should also have a cache in memory.
// We should find a good balance between memory and disk usage.
// TODO: think about a better implementation.

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
                    warn!("Document {:?} not found", id);
                    result.push(None);
                    continue;
                }
                std::result::Result::Ok(true) => {}
            };

            let doc: Document = BufferedFile::open(doc_path)
                .context("Cannot open document file")?
                .read_json_data()
                .context("Cannot read document data")?;

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

    fn add(&self, docs: Vec<(DocumentId, Document)>) -> Result<()> {
        for (doc_id, doc) in docs {
            let doc_path = self.path.join(format!("{}", doc_id.0));

            BufferedFile::create(doc_path)
                .context("Cannot create document file")?
                .write_json_data(&doc)
                .context("Cannot write document data")?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct DocumentStorageConfig {
    pub data_dir: PathBuf,
}

#[derive(Debug)]
pub struct DiskDocumentStorage {
    uncommitted: RwLock<HashMap<DocumentId, Document>>,
    committed: CommittedDiskDocumentStorage,
}

impl DiskDocumentStorage {
    pub fn try_new(config: DocumentStorageConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir).context("Cannot create document directory")?;

        Ok(Self {
            uncommitted: Default::default(),
            committed: CommittedDiskDocumentStorage {
                path: config.data_dir,
            },
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

    #[tracing::instrument(skip(self))]
    async fn get_documents_by_ids(
        &self,
        doc_ids: Vec<DocumentId>,
    ) -> Result<Vec<Option<Document>>> {
        debug!("Get documents");
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
                    if committed.is_none() {
                        warn!("Document not found");
                    }

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

    fn commit(&self) -> Result<()> {
        // This implementation is wrong:
        // in the mean time we "dran" + "collection" + "write on FS"
        // The documents aren't reachable. So the search output will not contain them.
        // We should follow the same path of the indexes.

        let mut uncommitted = match self.uncommitted.write() {
            std::result::Result::Ok(uncommitted) => uncommitted,
            std::result::Result::Err(e) => e.into_inner(),
        };
        let uncommitted: Vec<_> = uncommitted.drain().collect();

        self.committed
            .add(uncommitted)
            .context("Cannot commit documents")?;

        Ok(())
    }

    fn load(&mut self) -> Result<()> {
        Ok(())
    }
}
