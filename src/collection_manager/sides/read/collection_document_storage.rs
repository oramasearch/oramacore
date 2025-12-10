use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use oramacore_lib::fs::create_if_not_exists;
use serde::{Deserialize, Serialize};
use tracing::info;
use zebo::Zebo;

use crate::{
    collection_manager::sides::{
        read::document_storage::{DocumentStorage, ZeboDocument},
        DocumentStorageWriteOperation, Offset,
    },
    lock::{OramaAsyncLock, OramaSyncLock},
    metrics::{commit::COLLECTION_DOCUMENT_COMMIT_CALCULATION_TIME, CollectionCommitLabels},
    types::{CollectionId, DocumentId, RawJSONDocument},
};

// 1GB
const PAGE_SIZE: u64 = 1024 * 1024 * 1024;

struct UncommittedDocumentStorage {
    storage: OramaSyncLock<Vec<(DocumentId, Arc<RawJSONDocument>)>>,
}

impl UncommittedDocumentStorage {
    pub fn new() -> Self {
        Self {
            storage: OramaSyncLock::new("uncommitted_document_storage", Vec::new()),
        }
    }

    pub fn insert(&self, doc_id: DocumentId, doc: Arc<RawJSONDocument>) -> Result<()> {
        let mut storage_lock = self
            .storage
            .write("insert")
            .expect("Cannot lock uncommitted storage for write");
        storage_lock.push((doc_id, doc));
        Ok(())
    }

    pub fn insert_many<I: Iterator<Item = (DocumentId, Arc<RawJSONDocument>)>>(
        &self,
        items: I,
    ) -> Result<()> {
        let mut storage_lock = self
            .storage
            .write("insert")
            .expect("Cannot lock uncommitted storage for write");
        storage_lock.extend(items);
        Ok(())
    }

    pub fn get_documents_by_ids(
        &self,
        ids: &[DocumentId],
        results: &mut Vec<(DocumentId, Arc<RawJSONDocument>)>,
    ) -> Result<()> {
        let storage_lock = self
            .storage
            .read("get_documents_by_ids")
            .expect("Cannot lock uncommitted storage for read");

        for id in ids {
            let doc_opt = storage_lock
                .iter()
                .find(|(stored_id, _)| *stored_id == *id)
                .map(|(_, doc)| doc.clone());
            if let Some(doc) = &doc_opt {
                results.push((*id, doc.clone()));
            }
        }

        Ok(())
    }

    pub fn clone_inner(&self) -> Vec<(DocumentId, Arc<RawJSONDocument>)> {
        let storage_lock = self
            .storage
            .read("clone_inner")
            .expect("Cannot lock uncommitted storage for clone");
        storage_lock.clone()
    }

    pub fn clear(&self) {
        let mut storage_lock = self
            .storage
            .write("clear")
            .expect("Cannot lock uncommitted storage for clean");
        // NB: release also the memory, not only clear the vector
        **storage_lock = Vec::new();
    }
}

pub struct CollectionDocumentStorage {
    uncommitted: UncommittedDocumentStorage,
    zebo: Arc<OramaAsyncLock<Zebo<1_000_000, PAGE_SIZE, DocumentId>>>,
    global_document_storage: Arc<DocumentStorage>,
    first_non_global_doc_id: OramaSyncLock<DocumentId>,
    deleted_documents: OramaAsyncLock<Vec<DocumentId>>,
    collection_id: CollectionId,
}

impl CollectionDocumentStorage {
    pub fn new(
        global_document_storage: Arc<DocumentStorage>,
        base_dir: PathBuf,
        first_non_global_doc_id: DocumentId,
        collection_id: CollectionId,
    ) -> Result<Self> {
        let zebo_dir = base_dir.join("zebo");
        create_if_not_exists(&zebo_dir).context("Cannot create zebo directory")?;

        let zebo = Zebo::try_new(zebo_dir).context("Cannot create zebo")?;
        let zebo = Arc::new(OramaAsyncLock::new("zebo", zebo));

        Ok(Self {
            zebo,
            global_document_storage,
            first_non_global_doc_id: OramaSyncLock::new(
                "first_non_global_doc_id",
                first_non_global_doc_id,
            ),
            uncommitted: UncommittedDocumentStorage::new(),

            deleted_documents: OramaAsyncLock::new("deleted_documents", Vec::new()),

            collection_id,
        })
    }

    pub async fn update(&self, op: DocumentStorageWriteOperation) -> Result<()> {
        match op {
            DocumentStorageWriteOperation::InsertDocument { doc_id, doc } => {
                self.uncommitted.insert(doc_id, Arc::new(doc.0))?;
            }
            DocumentStorageWriteOperation::InsertDocuments(docs) => {
                if cfg!(test) && !docs.is_sorted_by(|a, b| a.0 < b.0) {
                    panic!("Add documents are not in order");
                }

                self.uncommitted.insert_many(
                    docs.into_iter()
                        .map(|(doc_id, doc)| (doc_id, Arc::new(doc.0))),
                )?;
            }
            DocumentStorageWriteOperation::DeleteDocuments { doc_ids } => {
                let mut lock = self.deleted_documents.write("delete_documents").await;
                lock.extend(doc_ids);
            }
        }
        Ok(())
    }

    fn split_into_uncommitted_local_and_global(
        &self,
        ids: Vec<DocumentId>,
    ) -> (Vec<DocumentId>, Vec<DocumentId>, Vec<DocumentId>) {
        let first_non_global_doc_id = self
            .first_non_global_doc_id
            .read("split_into_local_and_global")
            .unwrap();
        let (global_ids, local_ids): (Vec<DocumentId>, Vec<DocumentId>) =
            ids.into_iter().partition(|id| {
                if first_non_global_doc_id.0 == 0 {
                    return false;
                }
                *id < **first_non_global_doc_id
            });
        drop(first_non_global_doc_id);

        let uncommitted = self
            .uncommitted
            .storage
            .read("split_into_uncommitted_local_and_global")
            .expect("Cannot lock uncommitted storage for split");
        let first_uncommitted = uncommitted.first().map(|(id, _)| id.0);

        let (uncommitted_ids, committed_ids) = local_ids.into_iter().partition(|id| {
            if let Some(first_uncommitted) = first_uncommitted {
                return id.0 >= first_uncommitted;
            }
            false
        });

        (uncommitted_ids, committed_ids, global_ids)
    }

    pub async fn get_documents_by_ids(
        &self,
        mut ids: Vec<DocumentId>,
    ) -> Result<Vec<(DocumentId, Arc<RawJSONDocument>)>> {
        let mut results: Vec<(DocumentId, Arc<RawJSONDocument>)> = Vec::with_capacity(ids.len());

        let uncommitted_document_deletions =
            self.deleted_documents.read("get_documents_by_ids").await;
        ids.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
        drop(uncommitted_document_deletions);

        let (uncommitted_ids, local_ids, global_ids) =
            self.split_into_uncommitted_local_and_global(ids);

        // Get global documents
        {
            self.global_document_storage
                .get_documents_by_ids(global_ids, &mut results)
                .await?;
        }

        // Get local documents
        {
            let zebo_lock = self.zebo.read("get_documents_by_ids").await;
            let local_documents = zebo_lock.get_documents(local_ids)?.filter_map(|r| {
                let (id, doc) = r.ok()?;
                let doc = ZeboDocument::from_bytes(doc).ok()?;
                let raw = doc.as_raw_json_doc().ok()?;

                Some((id, raw))
            });
            results.extend(local_documents);
            drop(zebo_lock);
        }

        // Get uncommitted documents
        {
            self.uncommitted
                .get_documents_by_ids(&uncommitted_ids, &mut results)?;
        }

        Ok(results)
    }

    pub async fn commit(&self, _offset: Offset) -> Result<()> {
        info!("Commit colletion documents");

        let m = COLLECTION_DOCUMENT_COMMIT_CALCULATION_TIME.create(CollectionCommitLabels {
            collection: self.collection_id.to_string(),
            side: "read",
        });

        let uncommitted_docs = self.uncommitted.clone_inner();

        let uncommitted_document_deletions_lock = self.deleted_documents.read("commit").await;
        let to_delete = uncommitted_document_deletions_lock.clone();
        drop(uncommitted_document_deletions_lock);

        let mut zebo = self.zebo.write("commit").await;
        let docs: Vec<_> = uncommitted_docs
            .into_iter()
            .map(|(doc_id, doc)| (doc_id, ZeboDocument::FromJSONDoc(doc)))
            .collect();
        if !docs.is_empty() {
            zebo.reserve_space_for(&docs)
                .context("Cannot reserve space for documents")?
                .write_all()
                .context("Cannot write documents")?;
        }
        zebo.remove_documents(to_delete, false)
            .context("Cannot remove documents")?;

        self.uncommitted.clear();

        let mut deleted_documents_lock = self.deleted_documents.write("commit").await;
        deleted_documents_lock.clear();
        drop(deleted_documents_lock);

        drop(m);

        // NB: we don't commit the global document storage here, it is done at a higher level

        info!("Collection documents committed");

        Ok(())
    }
}

#[derive(Deserialize, Serialize, Debug)]
enum Dump {
    V1(DumpV1),
}

#[derive(Deserialize, Serialize, Debug)]
struct DumpV1 {
    document_id: u64,
}
