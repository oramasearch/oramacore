use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use debug_panic::debug_panic;
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};
use serde::{Deserialize, Serialize};
use tracing::{error, info};
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

    /// Takes a snapshot of the current uncommitted documents and returns them
    /// along with the snapshot length. The snapshot length is used later to drain
    /// only the snapshotted entries, preserving any concurrent inserts.
    pub fn snapshot(&self) -> (Vec<(DocumentId, Arc<RawJSONDocument>)>, usize) {
        let storage_lock = self
            .storage
            .read("snapshot")
            .expect("Cannot lock uncommitted storage for snapshot");
        let len = storage_lock.len();
        let data = storage_lock.clone();
        (data, len)
    }

    /// Drains only the first `snapshot_len` entries that were included in the snapshot.
    /// Entries appended concurrently after the snapshot was taken (at indices snapshot_len..)
    /// are preserved for the next commit cycle.
    pub fn drain_compacted(&self, snapshot_len: usize) {
        let mut storage_lock = self
            .storage
            .write("drain_compacted")
            .expect("Cannot lock uncommitted storage for drain");
        storage_lock.drain(..snapshot_len);
    }
}

pub struct CollectionDocumentStorage {
    uncommitted: UncommittedDocumentStorage,
    zebo: Arc<OramaAsyncLock<Zebo<1_000_000, PAGE_SIZE, DocumentId>>>,
    global_document_storage: Arc<DocumentStorage>,
    deleted_documents: OramaAsyncLock<Vec<DocumentId>>,
    collection_id: CollectionId,
    // DocIdStr -> DocumentId mapping for per-collection storage
    // NB: the lock is needed for a correct concurrent access
    doc_id_map: DocumentIdStrMap,
}

impl CollectionDocumentStorage {
    pub fn new(
        global_document_storage: Arc<DocumentStorage>,
        base_dir: PathBuf,
        collection_id: CollectionId,
    ) -> Result<Self> {
        let zebo_dir = base_dir.join("zebo");
        create_if_not_exists(&zebo_dir).context("Cannot create zebo directory")?;

        let mut zebo = Zebo::try_new(zebo_dir).context("Cannot create zebo")?;
        zebo.compact().context("Cannot compact zebo")?;
        let zebo = Arc::new(OramaAsyncLock::new("zebo", zebo));

        Ok(Self {
            zebo,
            global_document_storage,
            uncommitted: UncommittedDocumentStorage::new(),
            deleted_documents: OramaAsyncLock::new("deleted_documents", Vec::new()),
            collection_id,
            doc_id_map: DocumentIdStrMap::try_new(base_dir.join("doc_id_str_map.bin"))?,
        })
    }

    pub async fn update(&self, op: DocumentStorageWriteOperation) -> Result<()> {
        match op {
            // Old variants - handle without map updates
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
            // New variants - handle WITH map updates
            DocumentStorageWriteOperation::InsertDocumentWithDocIdStr {
                doc_id,
                doc_id_str,
                doc,
            } => {
                self.uncommitted.insert(doc_id, Arc::new(doc.0))?;

                self.doc_id_map
                    .update_doc_id_str_map(&[(doc_id, doc_id_str, ())])
                    .await
                    .context("Cannot update doc_id_str_map")?;
            }
            DocumentStorageWriteOperation::InsertDocumentsWithDocIdStr(docs) => {
                if cfg!(test) && !docs.is_sorted_by(|a, b| a.0 < b.0) {
                    panic!("Add documents are not in order");
                }

                self.doc_id_map
                    .update_doc_id_str_map(&docs)
                    .await
                    .context("Cannot update doc_id_str_map")?;

                self.uncommitted.insert_many(
                    docs.into_iter()
                        .map(|(doc_id, _, doc)| (doc_id, Arc::new(doc.0))),
                )?;
            }
            DocumentStorageWriteOperation::DeleteDocumentsWithDocIdStr(doc_id_pairs) => {
                // Extract doc_ids for deletion tracking
                let doc_ids: Vec<DocumentId> = doc_id_pairs.iter().map(|(id, _)| *id).collect();

                let mut lock = self.deleted_documents.write("delete_documents").await;
                lock.extend(doc_ids);
                drop(lock);

                self.doc_id_map
                    .delete_doc_id_str_map(&doc_id_pairs)
                    .await
                    .context("Cannot delete from doc_id_str_map")?;
            }
        }
        Ok(())
    }

    async fn split_into_uncommitted_local_and_global(
        &self,
        ids: Vec<DocumentId>,
    ) -> Result<(Vec<DocumentId>, Vec<DocumentId>, Vec<DocumentId>)> {
        let last_inserted_id = self
            .global_document_storage
            .get_last_inserted_document_id()
            .await
            .context("Cannot get last inserted document ID")?;

        let (global_ids, local_ids): (Vec<DocumentId>, Vec<DocumentId>) =
            if let Some(first_non_global_doc_id) = last_inserted_id {
                ids.into_iter()
                    .partition(|id| *id <= first_non_global_doc_id)
            } else {
                // All IDs are local
                (vec![], ids)
            };

        let uncommitted = self
            .uncommitted
            .storage
            .read("split_into_uncommitted_local_and_global")
            .expect("Cannot lock uncommitted storage for split");
        let first_uncommitted = uncommitted.first().map(|(id, _)| id.0);

        let (uncommitted_ids, committed_ids) = if let Some(first_uncommitted) = first_uncommitted {
            local_ids
                .into_iter()
                .partition(|id| id.0 >= first_uncommitted)
        } else {
            (vec![], local_ids)
        };

        Ok((uncommitted_ids, committed_ids, global_ids))
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
            self.split_into_uncommitted_local_and_global(ids).await?;

        // Get global documents
        {
            #[allow(deprecated)]
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

    /// Retrieves documents by their string document IDs.
    /// This method efficiently reads the doc_id_str_map once and then retrieves all matching documents.
    /// Missing document IDs are silently omitted from the result.
    pub async fn get_documents_by_str_ids(
        &self,
        doc_id_strs: Vec<String>,
    ) -> Result<HashMap<String, Arc<RawJSONDocument>>> {
        let map = self
            .doc_id_map
            .get_document_ids_by_str(doc_id_strs.as_slice())
            .await
            .context("Cannot get DocumentIds from doc_id_strs")?;

        let doc_ids: Vec<DocumentId> = map.values().cloned().collect();

        let documents = self.get_documents_by_ids(doc_ids).await?;
        let doc_id_doc_map: HashMap<_, _> = documents.into_iter().collect();

        let result: HashMap<_, _> = doc_id_strs
            .into_iter()
            .filter_map(|id_str| {
                map.get(&id_str)
                    .and_then(|doc_id| doc_id_doc_map.get(doc_id).map(|doc| (id_str, doc.clone())))
            })
            .collect();

        Ok(result)
    }

    pub async fn commit(&self, _offset: Offset) -> Result<()> {
        info!("Commit colletion documents");

        let m = COLLECTION_DOCUMENT_COMMIT_CALCULATION_TIME.create(CollectionCommitLabels {
            collection: self.collection_id.to_string(),
            side: "read",
        });

        // Snapshot uncommitted docs: record their count so we only drain what we commit.
        // Inserts that arrive concurrently (at indices snapshot_len..) are preserved.
        let (uncommitted_docs, docs_snapshot_len) = self.uncommitted.snapshot();

        // Snapshot deleted documents with the same position-based pattern.
        let deleted_documents_lock = self.deleted_documents.read("commit").await;
        let delete_snapshot_len = deleted_documents_lock.len();
        let to_delete = deleted_documents_lock.clone();
        drop(deleted_documents_lock);

        // Write to zebo — no locks held on uncommitted storage, so inserts can proceed.
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

        // Drain only the entries that were included in the snapshot.
        // Concurrent inserts appended after the snapshot are preserved for the next commit.
        self.uncommitted.drain_compacted(docs_snapshot_len);

        let mut deleted_documents_lock = self.deleted_documents.write("commit").await;
        deleted_documents_lock.drain(..delete_snapshot_len);
        drop(deleted_documents_lock);

        self.doc_id_map
            .commit()
            .await
            .context("Cannot commit doc_id_str_map")?;

        drop(m);

        // NB: we don't commit the global document storage here, it is done at a higher level

        info!("Collection documents committed");

        Ok(())
    }
}

#[derive(Clone)]
enum DocumentIdStrMapUncommittedChange {
    Insert(String, DocumentId),
    Delete(String, DocumentId),
}

struct DocumentIdStrMap {
    doc_id_str_map_path: OramaAsyncLock<PathBuf>,
    uncommitted: OramaAsyncLock<Vec<DocumentIdStrMapUncommittedChange>>,
}
impl DocumentIdStrMap {
    pub fn try_new(doc_id_str_map_path: PathBuf) -> Result<Self> {
        // Initialize empty map file if it does not exist
        if !doc_id_str_map_path.exists() {
            BufferedFile::create_or_overwrite(doc_id_str_map_path.clone())
                .with_context(|| format!("Cannot create or overwrite {doc_id_str_map_path:?}"))?
                .write_bincode_data(&DocumentIdStrMapDump::V1(
                    HashMap::<String, DocumentId>::new(),
                ))
                .context("Cannot serialize DocumentIdStrMap")?;
        }

        Ok(Self {
            doc_id_str_map_path: OramaAsyncLock::new("doc_id_str_map_path", doc_id_str_map_path),
            uncommitted: OramaAsyncLock::new("uncommitted", Vec::new()),
        })
    }

    async fn update_doc_id_str_map<T>(&self, docs: &[(DocumentId, String, T)]) -> Result<()> {
        let mut lock = self.uncommitted.write("update_doc_id_str_map").await;
        lock.reserve(docs.len());
        for (doc_id, doc_id_str, _) in docs {
            lock.push(DocumentIdStrMapUncommittedChange::Insert(
                doc_id_str.clone(),
                *doc_id,
            ));
        }
        drop(lock);

        Ok(())
    }

    async fn delete_doc_id_str_map(&self, doc_id_pairs: &[(DocumentId, String)]) -> Result<()> {
        let mut lock = self.uncommitted.write("update_doc_id_str_map").await;
        lock.reserve(doc_id_pairs.len());
        for (doc_id, doc_id_str) in doc_id_pairs {
            lock.push(DocumentIdStrMapUncommittedChange::Delete(
                doc_id_str.clone(),
                *doc_id,
            ));
        }
        drop(lock);

        Ok(())
    }

    async fn commit(&self) -> Result<()> {
        let lock = self.doc_id_str_map_path.write("commit").await;

        let mut map: HashMap<String, DocumentId> = BufferedFile::open(&**lock)
            .with_context(|| format!("Cannot open {:?}", **lock))?
            .read_bincode_data()
            .context("Cannot deserialize DocumentIdStrMap")?;

        // Snapshot the uncommitted changes: record their count so we only drain
        // what we process. Concurrent inserts at indices snapshot_len.. are preserved.
        let uncommitted_lock = self.uncommitted.read("commit_snapshot").await;
        let snapshot_len = uncommitted_lock.len();
        let snapshot: Vec<_> = uncommitted_lock
            .iter()
            .take(snapshot_len)
            .cloned()
            .collect();
        drop(uncommitted_lock);

        for change in snapshot {
            match change {
                DocumentIdStrMapUncommittedChange::Insert(doc_id_str, doc_id) => {
                    map.insert(doc_id_str, doc_id);
                }
                DocumentIdStrMapUncommittedChange::Delete(doc_id_str, doc_id) => {
                    if let Some(existing_doc_id) = map.get(&doc_id_str) {
                        if existing_doc_id != &doc_id {
                            debug_panic!(
                                "Trying to delete doc_id_str mapping for different DocumentId"
                            );
                            error!("Trying to delete doc_id_str mapping for different DocumentId");
                        } else {
                            map.remove(&doc_id_str);
                        }
                    }
                }
            }
        }

        BufferedFile::create_or_overwrite(lock.clone())
            .with_context(|| format!("Cannot create or overwrite {:?}", **lock))?
            .write_bincode_data(&map)
            .context("Cannot serialize DocumentIdStrMap")?;

        // Drain only the entries that were included in the snapshot.
        let mut uncommitted_lock = self.uncommitted.write("commit_drain").await;
        uncommitted_lock.drain(..snapshot_len);
        drop(uncommitted_lock);

        drop(lock);

        Ok(())
    }

    /// Get DocumentId from doc_id_str
    /// Returns None if the doc_id_str is not found in the map
    async fn get_document_ids_by_str(
        &self,
        doc_id_str: &[String],
    ) -> Result<HashMap<String, DocumentId>> {
        let mut ret = HashMap::with_capacity(doc_id_str.len());
        let file_path_lock = self.doc_id_str_map_path.write("insert_document").await;
        let f = std::fs::OpenOptions::new()
            .read(true)
            .open(&**file_path_lock)
            .context("Cannot open doc_id_str_map file")?;
        let map: HashMap<String, DocumentId> =
            bincode::deserialize_from(&f).context("Cannot deserialize doc_id_str_map")?;

        for doc_id_str in doc_id_str {
            if let Some(&doc_id) = map.get(doc_id_str) {
                ret.insert(doc_id_str.clone(), doc_id);
            }
        }

        let uncommitted_lock = self.uncommitted.read("get_document_ids_by_str").await;
        for change in &**uncommitted_lock {
            match change {
                DocumentIdStrMapUncommittedChange::Insert(doc_id_str_change, doc_id) => {
                    if doc_id_str.contains(doc_id_str_change) {
                        ret.insert(doc_id_str_change.clone(), *doc_id);
                    }
                }
                DocumentIdStrMapUncommittedChange::Delete(doc_id_str, doc_id) => {
                    if let Some(existing_doc_id) = map.get(doc_id_str) {
                        if existing_doc_id != doc_id {
                            debug_panic!(
                                "Trying to delete doc_id_str mapping for different DocumentId"
                            );
                            error!("Trying to delete doc_id_str mapping for different DocumentId");
                        } else {
                            ret.remove(doc_id_str);
                        }
                    }
                }
            }
        }
        drop(uncommitted_lock);
        drop(file_path_lock);

        Ok(ret)
    }
}

#[derive(Deserialize, Serialize, Debug)]
enum DocumentIdStrMapDump {
    V1(HashMap<String, DocumentId>),
}
