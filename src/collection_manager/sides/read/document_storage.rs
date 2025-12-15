use std::{borrow::Cow, fmt::Debug, path::PathBuf, sync::Arc, time::Duration};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, trace, warn};
use zebo::{Zebo, ZeboInfo};

#[allow(deprecated)]
use crate::collection_manager::sides::DocumentStorageWriteOperation;

use crate::{
    collection_manager::sides::DocumentToInsert,
    lock::OramaAsyncLock,
    metrics::{commit::DOCUMENT_COMMIT_CALCULATION_TIME, Empty},
    types::{DocumentId, RawJSONDocument},
};
use anyhow::{Context, Result};
use oramacore_lib::fs::{create_if_not_exists, read_file};

// 1 GB
static PAGE_SIZE: u64 = 1024 * 1024 * 1024;

#[derive(Debug)]
struct CommittedDiskDocumentStorage {
    zebo: Arc<OramaAsyncLock<Zebo<1_000_000, PAGE_SIZE, DocumentId>>>,
}
impl CommittedDiskDocumentStorage {
    async fn try_new(data_dir: PathBuf) -> Result<Self> {
        let zebo_dir = data_dir.join("zebo");
        create_if_not_exists(&zebo_dir).context("Cannot create zebo directory")?;

        let zebo_index_path = zebo_dir.join("index");
        let zebo = if std::fs::exists(zebo_index_path).unwrap_or(false) {
            info!("Zebo index exists");
            Zebo::try_new(zebo_dir).context("Cannot create zebo")?
        } else {
            warn!("Migrate documents to zebo");
            migrate_to_zebo(&data_dir)
                .await
                .context("Cannot migrate to zebo")?
        };

        let zebo = Arc::new(OramaAsyncLock::new("zebo", zebo));

        Ok(Self { zebo })
    }

    async fn get_documents_by_ids(
        &self,
        doc_ids: &[DocumentId],
        result: &mut Vec<(DocumentId, Arc<RawJSONDocument>)>,
    ) -> Result<()> {
        trace!(doc_len=?doc_ids.len(), "Read document");

        let zebo = self.zebo.read("get_documents_by_ids").await;
        let output = zebo
            .get_documents(doc_ids.to_vec())
            .context("Cannot get documents from zebo")?
            .filter_map(|r| {
                let (id, doc) = r.ok()?;
                let doc = ZeboDocument::from_bytes(doc).ok()?;
                let doc = doc.as_raw_json_doc().ok()?;
                Some((id, doc))
            });
        result.extend(output);
        drop(zebo);

        Ok(())
    }

    async fn apply(
        &self,
        new_docs: Vec<(DocumentId, Arc<RawJSONDocument>)>,
        to_delete: Vec<DocumentId>,
    ) -> Result<()> {
        info!(
            "Committing {} documents and deleting {} documents",
            new_docs.len(),
            to_delete.len()
        );
        trace!("New documents: {:?}", new_docs);
        trace!("To delete documents: {:?}", to_delete);
        self.apply_inner(new_docs, to_delete, 10).await?;
        info!("Documents committed");

        Ok(())
    }

    async fn apply_inner(
        &self,
        new_docs: Vec<(DocumentId, Arc<RawJSONDocument>)>,
        to_delete: Vec<DocumentId>,
        retry: u8,
    ) -> Result<()> {
        let result = timeout(Duration::from_secs(5), self.zebo.write("apply_inner")).await;
        let mut zebo = match result {
            Ok(zebo) => zebo,
            Err(_) => {
                if retry == 0 {
                    return Err(anyhow::anyhow!("Cannot acquire zebo lock"));
                }
                warn!("Zebo lock is held, retrying in 1 second...");
                sleep(Duration::from_secs(1)).await;
                return Box::pin(self.apply_inner(new_docs, to_delete, retry.saturating_sub(1)))
                    .await;
            }
        };
        let docs: Vec<_> = new_docs
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

        drop(zebo);

        Ok(())
    }
}

#[derive(Debug)]
pub struct DocumentStorageConfig {
    pub data_dir: PathBuf,
}

#[derive(Debug)]
pub struct DocumentStorage {
    uncommitted: OramaAsyncLock<Vec<(DocumentId, Arc<RawJSONDocument>)>>,
    committed: CommittedDiskDocumentStorage,
    uncommitted_document_deletions: OramaAsyncLock<Vec<DocumentId>>,
    last_document_id: OramaAsyncLock<Option<DocumentId>>,
}

impl DocumentStorage {
    pub async fn try_new(config: DocumentStorageConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir).context("Cannot create document directory")?;

        let committed = CommittedDiskDocumentStorage::try_new(config.data_dir)
            .await
            .context("Cannot create CommittedDiskDocumentStorage")?;

        Ok(Self {
            uncommitted: OramaAsyncLock::new("uncommitted", Default::default()),
            committed,
            uncommitted_document_deletions: OramaAsyncLock::new(
                "uncommitted_document_deletions",
                Default::default(),
            ),
            last_document_id: OramaAsyncLock::new("last_document_id", None),
        })
    }

    #[allow(deprecated)]
    pub async fn update(&self, op: DocumentStorageWriteOperation) -> Result<()> {
        match op {
            DocumentStorageWriteOperation::InsertDocument { doc_id, doc } => {
                self.add_document(doc_id, doc).await;
                let mut lock = self.last_document_id.write("update").await;
                **lock = None;
                Ok(())
            }
            DocumentStorageWriteOperation::InsertDocuments(docs) => {
                self.add_documents(docs).await;
                let mut lock = self.last_document_id.write("update").await;
                **lock = None;
                Ok(())
            }
            DocumentStorageWriteOperation::DeleteDocuments { doc_ids } => {
                self.delete_documents(doc_ids).await?;
                Ok(())
            }
            // New per-collection variants - should not reach global storage
            DocumentStorageWriteOperation::InsertDocumentWithDocIdStr { doc_id, doc, .. } => {
                self.add_document(doc_id, doc).await;
                let mut lock = self.last_document_id.write("update").await;
                **lock = None;
                Ok(())
            }
            DocumentStorageWriteOperation::InsertDocumentsWithDocIdStr(docs) => {
                let docs_without_str: Vec<(DocumentId, DocumentToInsert)> =
                    docs.into_iter().map(|(id, _, doc)| (id, doc)).collect();
                self.add_documents(docs_without_str).await;
                let mut lock = self.last_document_id.write("update").await;
                **lock = None;
                Ok(())
            }
            DocumentStorageWriteOperation::DeleteDocumentsWithDocIdStr(doc_id_pairs) => {
                let doc_ids: Vec<DocumentId> = doc_id_pairs.into_iter().map(|(id, _)| id).collect();
                self.delete_documents(doc_ids).await?;
                Ok(())
            }
        }
    }

    async fn add_documents(&self, docs: Vec<(DocumentId, DocumentToInsert)>) {
        if cfg!(test) && !docs.is_sorted_by(|a, b| a.0 < b.0) {
            panic!("Add documents are not in order");
        }

        let mut uncommitted = self.uncommitted.write("add_documents").await;
        uncommitted.extend(
            docs.into_iter()
                .map(|(doc_id, doc)| (doc_id, Arc::new(doc.0))),
        );
        drop(uncommitted);
    }

    async fn add_document(&self, doc_id: DocumentId, doc: DocumentToInsert) {
        self.add_documents(vec![(doc_id, doc)]).await;
    }

    async fn delete_documents(&self, doc_ids: Vec<DocumentId>) -> Result<()> {
        let mut uncommitted_document_deletions = self
            .uncommitted_document_deletions
            .write("delete_documents")
            .await;
        uncommitted_document_deletions.extend(doc_ids);
        Ok(())
    }

    pub async fn get_zebo_info(&self) -> Result<ZeboInfo> {
        let zebo = self.committed.zebo.read("get_zebo_info").await;
        zebo.get_info().context("Cannot get zebo info")
    }

    pub async fn get_last_inserted_document_id(&self) -> Result<Option<DocumentId>> {
        let lock = self
            .last_document_id
            .read("get_last_inserted_document_id")
            .await;
        if let Some(doc_id) = lock.as_ref() {
            return Ok(Some(*doc_id));
        }
        drop(lock);

        let zebo = self
            .committed
            .zebo
            .read("get_last_inserted_document_id")
            .await;
        let last_inserted_document_id = zebo
            .get_last_inserted_document_id()
            .context("Cannot get last inserted document ID")?;

        let last_uncommitted_inserted_document_id = self
            .uncommitted
            .read("get_last_inserted_document_id")
            .await
            .last()
            .map(|(id, _)| *id);

        let last = match (
            last_inserted_document_id,
            last_uncommitted_inserted_document_id,
        ) {
            (Some(committed_id), Some(uncommitted_id)) => Some(committed_id.max(uncommitted_id)),
            (Some(committed_id), None) => Some(committed_id),
            (None, Some(uncommitted_id)) => Some(uncommitted_id),
            (None, None) => None,
        };

        let mut lock = self
            .last_document_id
            .write("get_last_inserted_document_id")
            .await;
        **lock = last;
        drop(lock);

        Ok(last)
    }

    pub async fn get_documents_by_ids(
        &self,
        mut doc_ids: Vec<DocumentId>,
        result: &mut Vec<(DocumentId, Arc<RawJSONDocument>)>,
    ) -> Result<()> {
        let uncommitted_document_deletions = self
            .uncommitted_document_deletions
            .read("get_documents_by_ids")
            .await;
        doc_ids.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
        drop(uncommitted_document_deletions);

        trace!("Get from uncommitted documents");
        let uncommitted = self.uncommitted.read("get_documents_by_ids").await;
        let uncommitted_docs: Vec<_> = doc_ids
            .iter()
            .filter_map(|doc_id| {
                uncommitted
                    .iter()
                    .find(|i| i.0 == *doc_id)
                    .map(|(id, doc)| (*id, doc.clone()))
            })
            .collect();
        let doc_ids_found_in_uncommitted: Vec<_> = uncommitted.iter().map(|(id, _)| *id).collect();
        result.extend(uncommitted_docs);
        drop(uncommitted);
        trace!("Get from uncommitted documents done");

        debug!("Get from committed documents");
        doc_ids.retain(|id| !doc_ids_found_in_uncommitted.contains(id));
        let doc_ids_to_find_in_committed = doc_ids;
        self.committed
            .get_documents_by_ids(&doc_ids_to_find_in_committed, result)
            .await?;

        Ok(())
    }

    pub async fn commit(&self) -> Result<()> {
        info!("Commit documents");

        let m = DOCUMENT_COMMIT_CALCULATION_TIME.create(Empty {});

        let uncommitted_lock = self.uncommitted.read("commit").await;
        let uncommitted_docs: Vec<_> = uncommitted_lock.clone();
        drop(uncommitted_lock);

        let uncommitted_document_deletions_lock =
            self.uncommitted_document_deletions.read("commit").await;
        let uncommitted_document_deletions_docs = uncommitted_document_deletions_lock.clone();
        drop(uncommitted_document_deletions_lock);

        self.committed
            .apply(uncommitted_docs, uncommitted_document_deletions_docs)
            .await
            .context("Cannot commit documents")?;

        let mut uncommitted_lock = self.uncommitted.write("commit").await;
        uncommitted_lock.clear();
        drop(uncommitted_lock);

        drop(m);

        let mut lock = self.last_document_id.write("update").await;
        **lock = None;

        info!("Documents committed");

        Ok(())
    }
}

pub async fn migrate_to_zebo(data_dir: &PathBuf) -> Result<Zebo<1_000_000, PAGE_SIZE, DocumentId>> {
    let mut files_in_dir = std::fs::read_dir(data_dir)
        .context("Cannot read data directory")?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() {
                Some(path)
            } else {
                None
            }
        })
        .filter_map(|entry| {
            let file_name = entry.file_name()?;
            let file_name = file_name.to_str()?;
            let doc_id = file_name.parse::<u64>().ok()?;
            let doc_id = DocumentId(doc_id);
            Some(doc_id)
        })
        .collect::<Vec<_>>();
    files_in_dir.sort_by_key(|id| id.0);

    let mut zebo = Zebo::try_new(data_dir.join("zebo")).context("Cannot create zebo")?;

    for doc_id in files_in_dir {
        let doc_path = data_dir.join(doc_id.0.to_string());
        let data: wrapper::RawJSONDocumentWrapper = match read_file(doc_path).await {
            Ok(data) => data,
            Err(e) => {
                error!(error = ?e, "Cannot read document data");
                continue;
            }
        };
        let doc_id_str = data.0.id.clone().unwrap_or_default();
        let doc = data.0.inner.get();

        zebo.reserve_space_for(&[(
            doc_id,
            ZeboDocument::Split(Cow::Owned(doc_id_str), Cow::Borrowed(doc)),
        )])
        .unwrap()
        .write_all()
        .unwrap();
    }

    Ok(zebo)
}

static ZERO: &[u8] = b"\0";

#[derive(Debug)]
pub enum ZeboDocument<'s> {
    Split(Cow<'s, str>, Cow<'s, str>),
    FromJSONDoc(Arc<RawJSONDocument>),
}

impl zebo::Document for ZeboDocument<'_> {
    fn as_bytes(&self, v: &mut Vec<u8>) {
        match self {
            Self::Split(id, json) => {
                v.extend(id.as_bytes());
                v.extend(ZERO);
                v.extend(json.as_bytes());
            }
            Self::FromJSONDoc(a) => {
                if let Some(id) = &a.id {
                    v.extend(id.as_bytes())
                };
                v.extend(ZERO);
                v.extend(a.inner.get().as_bytes());
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Split(id, json) => id.len() + ZERO.len() + json.len(),
            Self::FromJSONDoc(a) => {
                let id_len = a.id.as_ref().map_or(0, |id| id.len());
                id_len + ZERO.len() + a.inner.get().len()
            }
        }
    }
}

impl ZeboDocument<'_> {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let mut parts = bytes.split(|b| *b == b'\0');
        let id = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Cannot split document bytes"))?;
        let data = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Cannot split document bytes"))?;

        let id = String::from_utf8(id.to_vec()).context("Cannot convert id to string")?;
        let data = String::from_utf8(data.to_vec()).context("Cannot convert data to string")?;

        Ok(ZeboDocument::Split(Cow::Owned(id), Cow::Owned(data)))
    }

    pub fn as_raw_json_doc(&self) -> Result<Arc<RawJSONDocument>> {
        match self {
            Self::Split(id, json) => {
                let inner = serde_json::value::RawValue::from_string(json.to_string())?;
                Ok(Arc::new(RawJSONDocument {
                    id: Some(id.to_string()),
                    inner,
                }))
            }
            Self::FromJSONDoc(a) => Ok(a.clone()),
        }
    }
}

// This module should not be used anymore.
// It is only used for migrating the documents to zebo.
mod wrapper {
    use std::{fmt::Debug, sync::Arc};

    use serde::{de::Unexpected, Deserialize, Serialize};

    use crate::types::RawJSONDocument;

    pub struct RawJSONDocumentWrapper(pub Arc<RawJSONDocument>);

    #[cfg(test)]
    impl PartialEq for RawJSONDocumentWrapper {
        fn eq(&self, other: &Self) -> bool {
            self.0.id == other.0.id && self.0.inner.get() == other.0.inner.get()
        }
    }
    impl Debug for RawJSONDocumentWrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_tuple("RawJSONDocumentWrapper")
                .field(&self.0)
                .finish()
        }
    }

    impl Serialize for RawJSONDocumentWrapper {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::ser::Serializer,
        {
            use serde::ser::SerializeTuple;
            let mut tuple = serializer.serialize_tuple(2)?;
            tuple.serialize_element(&self.0.id)?;
            tuple.serialize_element(&self.0.inner.get())?;
            tuple.end()
        }
    }

    impl<'de> Deserialize<'de> for RawJSONDocumentWrapper {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            use core::result::Result;
            use core::result::Result::*;
            use serde::de::{Error, Visitor};

            struct SerializableNumberVisitor;

            impl<'de> Visitor<'de> for SerializableNumberVisitor {
                type Value = RawJSONDocumentWrapper;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(
                        formatter,
                        "a tuple of size 2 consisting of a id string and the raw value"
                    )
                }

                fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: serde::de::SeqAccess<'de>,
                {
                    let id: Option<String> = seq.next_element()?;
                    let inner: Option<String> = seq.next_element()?;

                    let inner = match inner {
                        None => return Err(A::Error::missing_field("inner")),
                        Some(inner) => inner,
                    };

                    let inner = match serde_json::value::RawValue::from_string(inner) {
                        Err(_) => {
                            return Err(A::Error::invalid_value(
                                Unexpected::Str("Invalid RawValue"),
                                &"A valid RawValue",
                            ))
                        }
                        Ok(inner) => inner,
                    };

                    Result::Ok(RawJSONDocumentWrapper(Arc::new(RawJSONDocument {
                        id,
                        inner,
                    })))
                }
            }

            deserializer.deserialize_tuple(2, SerializableNumberVisitor)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn test_document_storage_raw_json_document_wrapper_serialize_deserialize() {
            let value = json!({
                "id": "the-id",
                "foo": "bar",
            });
            let raw_json_document: RawJSONDocument = value.try_into().unwrap();
            assert_eq!(raw_json_document.id, Some("the-id".to_string()));

            let raw_json_document_wrapper = RawJSONDocumentWrapper(Arc::new(raw_json_document));
            let serialized = serde_json::to_string(&raw_json_document_wrapper).unwrap();
            let deserialized: RawJSONDocumentWrapper = serde_json::from_str(&serialized).unwrap();

            assert_eq!(raw_json_document_wrapper, deserialized);
        }
    }
}
