use itertools::Itertools;
use std::{borrow::Cow, fmt::Debug, path::PathBuf, sync::Arc};
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};
use zebo::{Zebo, ZeboInfo};

use anyhow::{Context, Result};

use crate::{
    collection_manager::sides::{DocumentStorageWriteOperation, DocumentToInsert},
    file_utils::{create_if_not_exists, read_file},
    metrics::{commit::DOCUMENT_COMMIT_CALCULATION_TIME, Empty},
    types::{DocumentId, RawJSONDocument},
};

// 1 GB
static PAGE_SIZE: u64 = 1024 * 1024 * 1024;

#[derive(Debug)]
struct CommittedDiskDocumentStorage {
    zebo: Arc<RwLock<Zebo<1_000_000, PAGE_SIZE, DocumentId>>>,
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

        let zebo = Arc::new(RwLock::new(zebo));

        Ok(Self { zebo })
    }

    async fn get_documents_by_ids(
        &self,
        doc_ids: &[DocumentId],
    ) -> Result<Vec<Option<Arc<RawJSONDocument>>>> {
        trace!(doc_len=?doc_ids.len(), "Read document");

        let zebo = self.zebo.read().await;
        let output = zebo
            .get_documents(doc_ids.to_vec())
            .context("Cannot get documents from zebo")?;
        let output: Result<Vec<_>, zebo::ZeboError> = output.collect();
        let output = output.context("Failed to got documents")?;
        let mut output: Vec<_> = output
            .into_iter()
            .map(|d| {
                let doc = ZeboDocument::from_bytes(d.1).ok();
                (d.0, doc)
            })
            .collect();
        output.sort_by_key(|d| doc_ids.iter().find_position(|dd| **dd == d.0));

        let output: Vec<_> = output
            .into_iter()
            .map(|(_, d)| match d {
                None => None,
                Some(d) => d.into_raw_json_doc().ok(),
            })
            .collect();

        Ok(output)
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

        let mut zebo = self.zebo.write().await;
        zebo.add_documents_batch(
            new_docs
                .into_iter()
                .map(|(doc_id, doc)| (doc_id, ZeboDocument::FromJSONDoc(doc))),
            200,  // 200 at the time
            1024, // each document is 1kb
        )
        .context("Cannot add documents")?;

        zebo.remove_documents(to_delete, false)
            .context("Cannot remove documents")?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct DocumentStorageConfig {
    pub data_dir: PathBuf,
}

#[derive(Debug)]
pub struct DocumentStorage {
    uncommitted: tokio::sync::RwLock<Vec<(DocumentId, Arc<RawJSONDocument>)>>,
    committed: CommittedDiskDocumentStorage,
    uncommitted_document_deletions: tokio::sync::RwLock<Vec<DocumentId>>,
}

impl DocumentStorage {
    pub async fn try_new(config: DocumentStorageConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir).context("Cannot create document directory")?;

        let committed = CommittedDiskDocumentStorage::try_new(config.data_dir)
            .await
            .context("Cannot create CommittedDiskDocumentStorage")?;

        Ok(Self {
            uncommitted: Default::default(),
            committed,
            uncommitted_document_deletions: Default::default(),
        })
    }

    pub async fn update(&self, op: DocumentStorageWriteOperation) -> Result<()> {
        match op {
            DocumentStorageWriteOperation::InsertDocument { doc_id, doc } => {
                self.add_document(doc_id, doc).await;
                Ok(())
            }
            DocumentStorageWriteOperation::InsertDocuments(docs) => {
                self.add_documents(docs).await;
                Ok(())
            }
            DocumentStorageWriteOperation::DeleteDocuments { doc_ids } => {
                self.delete_documents(doc_ids).await?;
                Ok(())
            }
        }
    }

    async fn add_documents(&self, docs: Vec<(DocumentId, DocumentToInsert)>) {
        if cfg!(test) && !docs.is_sorted_by(|a, b| a.0 < b.0) {
            panic!("Add documents are not in order");
        }

        let mut uncommitted = self.uncommitted.write().await;
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
        let mut uncommitted_document_deletions = self.uncommitted_document_deletions.write().await;
        uncommitted_document_deletions.extend(doc_ids);
        Ok(())
    }

    pub async fn get_zebo_info(&self) -> Result<ZeboInfo> {
        let zebo = self.committed.zebo.read().await;
        zebo.get_info().context("Cannot get zebo info")
    }

    #[tracing::instrument(skip(self, doc_ids))]
    pub async fn get_documents_by_ids(
        &self,
        mut doc_ids: Vec<DocumentId>,
    ) -> Result<Vec<Option<Arc<RawJSONDocument>>>> {
        let uncommitted_document_deletions = self.uncommitted_document_deletions.read().await;
        doc_ids.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));

        debug!("Get from committed documents");
        let committed = self.committed.get_documents_by_ids(&doc_ids).await?;

        trace!("Get from uncommitted documents");
        let uncommitted = self.uncommitted.read().await;
        let uncommitted: Vec<_> = doc_ids
            .iter()
            .map(|doc_id| uncommitted.iter().find(|i| i.0 == *doc_id).cloned())
            .collect();
        trace!("Get from uncommitted documents done");

        let mut result = Vec::with_capacity(doc_ids.len());
        for (i, doc_id) in doc_ids.iter().enumerate() {
            let committed = committed.get(i).cloned().flatten();
            let uncommitted = uncommitted.get(i).cloned().flatten();

            if let Some((_, doc)) = uncommitted {
                result.push(Some(doc));
            } else {
                if committed.is_none() {
                    warn!("Document {:?} not found", doc_id);
                }
                result.push(committed);
            }
        }

        Ok(result)
    }

    pub async fn commit(&self) -> Result<()> {
        info!("Commit documents");

        let m = DOCUMENT_COMMIT_CALCULATION_TIME.create(Empty {});

        let uncommitted_lock = self.uncommitted.read().await;
        let uncommitted_docs: Vec<_> = uncommitted_lock.clone();
        drop(uncommitted_lock);

        let uncommitted_document_deletions_lock = self.uncommitted_document_deletions.read().await;
        let uncommitted_document_deletions_docs = uncommitted_document_deletions_lock.clone();
        drop(uncommitted_document_deletions_lock);

        self.committed
            .apply(uncommitted_docs, uncommitted_document_deletions_docs)
            .await
            .context("Cannot commit documents")?;

        let mut uncommitted_lock = self.uncommitted.write().await;
        uncommitted_lock.clear();
        drop(uncommitted_lock);

        drop(m);

        info!("Documents committed");

        Ok(())
    }

    pub fn load(&mut self) -> Result<()> {
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

        zebo.add_documents(vec![(
            doc_id,
            ZeboDocument::Split(Cow::Owned(doc_id_str), Cow::Borrowed(doc)),
        )])
        .unwrap();
    }

    Ok(zebo)
}

static ZERO: &[u8] = b"\0";

enum ZeboDocument<'s> {
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
}

impl ZeboDocument<'_> {
    fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
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

    fn into_raw_json_doc(&self) -> Result<Arc<RawJSONDocument>> {
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
