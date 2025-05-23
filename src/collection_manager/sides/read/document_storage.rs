use itertools::Itertools;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::Debug,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};
use zebo::Zebo;

use anyhow::{Context, Result};

use crate::{
    collection_manager::sides::DocumentStorageWriteOperation,
    file_utils::{create_if_not_exists, read_file},
    metrics::{commit::DOCUMENT_COMMIT_CALCULATION_TIME, Empty},
    types::{DocumentId, RawJSONDocument},
};

// The `CommittedDiskDocumentStorage` implementation is not optimal.
// Defenitely, we cannot read every time from disk, it is too heavy.
// We should be backed by the disk, but we should also have a cache in memory.
// We should find a good balance between memory and disk usage.
// TODO: think about a better implementation.

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
                let doc = match ZeboDocument::from_bytes(d.1) {
                    Ok(doc) => Some(doc),
                    Err(_) => None,
                };
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

    async fn add(&self, docs: Vec<(DocumentId, Arc<RawJSONDocument>)>) -> Result<()> {
        let mut zebo = self.zebo.write().await;
        zebo.add_documents(
            docs.into_iter()
                .map(|(doc_id, doc)| (doc_id, ZeboDocument::FromJSONDoc(doc))),
        )
        .context("Cannot add documents")?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct DocumentStorageConfig {
    pub data_dir: PathBuf,
}

#[derive(Debug)]
pub struct DocumentStorage {
    uncommitted: tokio::sync::RwLock<HashMap<DocumentId, Arc<RawJSONDocument>>>,
    committed: CommittedDiskDocumentStorage,
    uncommitted_document_deletions: tokio::sync::RwLock<HashSet<DocumentId>>,
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
                self.add_document(doc_id, doc.0).await
            }
            DocumentStorageWriteOperation::DeleteDocuments { doc_ids } => {
                self.delete_documents(doc_ids).await?;
                Ok(())
            }
        }
    }

    async fn add_document(&self, doc_id: DocumentId, doc: RawJSONDocument) -> Result<()> {
        let doc = Arc::new(doc);
        let mut uncommitted = self.uncommitted.write().await;
        if uncommitted.insert(doc_id, doc).is_some() {
            warn!("Document {:?} already exists. Overwritten.", doc_id);
        }
        Ok(())
    }

    async fn delete_documents(&self, doc_ids: Vec<DocumentId>) -> Result<()> {
        let mut uncommitted_document_deletions = self.uncommitted_document_deletions.write().await;
        uncommitted_document_deletions.extend(doc_ids);

        Ok(())
    }

    #[tracing::instrument(skip(self, doc_ids))]
    pub async fn get_documents_by_ids(
        &self,
        mut doc_ids: Vec<DocumentId>,
    ) -> Result<Vec<Option<Arc<RawJSONDocument>>>> {
        println!("get_documents_by_ids: {doc_ids:?}");

        let uncommitted_document_deletions = self.uncommitted_document_deletions.read().await;
        doc_ids.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));

        debug!("Get from committed documents");
        let committed = self.committed.get_documents_by_ids(&doc_ids).await?;

        trace!("Get from uncommitted documents");
        let uncommitted = self.uncommitted.read().await;
        let uncommitted: Vec<_> = doc_ids
            .iter()
            .map(|doc_id| uncommitted.get(doc_id).cloned())
            .collect();
        trace!("Get from uncommitted documents done");

        let mut result = Vec::with_capacity(doc_ids.len());
        for i in 0..doc_ids.len() {
            let doc_id = doc_ids[i];
            let committed = committed.get(i).cloned().flatten();
            let uncommitted = uncommitted.get(i).cloned().flatten();

            if let Some(doc) = uncommitted {
                result.push(Some(doc));
            } else {
                if committed.is_none() {
                    warn!("Document {:?} not found", doc_id);
                }
                result.push(committed);
            }
        }

        println!("got {:?}", result);

        Ok(result)
    }

    pub async fn commit(&self) -> Result<()> {
        info!("Commit documents");
        // This implementation is wrong:
        // in the mean time we "dran" + "collection" + "write on FS"
        // The documents aren't reachable. So the search output will not contain them.
        // We should follow the same path of the indexes.
        // TODO: fix me

        let m = DOCUMENT_COMMIT_CALCULATION_TIME.create(Empty {});

        let mut lock = self.uncommitted.write().await;
        let mut uncommitted_document_deletions = self.uncommitted_document_deletions.write().await;

        println!("Moving...");

        let doc_to_delete: Vec<_> = uncommitted_document_deletions.drain().collect();

        let mut uncommitted: Vec<_> = lock.drain().collect();
        uncommitted.sort_by_key(|(doc_id, _)| *doc_id);

        println!("Moving {uncommitted:?}");

        self.committed
            .add(uncommitted)
            .await
            .context("Cannot commit documents")?;
        let mut zebo = self.committed.zebo.write().await;

        println!("Deleting: {doc_to_delete:?}");

        zebo.remove_documents(doc_to_delete, false)
            .context("Cannot remove documents")?;

        drop(zebo);
        drop(uncommitted_document_deletions);
        drop(lock);
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
    fn as_bytes(&self) -> Cow<[Cow<[u8]>]> {
        match self {
            Self::Split(id, json) => {
                let mut bytes = Vec::with_capacity(3);
                bytes.push(Cow::Borrowed(id.as_bytes()));
                bytes.push(Cow::Borrowed(ZERO));
                bytes.push(Cow::Borrowed(json.as_bytes()));
                Cow::Owned(bytes)
            }
            Self::FromJSONDoc(a) => {
                let mut bytes = Vec::with_capacity(3);
                if let Some(id) = &a.id {
                    bytes.push(Cow::Borrowed(id.as_bytes()))
                };
                bytes.push(Cow::Borrowed(ZERO));

                bytes.push(Cow::Borrowed(a.inner.get().as_bytes()));
                Cow::Owned(bytes)
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
