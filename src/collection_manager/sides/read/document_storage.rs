use serde::{de::Unexpected, Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    path::PathBuf,
    sync::Arc,
};
use tracing::{debug, info, trace, warn};

use anyhow::{anyhow, Context, Result};

use crate::{
    file_utils::{create_or_overwrite, read_file},
    metrics::{commit::DOCUMENT_COMMIT_CALCULATION_TIME, Empty},
    types::{DocumentId, RawJSONDocument},
};

// The `CommittedDiskDocumentStorage` implementation is not optimal.
// Defenitely, we cannot read every time from disk, it is too heavy.
// We should be backed by the disk, but we should also have a cache in memory.
// We should find a good balance between memory and disk usage.
// TODO: think about a better implementation.

#[derive(Debug)]
struct CommittedDiskDocumentStorage {
    path: PathBuf,
    cache: tokio::sync::RwLock<HashMap<DocumentId, Arc<RawJSONDocument>>>,
}
impl CommittedDiskDocumentStorage {
    fn new(path: PathBuf) -> Self {
        let cache: tokio::sync::RwLock<HashMap<DocumentId, Arc<RawJSONDocument>>> =
            Default::default();

        Self { path, cache }
    }

    async fn get_documents_by_ids(
        &self,
        doc_ids: &[DocumentId],
    ) -> Result<Vec<Option<Arc<RawJSONDocument>>>> {
        let lock = self.cache.read().await;
        let mut from_cache: HashMap<DocumentId, Arc<RawJSONDocument>> = doc_ids
            .iter()
            .filter_map(|id| lock.get(id).map(|d| (*id, d.clone())))
            .collect();
        drop(lock);

        trace!(doc_len=?doc_ids.len(), "Read document");
        let mut result = Vec::with_capacity(doc_ids.len());
        for id in doc_ids {
            trace!(?id, "Reading document");
            if let Some(d) = from_cache.remove(id) {
                trace!("In cache. skip");
                result.push(Some(d));
                continue;
            }

            let doc_path = self.path.join(format!("{}", id.0));
            trace!(?doc_path, "Check on FS");

            let exists = tokio::fs::try_exists(&doc_path).await;
            match exists {
                Err(e) => {
                    return Err(anyhow!(
                        "Error while checking if the document exists: {:?}",
                        e
                    ));
                }
                std::result::Result::Ok(false) => {
                    trace!("Not found on FS");
                    result.push(None);
                    continue;
                }
                std::result::Result::Ok(true) => {}
            };

            let doc: RawJSONDocumentWrapper = match read_file(doc_path).await {
                std::result::Result::Err(_) => {
                    // It could happen the `commit` method creates the file (so the previous check passes)
                    // but not with the full content written.
                    // In that case, `read_file` will fail.
                    // If it happens, this arm is triggered.
                    trace!("Error on read from FS");
                    result.push(None);
                    continue;
                }
                std::result::Result::Ok(doc) => doc,
            };

            let doc = doc.0;
            let mut lock = self.cache.write().await;
            lock.insert(*id, doc.clone());
            drop(lock);

            result.push(Some(doc));
        }

        Ok(result)
    }

    async fn add(&self, docs: Vec<(DocumentId, Arc<RawJSONDocument>)>) -> Result<()> {
        for (doc_id, doc) in docs {
            let doc_path = self.path.join(format!("{}", doc_id.0));

            let doc = RawJSONDocumentWrapper(doc);
            create_or_overwrite(doc_path, &doc)
                .await
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
pub struct DocumentStorage {
    uncommitted: tokio::sync::RwLock<HashMap<DocumentId, Arc<RawJSONDocument>>>,
    committed: CommittedDiskDocumentStorage,
    uncommitted_document_deletions: tokio::sync::RwLock<HashSet<DocumentId>>,
}

impl DocumentStorage {
    pub fn try_new(config: DocumentStorageConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir).context("Cannot create document directory")?;

        Ok(Self {
            uncommitted: Default::default(),
            committed: CommittedDiskDocumentStorage::new(config.data_dir),
            uncommitted_document_deletions: Default::default(),
        })
    }

    pub async fn add_document(&self, doc_id: DocumentId, doc: RawJSONDocument) -> Result<()> {
        let doc = Arc::new(doc);
        let mut uncommitted = self.uncommitted.write().await;
        if uncommitted.insert(doc_id, doc).is_some() {
            warn!("Document {:?} already exists. Overwritten.", doc_id);
        }
        Ok(())
    }

    pub async fn delete_document(&self, doc_id: &DocumentId) -> Result<()> {
        let mut uncommitted_document_deletions = self.uncommitted_document_deletions.write().await;
        uncommitted_document_deletions.insert(*doc_id);

        Ok(())
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
            .into_iter()
            .map(|doc_id| uncommitted.get(&doc_id).cloned())
            .collect();
        trace!("Get from uncommitted documents done");

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

    pub async fn commit(&self) -> Result<()> {
        info!("Commit documents");
        // This implementation is wrong:
        // in the mean time we "dran" + "collection" + "write on FS"
        // The documents aren't reachable. So the search output will not contain them.
        // We should follow the same path of the indexes.
        // TODO: fix me

        let m = DOCUMENT_COMMIT_CALCULATION_TIME.create(Empty {});
        let mut lock = self.uncommitted.write().await;
        let uncommitted: Vec<_> = lock.drain().collect();
        drop(lock);

        self.committed
            .add(uncommitted)
            .await
            .context("Cannot commit documents")?;
        drop(m);

        let mut uncommitted_document_deletions = self.uncommitted_document_deletions.write().await;
        for doc_id in uncommitted_document_deletions.drain() {
            let doc_path = self.committed.path.join(format!("{}", doc_id.0));
            match tokio::fs::remove_file(doc_path).await {
                Ok(_) => {
                    debug!("Document {:?} deleted", doc_id);
                }
                // Not found is not an error
                Err(e) if e.raw_os_error() == Some(2) => {
                    warn!("Error while deleting document {:?}: {:?}", doc_id, e);
                }
                Err(e) => {
                    return Err(anyhow!(
                        "Error while deleting document {:?}: {:?}",
                        doc_id,
                        e
                    ))
                }
            }
        }
        uncommitted_document_deletions.clear();
        drop(uncommitted_document_deletions);

        info!("Documents committed");

        Ok(())
    }

    pub fn load(&mut self) -> Result<()> {
        Ok(())
    }
}

struct RawJSONDocumentWrapper(Arc<RawJSONDocument>);

#[cfg(test)]
impl PartialEq for RawJSONDocumentWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id && self.0.inner.get() == other.0.inner.get()
    }
}
#[cfg(test)]
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
