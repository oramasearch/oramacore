use futures::Stream;
use serde::{de::Unexpected, Deserialize, Serialize};
use serde_json::value::RawValue;
use std::{borrow::Cow, path::PathBuf, sync::Arc};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use zebo::Zebo;

use anyhow::{Context, Result};
use tracing::{error, info, warn};

use crate::types::{DocumentId, RawJSONDocument};
use oramacore_lib::fs::{create_if_not_exists, read_file};

// 1GB
const PAGE_SIZE: u64 = 1024 * 1024 * 1024;

pub struct DocumentStorage {
    zebo: Arc<RwLock<Zebo<1_000_000, PAGE_SIZE, DocumentId>>>,
}

impl DocumentStorage {
    pub async fn try_new(data_dir: PathBuf) -> Result<Self> {
        create_if_not_exists(&data_dir).context("Cannot create data directory")?;

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

        Ok(Self {
            zebo: Arc::new(RwLock::new(zebo)),
        })
    }

    pub async fn insert_many(&self, docs: &[(DocumentId, ZeboDocument<'_>)]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }

        let mut zebo = self.zebo.write().await;
        let space = zebo
            .reserve_space_for(docs)
            .context("Cannot reserve space in zebo")?;
        drop(zebo);

        space.write_all().context("Cannot write documents")?;

        Ok(())
    }

    pub async fn remove(&self, ids: Vec<DocumentId>) {
        if !ids.is_empty() {
            let mut zebo = self.zebo.write().await;
            zebo.remove_documents(ids, false).unwrap();
        }
    }

    pub async fn stream_documents(
        &self,
        ids: Vec<DocumentId>,
    ) -> impl Stream<Item = (DocumentId, RawJSONDocument)> {
        let (tx, rx) = mpsc::channel(100);

        let zebo = self.zebo.clone();
        tokio::spawn(async move {
            let zebo = zebo.read().await;

            let docs_iter = match zebo.get_documents(ids) {
                Ok(a) => a,
                Err(e) => {
                    error!(error = ?e, "Cannot get documents");
                    return;
                }
            };

            for doc in docs_iter {
                let (doc_id, data) = match doc {
                    Ok(doc) => doc,
                    Err(e) => {
                        error!(error = ?e, "Cannot get document");
                        continue;
                    }
                };

                let doc = match ZeboDocument::from_bytes(data)
                    .context("Cannot convert bytes to document")
                {
                    Ok(doc) => doc,
                    Err(e) => {
                        error!(error = ?e, "Cannot convert bytes to document");
                        continue;
                    }
                };
                let inner = RawValue::from_string(doc.1.to_string())
                    .context("Cannot convert bytes to document")
                    .unwrap();
                let d = RawJSONDocument {
                    id: Some(doc.0.to_string()),
                    inner,
                };

                if let Err(e) = tx.send((doc_id, d)).await {
                    error!(error = ?e, "Cannot send document data. Stopped stream_documents");
                    break;
                }
            }
        });

        ReceiverStream::new(rx)
    }
}

#[derive(Debug)]
struct RawJSONDocumentWrapper(RawJSONDocument);

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

                Result::Ok(RawJSONDocumentWrapper(RawJSONDocument { id, inner }))
            }
        }

        deserializer.deserialize_tuple(2, SerializableNumberVisitor)
    }
}

impl zebo::DocumentId for DocumentId {
    fn as_u64(&self) -> u64 {
        self.0
    }

    fn from_u64(id: u64) -> Self {
        Self(id)
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
        let data: RawJSONDocumentWrapper = match read_file(doc_path).await {
            Ok(data) => data,
            Err(e) => {
                error!(error = ?e, "Cannot read document data");
                continue;
            }
        };
        let doc_id_str = data.0.id.unwrap_or_default();
        let doc = data.0.inner.get();

        zebo.reserve_space_for(&[(
            doc_id,
            ZeboDocument(Cow::Owned(doc_id_str), Cow::Borrowed(doc)),
        )])
        .unwrap()
        .write_all()
        .unwrap();
    }

    Ok(zebo)
}

static ZERO: &[u8] = b"\0";

pub struct ZeboDocument<'s>(Cow<'s, str>, Cow<'s, str>);

impl zebo::Document for ZeboDocument<'_> {
    fn as_bytes(&self, v: &mut Vec<u8>) {
        v.extend(self.0.as_bytes());
        v.extend(ZERO);
        v.extend(self.1.as_bytes());
    }

    fn len(&self) -> usize {
        self.0.len() + ZERO.len() + self.1.len()
    }
}

impl<'s> ZeboDocument<'s> {
    pub fn new(id: Cow<'s, str>, content: Cow<'s, str>) -> Self {
        Self(id, content)
    }

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

        Ok(ZeboDocument(Cow::Owned(id), Cow::Owned(data)))
    }
}
