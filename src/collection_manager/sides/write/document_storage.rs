use futures::Stream;
use serde::{de::Unexpected, Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use anyhow::{Context, Result};
use tracing::error;

use crate::{
    file_utils::{create_if_not_exists, read_file, BufferedFile},
    types::{Document, DocumentId, RawJSONDocument},
};

pub struct DocumentStorage {
    data_dir: PathBuf,
}

impl DocumentStorage {
    pub fn try_new(data_dir: PathBuf) -> Result<Self> {
        create_if_not_exists(&data_dir).context("Cannot create data directory")?;

        Ok(Self { data_dir })
    }

    pub async fn insert(
        &self,
        id: DocumentId,
        doc_id_str: String,
        document: Document,
    ) -> Result<()> {
        let document: RawJSONDocument = document.into_raw(doc_id_str)?;
        let doc_path = self.data_dir.join(id.0.to_string());
        let data = RawJSONDocumentWrapper(document);
        BufferedFile::create_or_overwrite(doc_path)
            .context("Cannot create document data")?
            .write_json_data(&data)
            .context("Cannot write document data")?;
        Ok(())
    }

    pub async fn remove(&self, ids: Vec<DocumentId>) {
        for id in ids {
            let doc_path = self.data_dir.join(id.0.to_string());
            if let Err(e) = tokio::fs::remove_file(doc_path).await {
                // We ignore the error because we want to proceed with the next document
                error!(error = ?e, "Cannot remove document data");
            }
        }
    }

    pub async fn stream_documents(
        &self,
        ids: Vec<DocumentId>,
    ) -> impl Stream<Item = (DocumentId, RawJSONDocument)> {
        let (tx, rx) = mpsc::channel(100);

        let data_dir = self.data_dir.clone();
        tokio::spawn(async move {
            for id in ids {
                let doc_path = data_dir.join(id.0.to_string());
                let data: RawJSONDocumentWrapper = match read_file(doc_path).await {
                    Ok(data) => data,
                    Err(e) => {
                        error!(error = ?e, "Cannot read document data");
                        continue;
                    }
                };
                if let Err(e) = tx.send((id, data.0)).await {
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
