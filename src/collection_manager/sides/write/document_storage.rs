use serde::{de::Unexpected, Deserialize, Serialize};
use std::
    path::PathBuf;

use anyhow::{Context, Ok, Result};

use crate::{
    file_utils::{create_if_not_exists, create_or_overwrite},
    types::{Document, DocumentId, RawJSONDocument},
};

pub struct DocumentStorage {
    data_dir: PathBuf,
}

impl DocumentStorage {
    pub fn try_new(data_dir: PathBuf) -> Result<Self> {
        create_if_not_exists(&data_dir)
            .context("Cannot create data directory")?;

        Ok(Self {
            data_dir,
        })
    }

    pub async fn insert(&self, id: DocumentId, document: Document) -> Result<()> {
        let document: RawJSONDocument = document.into_raw()?;
        let doc_path = self.data_dir.join(id.0.to_string());
        let data = RawJSONDocumentWrapper(document);
        create_or_overwrite(doc_path, &data)
            .await
            .context("Cannot write document data")?;
        Ok(())
    }
}

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
