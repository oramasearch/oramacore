use std::{collections::HashMap, str::FromStr};

use serde::{ser::SerializeTuple, Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::{
    collection_manager::sides::{hooks::HookName, index::IndexedValue, OramaModelSerializable},
    nlp::locales::Locale,
    types::{
        ApiKey, CollectionId, DocumentFields, DocumentId, FieldId, IndexId, Number, RawJSONDocument,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Term(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermStringField {
    pub positions: Vec<usize>,
}

pub type InsertStringTerms = HashMap<Term, TermStringField>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentFieldIndexOperation {
    IndexString {
        field_length: u16,
        terms: InsertStringTerms,
    },
    IndexEmbedding {
        value: Vec<f32>,
    },
    IndexStringFilter {
        value: String,
    },
    IndexNumber {
        value: NumberWrapper,
    },
    IndexBoolean {
        value: bool,
    },
}

#[derive(Debug, Clone)]
pub struct NumberWrapper(pub Number);
impl Serialize for NumberWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        let b = match self.0 {
            Number::I32(i) => (1_u8, i.to_be_bytes()),
            Number::F32(f) => (2_u8, f.to_be_bytes()),
        };
        let mut seq = serializer.serialize_tuple(5)?;
        seq.serialize_element(&b.0)?;
        seq.serialize_element(&b.1[0])?;
        seq.serialize_element(&b.1[1])?;
        seq.serialize_element(&b.1[2])?;
        seq.serialize_element(&b.1[3])?;
        seq.end()
    }
}
impl<'de> Deserialize<'de> for NumberWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let b = <(u8, u8, u8, u8, u8)>::deserialize(deserializer)?;
        let num = match b.0 {
            1 => Number::I32(i32::from_be_bytes([b.1, b.2, b.3, b.4])),
            2 => Number::F32(f32::from_be_bytes([b.1, b.2, b.3, b.4])),
            _ => {
                return Err(serde::de::Error::custom(format!(
                    "Invalid tag value: {}",
                    b.0
                )))
            }
        };
        Ok(NumberWrapper(num))
    }
}

// We wrap RawJSONDocument in a struct to implement Serialize and Deserialize
// This is because RawJSONDocument is a newtype over RawValue, which is not Serialize/Deserialize
// and bincode does not support well newtypes.
#[derive(Debug, Clone)]
pub struct DocumentToInsert(pub RawJSONDocument);
impl Serialize for DocumentToInsert {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        let mut seq = serializer.serialize_tuple(2)?;
        seq.serialize_element(&self.0.id)?;
        seq.serialize_element(self.0.inner.get())?;
        seq.end()
    }
}
impl<'de> Deserialize<'de> for DocumentToInsert {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let a = <(Option<String>, &str)>::deserialize(deserializer)?;
        let raw = RawValue::from_string(a.1.to_string())
            .expect("The deserialization of RawValue should not fail");
        let doc = RawJSONDocument {
            id: a.0,
            inner: raw,
        };
        Ok(DocumentToInsert(doc))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexWriteOperationFieldType {
    Number,
    Bool,
    StringFilter,
    String,
    Embedding(OramaModelSerializable),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexWriteOperation {
    CreateField2 {
        field_id: FieldId,
        field_path: Box<[String]>,
        is_array: bool,
        field_type: IndexWriteOperationFieldType,
    },
    Index {
        doc_id: DocumentId,
        indexed_values: Vec<IndexedValue>,
    },
    DeleteDocuments {
        doc_ids: Vec<DocumentId>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionWriteOperation {
    InsertDocument {
        doc_id: DocumentId,
        doc: DocumentToInsert,
    },
    DeleteDocuments {
        doc_ids: Vec<DocumentId>,
    },
    CreateField {
        field_id: FieldId,
        field_name: String,
        field: TypedFieldWrapper,
    },
    Index(DocumentId, FieldId, DocumentFieldIndexOperation),
    CreateIndex2 {
        index_id: IndexId,
        locale: Locale,
    },
    DeleteIndex2 {
        index_id: IndexId,
    },
    IndexWriteOperation(IndexId, IndexWriteOperation),
    InsertDocument2 {
        index_id: IndexId,
        doc_id: DocumentId,
        doc_id_str: String,
        json: String,
    },
    IndexDocument2 {
        index_id: IndexId,
        doc_id: DocumentId,
        indexed_values: Vec<IndexedValue>,
    },
    DeleteDocuments2 {
        index_id: IndexId,
        doc_ids: Vec<DocumentId>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedFieldWrapper {
    Text(Locale),
    Embedding(EmbeddingTypedFieldWrapper),
    Number,
    Bool,
    ArrayText(Locale),
    ArrayNumber,
    ArrayBoolean,
    String,
    ArrayString,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingTypedFieldWrapper {
    pub model: OramaModelSerializable,
    pub document_fields: DocumentFieldsWrapper,
}

#[derive(Debug, Clone)]
pub struct DocumentFieldsWrapper(pub DocumentFields);
impl Serialize for DocumentFieldsWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_tuple(2)?;
        let (id, opt) = match self.0 {
            DocumentFields::AllStringProperties => (1_u8, None),
            DocumentFields::Properties(ref props) => (2, Some(props.clone())),
            DocumentFields::Hook(ref hook) => (3, Some(vec![hook.to_string()])),
        };
        seq.serialize_element(&id)?;
        seq.serialize_element(&opt)?;
        seq.end()
    }
}
impl<'de> Deserialize<'de> for DocumentFieldsWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let a = <(u8, Option<Vec<String>>)>::deserialize(deserializer)?;
        let doc = match a.0 {
            1 => DocumentFields::AllStringProperties,
            2 => DocumentFields::Properties(a.1.unwrap()),
            3 => DocumentFields::Hook(HookName::from_str(&a.1.unwrap()[0]).unwrap()),
            _ => {
                return Err(serde::de::Error::custom(format!(
                    "Invalid tag value: {}",
                    a.0
                )))
            }
        };
        Ok(DocumentFieldsWrapper(doc))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KVWriteOperation {
    Create(String, String),
    Delete(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentStorageWriteOperation {
    InsertDocument {
        doc_id: DocumentId,
        doc: DocumentToInsert,
    },
    DeleteDocuments {
        doc_ids: Vec<DocumentId>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WriteOperation {
    KV(KVWriteOperation),
    CreateCollection {
        id: CollectionId,
        #[serde(
            deserialize_with = "deserialize_api_key",
            serialize_with = "serialize_api_key"
        )]
        read_api_key: ApiKey,
        description: Option<String>,
        default_locale: Locale,
    },
    DeleteCollection(CollectionId),
    Collection(CollectionId, CollectionWriteOperation),
    SubstituteCollection {
        subject_collection_id: CollectionId,
        target_collection_id: CollectionId,
        reference: Option<String>,
    },
    DocumentStorage(DocumentStorageWriteOperation),
}

impl WriteOperation {
    pub fn get_type_id(&self) -> &'static str {
        match self {
            WriteOperation::CreateCollection { .. } => "create_collection",
            WriteOperation::DeleteCollection(_) => "delete_collection",
            WriteOperation::Collection(_, CollectionWriteOperation::CreateField { .. }) => {
                "create_field"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::DeleteDocuments { .. }) => {
                "delete_documents"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::InsertDocument { .. }) => {
                "insert_document"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::Index(_, _, _)) => "index",
            WriteOperation::KV(KVWriteOperation::Create(_, _)) => "kv_create",
            WriteOperation::KV(KVWriteOperation::Delete(_)) => "kv_delete",
            WriteOperation::SubstituteCollection { .. } => "substitute_collection",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::CreateField2 { .. },
                ),
            ) => "create_field_2",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(_, IndexWriteOperation::Index { .. }),
            ) => "index_document_2",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::DeleteDocuments { .. },
                ),
            ) => "delete_document_2",
            WriteOperation::Collection(_, CollectionWriteOperation::InsertDocument2 { .. }) => {
                "insert_document_2"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::CreateIndex2 { .. }) => {
                "create_index"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::DeleteIndex2 { .. }) => {
                "delete_index"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::IndexDocument2 { .. }) => {
                "index_document"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::DeleteDocuments2 { .. }) => {
                "delete_documents_2"
            }
            WriteOperation::DocumentStorage(DocumentStorageWriteOperation::InsertDocument {
                ..
            }) => "document_storage_insert_document",
            WriteOperation::DocumentStorage(DocumentStorageWriteOperation::DeleteDocuments {
                ..
            }) => "document_storage_delete_documents",
        }
    }
}

pub fn serialize_api_key<S>(x: &ApiKey, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    s.serialize_str(x.expose())
}
pub fn deserialize_api_key<'de, D>(deserializer: D) -> Result<ApiKey, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    String::deserialize(deserializer).and_then(|s| {
        ApiKey::try_new(s)
            .map_err(|e| serde::de::Error::custom(format!("Invalid API key: {:?}", e)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::value::RawValue;

    use crate::{
        ai::OramaModel,
        collection_manager::sides::{hooks::HookName, OramaModelSerializable},
        nlp::locales::Locale,
        types::{CollectionId, DocumentId, RawJSONDocument},
    };

    #[test]
    fn test_bincode() {
        let ops = [
            WriteOperation::CreateCollection {
                id: CollectionId::try_new("col").unwrap(),
                read_api_key: ApiKey::try_new("foo").unwrap(),
                description: Some("bar".to_string()),
                default_locale: Locale::AR,
            },
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Text(Locale::IT),
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Bool,
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Embedding(EmbeddingTypedFieldWrapper {
                        document_fields: DocumentFieldsWrapper(DocumentFields::Properties(vec![
                            "foo".to_string(),
                        ])),
                        model: OramaModelSerializable(OramaModel::BgeBase),
                    }),
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Embedding(EmbeddingTypedFieldWrapper {
                        document_fields: DocumentFieldsWrapper(DocumentFields::Properties(vec![
                            "foo".to_string(),
                        ])),
                        model: OramaModelSerializable(OramaModel::BgeBase),
                    }),
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Embedding(EmbeddingTypedFieldWrapper {
                        document_fields: DocumentFieldsWrapper(DocumentFields::AllStringProperties),
                        model: OramaModelSerializable(OramaModel::BgeBase),
                    }),
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Embedding(EmbeddingTypedFieldWrapper {
                        document_fields: DocumentFieldsWrapper(DocumentFields::Hook(
                            HookName::SelectEmbeddingsProperties,
                        )),
                        model: OramaModelSerializable(OramaModel::BgeBase),
                    }),
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Number,
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::CreateField {
                    field_id: FieldId(1),
                    field_name: "Foo".to_string(),
                    field: TypedFieldWrapper::Number,
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::DeleteDocuments {
                    doc_ids: vec![DocumentId(1)],
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::InsertDocument {
                    doc_id: DocumentId(1),
                    doc: DocumentToInsert(RawJSONDocument {
                        id: Some("docid".to_string()),
                        inner: RawValue::from_string("{}".to_string()).unwrap(),
                    }),
                },
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::Index(
                    DocumentId(1),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexString {
                        field_length: 1,
                        terms: HashMap::from([(
                            Term("foo".to_string()),
                            TermStringField { positions: vec![1] },
                        )]),
                    },
                ),
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::Index(
                    DocumentId(1),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexBoolean { value: false },
                ),
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::Index(
                    DocumentId(1),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexNumber {
                        value: NumberWrapper(Number::I32(1)),
                    },
                ),
            ),
            WriteOperation::Collection(
                CollectionId::try_new("col").unwrap(),
                CollectionWriteOperation::Index(
                    DocumentId(1),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexNumber {
                        value: NumberWrapper(Number::F32(1.5)),
                    },
                ),
            ),
        ];

        for op in ops {
            let serialized = bincode::serialize(&op).unwrap();
            let _: WriteOperation = bincode::deserialize(&serialized).unwrap();
        }
    }

    #[test]
    fn test_bincode_index_number() {
        let op = WriteOperation::Collection(
            CollectionId::try_new("col").unwrap(),
            CollectionWriteOperation::Index(
                DocumentId(1),
                FieldId(1),
                DocumentFieldIndexOperation::IndexNumber {
                    value: NumberWrapper(Number::I32(1)),
                },
            ),
        );
        let serialized = bincode::serialize(&op).unwrap();
        let a: WriteOperation = bincode::deserialize(&serialized).unwrap();

        let WriteOperation::Collection(
            _,
            CollectionWriteOperation::Index(
                _,
                _,
                DocumentFieldIndexOperation::IndexNumber { value },
            ),
        ) = a
        else {
            panic!("Expected Index operation");
        };
        assert_eq!(value.0, Number::I32(1));
    }
}
