use crate::{
    collection_manager::sides::write::index::{EnumStrategy, IndexedValue},
    python::embeddings::Model,
    types::{
        ApiKey, CollectionId, DocumentFields, DocumentId, FieldId, IndexId, Number, RawJSONDocument,
    },
};
use oramacore_lib::generic_kv::KVWriteOperation;
use oramacore_lib::hook_storage::HookOperation;
use oramacore_lib::nlp::locales::Locale;
use oramacore_lib::pin_rules::PinRuleOperation;
use oramacore_lib::shelves::ShelfOperation;
use serde::{ser::SerializeTuple, Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Term(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermStringField {
    pub exact_positions: Vec<usize>,
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
    Date,
    GeoPoint,
    String,
    Embedding(Model),
}

#[derive(Clone, Serialize, Deserialize)]
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
    /// Index operation with optional OCM (Orama Custom Multiplier) support.
    /// This variant is used for new documents and is backward compatible with Index.
    Index2 {
        doc_id: DocumentId,
        indexed_values: Vec<IndexedValue>,
        /// Optional custom multiplier applied to the document's score during search.
        /// Only positive values are valid; zero, negative, or absent values are ignored.
        ocm: Option<f32>,
    },
    IndexEmbedding {
        data: EmbeddingIndexData,
    },
    DeleteDocuments {
        doc_ids: Vec<DocumentId>,
    },
    PinRule(PinRuleOperation<DocumentId>),
}

pub type EmbeddingIndexData = Vec<(FieldId, Vec<(DocumentId, Vec<Vec<f32>>)>)>;

impl Debug for IndexWriteOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateField2 {
                field_id,
                field_path,
                is_array,
                field_type,
            } => f
                .debug_struct("CreateField2")
                .field("field_id", field_id)
                .field("field_path", field_path)
                .field("is_array", is_array)
                .field("field_type", field_type)
                .finish(),
            Self::Index {
                doc_id,
                indexed_values,
            } => f
                .debug_struct("Index")
                .field("doc_id", doc_id)
                .field("indexed_values", indexed_values)
                .finish(),
            Self::Index2 {
                doc_id,
                indexed_values,
                ocm,
            } => f
                .debug_struct("Index2")
                .field("doc_id", doc_id)
                .field("indexed_values", indexed_values)
                .field("ocm", ocm)
                .finish(),
            Self::IndexEmbedding { .. } => {
                f.debug_struct("IndexEmbedding")
                    // Avoid log embedding
                    .finish_non_exhaustive()
            }
            Self::PinRule(rule) => f.debug_struct("PinRule").field("rule", rule).finish(),
            Self::DeleteDocuments { doc_ids } => f
                .debug_struct("DeleteDocuments")
                .field("doc_ids", doc_ids)
                .finish(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplaceIndexReason {
    IndexResynced,
    CollectionReindexed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionWriteOperation {
    Hook(HookOperation),
    CreateIndex2 {
        index_id: IndexId,
        locale: Locale,
    },
    CreateIndex3 {
        index_id: IndexId,
        locale: Locale,
        enum_strategy: EnumStrategy,
    },
    CreateTemporaryIndex2 {
        index_id: IndexId,
        locale: Locale,
    },
    CreateTemporaryIndex3 {
        index_id: IndexId,
        locale: Locale,
        enum_strategy: EnumStrategy,
    },
    ReplaceIndex {
        reason: ReplaceIndexReason,
        runtime_index_id: IndexId,
        temp_index_id: IndexId,
        reference: Option<String>,
    },
    DeleteIndex2 {
        index_id: IndexId,
    },
    DeleteTempIndex {
        temp_index_id: IndexId,
    },
    IndexWriteOperation(IndexId, IndexWriteOperation),
    UpdateMcpDescription {
        mcp_description: Option<String>,
    },
    PinRule(PinRuleOperation<DocumentId>),
    Shelf(ShelfOperation<DocumentId>),
    DocumentStorage(DocumentStorageWriteOperation),
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
    pub model: Model,
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
            // DocumentFields::Hook(ref hook) => (3, Some(vec![hook.to_string()])),
            DocumentFields::Automatic => (4, None),
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
            // 3 => DocumentFields::Hook(HookName::from_str(&a.1.unwrap()[0]).unwrap()),
            4 => DocumentFields::Automatic,
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
pub enum DocumentStorageWriteOperation {
    InsertDocument {
        doc_id: DocumentId,
        doc: DocumentToInsert,
    },
    InsertDocuments(Vec<(DocumentId, DocumentToInsert)>),
    DeleteDocuments {
        doc_ids: Vec<DocumentId>,
    },
    // New variants for per-collection storage with document_id_str
    InsertDocumentWithDocIdStr {
        doc_id: DocumentId,
        doc_id_str: String,
        doc: DocumentToInsert,
    },
    InsertDocumentsWithDocIdStr(Vec<(DocumentId, String, DocumentToInsert)>),
    DeleteDocumentsWithDocIdStr(Vec<(DocumentId, String)>),
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
        mcp_description: Option<String>,
        default_locale: Locale,
    },
    CreateCollection2 {
        id: CollectionId,
        #[serde(
            deserialize_with = "deserialize_api_key",
            serialize_with = "serialize_api_key"
        )]
        read_api_key: ApiKey,
        #[serde(
            deserialize_with = "deserialize_api_key",
            serialize_with = "serialize_api_key"
        )]
        write_api_key: ApiKey,
        description: Option<String>,
        mcp_description: Option<String>,
        default_locale: Locale,
    },
    DeleteCollection(CollectionId),
    Collection(CollectionId, CollectionWriteOperation),
    #[deprecated(note = "Use CollectionWriteOperation::DocumentStorage instead")]
    #[allow(deprecated)]
    DocumentStorage(DocumentStorageWriteOperation),
}

impl WriteOperation {
    pub fn get_type_id(&self) -> &'static str {
        match self {
            WriteOperation::CreateCollection { .. } => "create_collection",
            WriteOperation::CreateCollection2 { .. } => "create_collection2",
            WriteOperation::DeleteCollection(_) => "delete_collection",
            WriteOperation::KV(KVWriteOperation::Create(_, _)) => "kv_create",
            WriteOperation::KV(KVWriteOperation::Delete(_)) => "kv_delete",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::CreateField2 { .. },
                ),
            ) => "create_field_2",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::UpdateMcpDescription { .. },
            ) => "update_mcp_description",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(_, IndexWriteOperation::Index { .. }),
            ) => "index_document_2",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::Index2 { .. },
                ),
            ) => "index_document_2",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::DeleteDocuments { .. },
                ),
            ) => "delete_document_2",
            WriteOperation::Collection(_, CollectionWriteOperation::CreateIndex2 { .. }) => {
                "create_index"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::CreateIndex3 { .. }) => {
                "create_index"
            }
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Hook(HookOperation::Delete(_)),
            ) => "delete_hook",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Hook(HookOperation::Insert(_, _)),
            ) => "insert_hook",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::CreateTemporaryIndex2 { .. },
            ) => "create_temp_index",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::CreateTemporaryIndex3 { .. },
            ) => "create_temp_index",
            WriteOperation::Collection(_, CollectionWriteOperation::ReplaceIndex { .. }) => {
                "substitute_collection"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::DeleteIndex2 { .. }) => {
                "delete_index"
            }
            WriteOperation::Collection(_, CollectionWriteOperation::DeleteTempIndex { .. }) => {
                "delete_expired_temp_index"
            }
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::IndexEmbedding { .. },
                ),
            ) => "index_embedding",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::PinRule(PinRuleOperation::Insert(_)),
                ),
            ) => "insert_pin_rule",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::IndexWriteOperation(
                    _,
                    IndexWriteOperation::PinRule(PinRuleOperation::Delete(_)),
                ),
            ) => "delete_pin_rule",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::PinRule(PinRuleOperation::Insert(_)),
            ) => "insert_merchandising_pin_rule",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::PinRule(PinRuleOperation::Delete(_)),
            ) => "delete_merchandising_pin_rule",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Shelf(ShelfOperation::Insert(_)),
            ) => "insert_merchandising_shelf",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Shelf(ShelfOperation::Delete(_)),
            ) => "delete_merchandising_shelf",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::InsertDocument { .. },
                ),
            ) => "collection_document_storage_insert_document",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::InsertDocuments { .. },
                ),
            ) => "collection_document_storage_insert_documents",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::DeleteDocuments { .. },
                ),
            ) => "collection_document_storage_delete_documents",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::InsertDocumentWithDocIdStr { .. },
                ),
            ) => "collection_document_storage_insert_document_with_doc_id_str",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::InsertDocumentsWithDocIdStr { .. },
                ),
            ) => "collection_document_storage_insert_documents_with_doc_id_str",
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::DeleteDocumentsWithDocIdStr { .. },
                ),
            ) => "collection_document_storage_delete_documents_with_doc_id_str",
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(DocumentStorageWriteOperation::InsertDocument {
                ..
            }) => "document_storage_insert_document",
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(DocumentStorageWriteOperation::InsertDocuments(_)) => {
                "document_storage_insert_documents"
            }
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(DocumentStorageWriteOperation::DeleteDocuments {
                ..
            }) => "document_storage_delete_documents",
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(
                DocumentStorageWriteOperation::InsertDocumentWithDocIdStr { .. },
            ) => "document_storage_insert_document_with_doc_id_str",
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(
                DocumentStorageWriteOperation::InsertDocumentsWithDocIdStr(_),
            ) => "document_storage_insert_documents_with_doc_id_str",
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(
                DocumentStorageWriteOperation::DeleteDocumentsWithDocIdStr(_),
            ) => "document_storage_delete_documents_with_doc_id_str",
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
        ApiKey::try_new(s).map_err(|e| serde::de::Error::custom(format!("Invalid API key: {e:?}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::value::RawValue;

    use crate::types::{CollectionId, DocumentId, RawJSONDocument, SerializableNumber};
    use oramacore_lib::nlp::locales::Locale;

    #[test]
    fn test_bincode() {
        let collection_id = CollectionId::try_new("col").unwrap();
        let index_id = IndexId::try_new("index").unwrap();
        let locale = Locale::EN;
        let field_id = FieldId(1);
        let field_path = vec!["foo".to_string(), "bar".to_string()].into_boxed_slice();

        let ops = [
            WriteOperation::CreateCollection {
                id: collection_id,
                read_api_key: ApiKey::try_new("foo").unwrap(),
                description: Some("bar".to_string()),
                mcp_description: None,
                default_locale: locale,
            },
            WriteOperation::DeleteCollection(collection_id),
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(DocumentStorageWriteOperation::DeleteDocuments {
                doc_ids: vec![DocumentId(2)],
            }),
            #[allow(deprecated)]
            WriteOperation::DocumentStorage(DocumentStorageWriteOperation::InsertDocument {
                doc_id: DocumentId(1),
                doc: DocumentToInsert(RawJSONDocument {
                    id: Some("foo".to_string()),
                    inner: RawValue::from_string("{\"foo\": \"bar\"}".to_string()).unwrap(),
                }),
            }),
            WriteOperation::KV(KVWriteOperation::Create(
                "key".to_string(),
                "value".to_string(),
            )),
            WriteOperation::KV(KVWriteOperation::Delete("key".to_string())),
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::CreateIndex2 { index_id, locale },
            ),
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::CreateTemporaryIndex2 { index_id, locale },
            ),
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::ReplaceIndex {
                    reason: ReplaceIndexReason::CollectionReindexed,
                    runtime_index_id: index_id,
                    temp_index_id: index_id,
                    reference: Some("doo".to_string()),
                },
            ),
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    index_id,
                    IndexWriteOperation::CreateField2 {
                        field_id,
                        field_path: field_path.clone(),
                        is_array: false,
                        field_type: IndexWriteOperationFieldType::Embedding(Model::BGESmall),
                    },
                ),
            ),
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    index_id,
                    IndexWriteOperation::Index {
                        doc_id: DocumentId(1),
                        indexed_values: Vec::from([
                            IndexedValue::FilterBool(field_id, true),
                            IndexedValue::FilterNumber(
                                field_id,
                                SerializableNumber(Number::F32(1.0)),
                            ),
                            IndexedValue::FilterString(field_id, "foo".to_string()),
                            IndexedValue::ScoreString(
                                field_id,
                                1,
                                HashMap::from([
                                    (
                                        Term("foo".to_string()),
                                        TermStringField {
                                            positions: vec![1],
                                            exact_positions: vec![3],
                                        },
                                    ),
                                    (
                                        Term("bar".to_string()),
                                        TermStringField {
                                            positions: vec![2],
                                            exact_positions: vec![4],
                                        },
                                    ),
                                ]),
                            ),
                        ]),
                    },
                ),
            ),
            // Test Index2 with OCM (Orama Custom Multiplier)
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    index_id,
                    IndexWriteOperation::Index2 {
                        doc_id: DocumentId(2),
                        indexed_values: Vec::from([IndexedValue::FilterBool(field_id, false)]),
                        ocm: Some(2.0),
                    },
                ),
            ),
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    index_id,
                    IndexWriteOperation::DeleteDocuments {
                        doc_ids: vec![DocumentId(1)],
                    },
                ),
            ),
            WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    index_id,
                    IndexWriteOperation::IndexEmbedding {
                        data: vec![(field_id, vec![(DocumentId(1), vec![vec![1.0, 2.0, 3.0]])])],
                    },
                ),
            ),
        ];

        for op in ops {
            let serialized = bincode::serialize(&op).unwrap();
            let _: WriteOperation = bincode::deserialize(&serialized).unwrap();
        }
    }
}
