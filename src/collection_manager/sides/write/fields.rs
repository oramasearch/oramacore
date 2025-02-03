use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    sync::Arc,
};

use anyhow::Result;
use axum_openapi3::utoipa::{openapi::schema::AnyOfBuilder, PartialSchema, ToSchema};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::{
    ai::OramaModel,
    collection_manager::{
        dto::{DocumentFields, FieldId, Number},
        sides::hooks::{HookName, HooksRuntime},
    },
    metrics::{
        Empty, StringCalculationLabels, EMBEDDING_REQUEST_GAUDGE, PENDING_EMBEDDING_REQUEST_GAUDGE,
        STRING_CALCULATION_METRIC,
    },
    nlp::{locales::Locale, TextParser},
    types::{CollectionId, DocumentId, FlattenDocument, ValueType},
};

use super::{
    embedding::{EmbeddingCalculationRequest, EmbeddingCalculationRequestInput},
    CollectionWriteOperation, DocumentFieldIndexOperation, OperationSender, Term, TermStringField,
    WriteOperation,
};

pub type FieldsToIndex = DashMap<String, (ValueType, CollectionField)>;

pub enum CollectionField {
    Number(NumberField),
    Bool(BoolField),
    String(StringField),
    Embedding(EmbeddingField),
}
impl CollectionField {
    pub fn new_number(collection_id: CollectionId, field_id: FieldId, field_name: String) -> Self {
        CollectionField::Number(NumberField::new(collection_id, field_id, field_name, false))
    }

    pub fn new_arr_number(collection_id: CollectionId, field_id: FieldId, field_name: String) -> Self {
        CollectionField::Number(NumberField::new(collection_id, field_id, field_name, true))
    }

    pub fn new_bool(collection_id: CollectionId, field_id: FieldId, field_name: String) -> Self {
        CollectionField::Bool(BoolField::new(collection_id, field_id, field_name))
    }

    pub fn new_string(
        parser: Arc<TextParser>,
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
    ) -> Self {
        CollectionField::String(StringField::new(
            parser,
            collection_id,
            field_id,
            field_name,
            false,
        ))
    }

    pub fn new_arr_string(
        parser: Arc<TextParser>,
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
    ) -> Self {
        CollectionField::String(StringField::new(
            parser,
            collection_id,
            field_id,
            field_name,
            true,
        ))
    }

    pub fn new_embedding(
        model: OramaModel,
        document_fields: DocumentFields,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
        hooks_runtime: Arc<HooksRuntime>,
        collection_id: CollectionId,
        field_id: FieldId,
    ) -> Self {
        CollectionField::Embedding(EmbeddingField::new(
            model,
            document_fields,
            embedding_sender,
            hooks_runtime,
            collection_id,
            field_id,
        ))
    }

    pub fn set_embedding_hook(&mut self, name: HookName) {
        if let CollectionField::Embedding(f) = self {
            f.document_fields = DocumentFields::Hook(name)
        }
    }

    pub async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        match self {
            CollectionField::Number(f) => f.get_write_operations(doc_id, doc, sender).await,
            CollectionField::Bool(f) => f.get_write_operations(doc_id, doc, sender).await,
            CollectionField::String(f) => f.get_write_operations(doc_id, doc, sender).await,
            CollectionField::Embedding(f) => f.get_write_operations(doc_id, doc, sender).await,
        }
    }

    pub fn serialized(&self) -> SerializedFieldIndexer {
        match self {
            CollectionField::Number(f) => f.serialized(),
            CollectionField::Bool(f) => f.serialized(),
            CollectionField::String(f) => f.serialized(),
            CollectionField::Embedding(f) => f.serialized(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OramaModelSerializable(pub OramaModel);

impl Serialize for OramaModelSerializable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.0.as_str_name().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OramaModelSerializable {
    fn deserialize<D>(deserializer: D) -> Result<OramaModelSerializable, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let model_name = String::deserialize(deserializer)?;
        let model = OramaModel::from_str_name(&model_name)
            .ok_or_else(|| serde::de::Error::custom("Invalid model name"))?;
        Ok(OramaModelSerializable(model))
    }
}

impl PartialSchema for OramaModelSerializable {
    fn schema(
    ) -> axum_openapi3::utoipa::openapi::RefOr<axum_openapi3::utoipa::openapi::schema::Schema> {
        let b = AnyOfBuilder::new()
            .item(OramaModel::BgeSmall.as_str_name())
            .item(OramaModel::BgeBase.as_str_name())
            .item(OramaModel::BgeLarge.as_str_name())
            .item(OramaModel::MultilingualE5Small.as_str_name())
            .item(OramaModel::MultilingualE5Base.as_str_name())
            .item(OramaModel::MultilingualE5Large.as_str_name());
        axum_openapi3::utoipa::openapi::RefOr::T(b.into())
    }
}
impl ToSchema for OramaModelSerializable {}

#[derive(Debug, Serialize, Deserialize)]
pub enum SerializedFieldIndexer {
    Number,
    Bool,
    String(Locale),
    Embedding(OramaModelSerializable, DocumentFields),
}

#[derive(Debug)]
pub struct NumberField {
    collection_id: CollectionId,
    field_id: FieldId,
    field_name: String,
    is_array: bool,
}

impl NumberField {
    pub fn new(collection_id: CollectionId, field_id: FieldId, field_name: String, is_array: bool) -> Self {
        Self {
            collection_id,
            field_id,
            field_name,
            is_array,
        }
    }
}

impl NumberField {
    async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        let value = doc.get(&self.field_name);
        let data: Vec<Number> = match value {
            None => return Ok(()),
            Some(value) => {
                if self.is_array {
                    match value.as_array() {
                        None => return Ok(()),
                        Some(value) => {
                            value.iter().filter_map(|v| {
                                Number::try_from(v).ok()
                            }).collect()
                        }
                    }
                } else {
                    if let Ok(v) = Number::try_from(value) {
                        vec![v]
                    } else {
                        return Ok(());
                    }
                }
            }
        };
        if data.is_empty() {
            return Ok(());
        }

        for value in data {
            let op = WriteOperation::Collection(
                self.collection_id.clone(),
                CollectionWriteOperation::Index(
                    doc_id,
                    self.field_id,
                    DocumentFieldIndexOperation::IndexNumber { value },
                ),
            );
            sender.send(op).await?;
        }

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Number
    }
}

#[derive(Debug)]
pub struct BoolField {
    collection_id: CollectionId,
    field_id: FieldId,
    field_name: String,
}

impl BoolField {
    pub fn new(collection_id: CollectionId, field_id: FieldId, field_name: String) -> Self {
        Self {
            collection_id,
            field_id,
            field_name,
        }
    }

    async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        let value = doc.get(&self.field_name);

        let value = match value {
            None => return Ok(()),
            Some(value) => match value.as_bool() {
                // If the document has a field with the name `field_name` but the value isn't a boolean
                // we ignore it.
                // Should we bubble up an error?
                // TODO: think about it
                None => return Ok(()),
                Some(value) => value,
            },
        };

        let op = WriteOperation::Collection(
            self.collection_id.clone(),
            CollectionWriteOperation::Index(
                doc_id,
                self.field_id,
                DocumentFieldIndexOperation::IndexBoolean { value },
            ),
        );

        sender.send(op).await?;

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Bool
    }
}

#[derive(Debug)]
pub struct StringField {
    collection_id: CollectionId,
    field_id: FieldId,
    field_name: String,
    parser: Arc<TextParser>,
    is_array: bool,
}
impl StringField {
    pub fn new(
        parser: Arc<TextParser>,
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
        is_array: bool,
    ) -> Self {
        Self {
            parser,
            collection_id,
            field_id,
            field_name,
            is_array,
        }
    }

    pub async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        let metric = STRING_CALCULATION_METRIC.create(StringCalculationLabels {
            collection: self.collection_id.0.clone(),
            field: self.field_name.to_string(),
        });

        let value = doc.get(&self.field_name);
        let data = match value {
            None => return Ok(()),
            Some(value) => {
                if self.is_array {
                    match value.as_array() {
                        None => return Ok(()),
                        Some(value) => {
                            let mut data = Vec::new();
                            for value in value {
                                if let Some(value) = value.as_str() {
                                    data.extend(self.parser.tokenize_and_stem(value));
                                }
                            }
                            data
                        }
                    }
                } else {
                    match value.as_str() {
                        None => return Ok(()),
                        Some(value) => self.parser.tokenize_and_stem(value),
                    }
                }
            }
        };

        let field_length = data.len().min(u16::MAX as usize - 1) as u16;

        let mut terms: HashMap<Term, TermStringField> = Default::default();
        for (position, (original, stemmeds)) in data.into_iter().enumerate() {
            // This `for` loop wants to build the `terms` hashmap
            // it is a `HashMap<String, (u32, HashMap<(DocumentId, FieldId), Posting>)>`
            // that means we:
            // term as string -> (term count, HashMap<(DocumentId, FieldId), Posting>)
            // Here we don't want to store Posting into PostingListStorage,
            // that is business of the IndexReader.
            // Instead, here we want to extrapolate data from the document.
            // The real storage leaves in the IndexReader.
            // `original` & `stemmeds` appears in the `terms` hashmap with the "same value"
            // ie: the position of the origin and stemmed term are the same.

            let original = Term(original);
            match terms.entry(original) {
                Entry::Occupied(mut entry) => {
                    let p: &mut TermStringField = entry.get_mut();

                    p.positions.push(position);
                }
                Entry::Vacant(entry) => {
                    let p = TermStringField {
                        positions: vec![position],
                    };
                    entry.insert(p);
                }
            };

            for stemmed in stemmeds {
                let stemmed = Term(stemmed);
                match terms.entry(stemmed) {
                    Entry::Occupied(mut entry) => {
                        let p: &mut TermStringField = entry.get_mut();
                        p.positions.push(position);
                    }
                    Entry::Vacant(entry) => {
                        let p = TermStringField {
                            positions: vec![position],
                        };
                        entry.insert(p);
                    }
                };
            }
        }

        drop(metric);

        let op = WriteOperation::Collection(
            self.collection_id.clone(),
            CollectionWriteOperation::Index(
                doc_id,
                self.field_id,
                DocumentFieldIndexOperation::IndexString {
                    field_length,
                    terms,
                },
            ),
        );

        sender.send(op).await?;

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::String(self.parser.locale())
    }
}

#[derive(Debug)]
pub struct EmbeddingField {
    collection_id: CollectionId,
    field_id: FieldId,

    model: OramaModel,
    document_fields: DocumentFields,
    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
    hooks_runtime: Arc<HooksRuntime>,
}

impl EmbeddingField {
    pub fn new(
        model: OramaModel,
        document_fields: DocumentFields,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
        hooks_runtime: Arc<HooksRuntime>,
        collection_id: CollectionId,
        field_id: FieldId,
    ) -> Self {
        Self {
            model,
            document_fields,
            embedding_sender,
            hooks_runtime,
            collection_id,
            field_id,
        }
    }
}

impl EmbeddingField {
    async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        let input: String = match &self.document_fields {
            DocumentFields::Properties(v) => v
                .iter()
                .filter_map(|field_name| {
                    let value = doc.get(field_name).and_then(|v| v.as_str());
                    value
                })
                .collect(),
            DocumentFields::Hook(hook_name) => {
                let hook_exec_result = self
                    .hooks_runtime
                    .eval(self.collection_id.clone(), hook_name.clone(), doc.clone()) // @todo: make sure we pass unflatten document here
                    .await;

                let input: SelectEmbeddingPropertiesReturnType = match hook_exec_result {
                    Some(Ok(input)) => input,
                    _ => return Ok(()),
                };

                match input {
                    SelectEmbeddingPropertiesReturnType::Properties(v) => v
                        .iter()
                        .filter_map(|field_name| {
                            let value = doc.get(field_name).and_then(|v| v.as_str());
                            value
                        })
                        .collect(),
                    SelectEmbeddingPropertiesReturnType::Text(v) => v,
                }
            }
            DocumentFields::AllStringProperties => {
                let mut input = String::new();
                for (_, value) in doc.iter() {
                    if let Some(value) = value.as_str() {
                        input.push_str(value);
                    }
                }
                input
            }
        };

        // The input could be:
        // - empty: we should skip this (???)
        // - "normal": it is ok
        // - "too long": we should chunk it in a smart way
        // TODO: implement that logic

        PENDING_EMBEDDING_REQUEST_GAUDGE
            .create(Empty {})
            .increment_by_one();
        self.embedding_sender
            .send(EmbeddingCalculationRequest {
                model: self.model,
                input: EmbeddingCalculationRequestInput {
                    text: input,
                    coll_id: self.collection_id.clone(),
                    doc_id,
                    field_id: self.field_id,
                    op_sender: sender,
                },
            })
            .await?;
        PENDING_EMBEDDING_REQUEST_GAUDGE
            .create(Empty {})
            .decrement_by_one();
        EMBEDDING_REQUEST_GAUDGE.create(Empty {}).increment_by_one();

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Embedding(
            OramaModelSerializable(self.model),
            self.document_fields.clone(),
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum SelectEmbeddingPropertiesReturnType {
    Properties(Vec<String>),
    Text(String),
}
