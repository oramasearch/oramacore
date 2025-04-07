use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    sync::Arc,
};

use anyhow::{Context, Result};
use axum_openapi3::utoipa::{openapi::schema::AnyOfBuilder, PartialSchema, ToSchema};
use serde::{Deserialize, Serialize};

use crate::{
    ai::OramaModel,
    collection_manager::{
        dto::{DocumentFields, FieldId, Number},
        sides::{
            hooks::{HookName, HooksRuntime, SelectEmbeddingPropertiesReturnType},
            CollectionWriteOperation, DocumentFieldIndexOperation, NumberWrapper, OperationSender,
            Term, TermStringField, WriteOperation,
        },
    },
    types::{CollectionId, DocumentId, FlattenDocument, ValueType},
};
use nlp::{
    chunker::{Chunker, ChunkerConfig},
    locales::Locale,
    TextParser,
};

use super::embedding::{EmbeddingCalculationRequest, EmbeddingCalculationRequestInput};

pub enum CollectionFilterField {
    Number(NumberFilterField),
    Bool(BoolFilterField),
    String(StringFilterField),
}

impl CollectionFilterField {
    pub fn new_number(collection_id: CollectionId, field_id: FieldId, field_name: String) -> Self {
        CollectionFilterField::Number(NumberFilterField::new(
            collection_id,
            field_id,
            field_name,
            false,
        ))
    }

    pub fn new_arr_number(
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
    ) -> Self {
        CollectionFilterField::Number(NumberFilterField::new(
            collection_id,
            field_id,
            field_name,
            true,
        ))
    }

    pub fn new_bool(collection_id: CollectionId, field_id: FieldId, field_name: String) -> Self {
        CollectionFilterField::Bool(BoolFilterField::new(
            collection_id,
            field_id,
            field_name,
            false,
        ))
    }

    pub fn new_arr_bool(
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
    ) -> Self {
        CollectionFilterField::Bool(BoolFilterField::new(
            collection_id,
            field_id,
            field_name,
            true,
        ))
    }

    pub fn new_string(collection_id: CollectionId, field_id: FieldId, field_name: String) -> Self {
        CollectionFilterField::String(StringFilterField::new(
            collection_id,
            field_id,
            field_name,
            false,
        ))
    }

    pub fn new_arr_string(
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
    ) -> Self {
        CollectionFilterField::String(StringFilterField::new(
            collection_id,
            field_id,
            field_name,
            true,
        ))
    }

    pub fn field_name(&self) -> String {
        match self {
            CollectionFilterField::Number(f) => f.field_name.clone(),
            CollectionFilterField::Bool(f) => f.field_name.clone(),
            CollectionFilterField::String(f) => f.field_name.clone(),
        }
    }

    pub fn field_type(&self) -> ValueType {
        match self {
            CollectionFilterField::Number(_) => ValueType::Scalar(crate::types::ScalarType::Number),
            CollectionFilterField::Bool(_) => ValueType::Scalar(crate::types::ScalarType::Boolean),
            CollectionFilterField::String(_) => ValueType::Scalar(crate::types::ScalarType::String),
        }
    }

    pub fn field_type_str(&self) -> &'static str {
        match self {
            CollectionFilterField::Number(_) => "number_filter",
            CollectionFilterField::Bool(_) => "bool_filter",
            CollectionFilterField::String(_) => "string_filter",
        }
    }

    pub async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        match self {
            CollectionFilterField::Number(f) => f.get_write_operations(doc_id, doc, sender).await,
            CollectionFilterField::Bool(f) => f.get_write_operations(doc_id, doc, sender).await,
            CollectionFilterField::String(f) => f.get_write_operations(doc_id, doc, sender).await,
        }
    }

    pub fn serialized(&self) -> SerializedFieldIndexer {
        match self {
            CollectionFilterField::Number(f) => f.serialized(),
            CollectionFilterField::Bool(f) => f.serialized(),
            CollectionFilterField::String(f) => f.serialized(),
        }
    }
}

pub enum CollectionScoreField {
    String(StringField),
    Embedding(EmbeddingField),
}

impl CollectionScoreField {
    pub fn new_string(
        parser: Arc<TextParser>,
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
    ) -> Self {
        Self::String(StringField::new(
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
        Self::String(StringField::new(
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
        field_name: String,
    ) -> Self {
        Self::Embedding(EmbeddingField::new(
            model,
            document_fields,
            embedding_sender,
            hooks_runtime,
            collection_id,
            field_id,
            field_name,
        ))
    }

    pub fn set_embedding_hook(&mut self, name: HookName) {
        if let Self::Embedding(f) = self {
            f.document_fields = DocumentFields::Hook(name)
        }
    }

    pub fn field_name(&self) -> String {
        match self {
            CollectionScoreField::String(f) => f.field_name.clone(),
            CollectionScoreField::Embedding(f) => f.field_name.clone(),
        }
    }

    pub fn field_type_str(&self) -> &'static str {
        match self {
            CollectionScoreField::String(_) => "string",
            CollectionScoreField::Embedding(_) => "embedding",
        }
    }

    pub fn field_type(&self) -> ValueType {
        match self {
            CollectionScoreField::String(_) => ValueType::Scalar(crate::types::ScalarType::String),
            CollectionScoreField::Embedding(_) => {
                ValueType::Complex(crate::types::ComplexType::Embedding)
            }
        }
    }

    pub async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        match self {
            CollectionScoreField::String(f) => f.get_write_operations(doc_id, doc, sender).await,
            CollectionScoreField::Embedding(f) => f.get_write_operations(doc_id, doc, sender).await,
        }
    }

    pub fn serialized(&self) -> SerializedFieldIndexer {
        match self {
            CollectionScoreField::String(f) => f.serialized(),
            CollectionScoreField::Embedding(f) => f.serialized(),
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
    StringFilter,
    String(Locale),
    Embedding(OramaModelSerializable, DocumentFields),
}

#[derive(Debug)]
pub struct NumberFilterField {
    collection_id: CollectionId,
    field_id: FieldId,
    field_name: String,
    is_array: bool,
}

impl NumberFilterField {
    pub fn new(
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
        is_array: bool,
    ) -> Self {
        Self {
            collection_id,
            field_id,
            field_name,
            is_array,
        }
    }
}

impl NumberFilterField {
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
                        Some(value) => value
                            .iter()
                            .filter_map(|v| Number::try_from(v).ok())
                            .collect(),
                    }
                } else if let Ok(v) = Number::try_from(value) {
                    vec![v]
                } else {
                    return Ok(());
                }
            }
        };
        if data.is_empty() {
            return Ok(());
        }

        for value in data {
            let op = WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::Index(
                    doc_id,
                    self.field_id,
                    DocumentFieldIndexOperation::IndexNumber {
                        value: NumberWrapper(value),
                    },
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
pub struct BoolFilterField {
    collection_id: CollectionId,
    field_id: FieldId,
    field_name: String,
    is_array: bool,
}

impl BoolFilterField {
    pub fn new(
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
        is_array: bool,
    ) -> Self {
        Self {
            collection_id,
            field_id,
            field_name,
            is_array,
        }
    }

    async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        let value = doc.get(&self.field_name);
        let data: Vec<bool> = match value {
            None => return Ok(()),
            Some(value) => {
                if self.is_array {
                    match value.as_array() {
                        None => return Ok(()),
                        Some(value) => value.iter().filter_map(|v| v.as_bool()).collect(),
                    }
                } else if let Some(v) = value.as_bool() {
                    vec![v]
                } else {
                    // If the document has a field with the name `field_name` but the value isn't a boolean
                    // we ignore it.
                    // Should we bubble up an error?
                    // TODO: think about it
                    return Ok(());
                }
            }
        };

        for value in data {
            let op = WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::Index(
                    doc_id,
                    self.field_id,
                    DocumentFieldIndexOperation::IndexBoolean { value },
                ),
            );

            sender.send(op).await?;
        }

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Bool
    }
}

#[derive(Debug)]
pub struct StringFilterField {
    collection_id: CollectionId,
    field_id: FieldId,
    field_name: String,
    is_array: bool,
}

impl StringFilterField {
    pub fn new(
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
        is_array: bool,
    ) -> Self {
        Self {
            collection_id,
            field_id,
            field_name,
            is_array,
        }
    }

    async fn get_write_operations(
        &self,
        doc_id: DocumentId,
        doc: &FlattenDocument,
        sender: OperationSender,
    ) -> Result<()> {
        let value = doc.get(&self.field_name);
        let data: Vec<String> = match value {
            None => return Ok(()),
            Some(value) => {
                if self.is_array {
                    match value.as_array() {
                        None => return Ok(()),
                        Some(value) => value
                            .iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect(),
                    }
                } else if let Some(v) = value.as_str() {
                    vec![v.to_string()]
                } else {
                    // If the document has a field with the name `field_name` but the value isn't a string
                    // we ignore it.
                    // Should we bubble up an error?
                    // TODO: think about it
                    return Ok(());
                }
            }
        };

        for value in data {
            // TODO: put this "25" in the collection config
            if value.len() < 25 {
                let op = WriteOperation::Collection(
                    self.collection_id,
                    CollectionWriteOperation::Index(
                        doc_id,
                        self.field_id,
                        DocumentFieldIndexOperation::IndexStringFilter { value },
                    ),
                );

                sender.send(op).await?;
            }
        }

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::StringFilter
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

        let op = WriteOperation::Collection(
            self.collection_id,
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

pub struct EmbeddingField {
    collection_id: CollectionId,
    field_id: FieldId,
    field_name: String,

    model: OramaModel,
    document_fields: DocumentFields,
    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
    hooks_runtime: Arc<HooksRuntime>,

    chunker: Chunker,
}

impl OramaModel {
    pub fn senquence_length(&self) -> usize {
        //
        // From Michele slack message:
        // https://oramasearch.slack.com/archives/D0571JYV5LK/p1742488393750479
        // ```
        // intfloat/multilingual-e5-small: 512
        // intfloat/multilingual-e5-base: 512
        // intfloat/multilingual-e5-large: 512
        // BAAI/bge-small-en: 512
        // BAAI/bge-base-en: 512
        // BAAI/bge-large-en: 512
        // sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: 128
        // jinaai/jina-embeddings-v2-base-code: 8000 (ma fai comunque massimo 512 o 1024)
        // ```
        match self {
            OramaModel::MultilingualE5Small => 512,
            OramaModel::MultilingualE5Base => 512,
            OramaModel::MultilingualE5Large => 512,
            OramaModel::BgeSmall => 512,
            OramaModel::BgeBase => 512,
            OramaModel::BgeLarge => 512,
            OramaModel::MultilingualMiniLml12v2 => 128,
            OramaModel::JinaEmbeddingsV2BaseCode => 512,
        }
    }

    pub fn overlap(&self) -> usize {
        // https://oramasearch.slack.com/archives/D0571JYV5LK/p1742488431564979
        self.senquence_length() * 2 / 100
    }
}

impl EmbeddingField {
    pub fn new(
        model: OramaModel,
        document_fields: DocumentFields,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
        hooks_runtime: Arc<HooksRuntime>,
        collection_id: CollectionId,
        field_id: FieldId,
        field_name: String,
    ) -> Self {
        let max_tokens = model.senquence_length();
        let overlap = model.overlap();

        let chunker = Chunker::try_new(ChunkerConfig {
            max_tokens,
            overlap: Some(overlap),
        })
        .expect("Hardcoded data are valid");

        Self {
            model,
            document_fields,
            embedding_sender,
            hooks_runtime,
            collection_id,
            field_id,
            field_name,
            chunker,
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
            DocumentFields::Hook(_) => {
                let hook_exec_result = self
                    .hooks_runtime
                    .calculate_text_for_embedding(self.collection_id, doc.clone()) // @todo: make sure we pass unflatten document here
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
                for (k, value) in doc.iter() {
                    // Don't include the id in the input
                    if k == "id" {
                        continue;
                    }
                    if let Some(value) = value.as_str() {
                        input.push_str(value);
                    }
                }
                input
            }
        };

        // The input could be:
        // - empty: we should skip this
        // - "normal": it is ok
        // - "too long": we should chunk it in a smart way

        if input.trim().is_empty() {
            return Ok(());
        }

        let chunked_data = self.chunker.chunk_text(&input);

        let reserved = self
            .embedding_sender
            .reserve_many(chunked_data.len())
            .await
            .context("Unable to reserve space for embedding")?;
        for (chunk, reserve) in chunked_data.into_iter().zip(reserved) {
            reserve.send(EmbeddingCalculationRequest {
                model: self.model,
                input: EmbeddingCalculationRequestInput {
                    text: chunk,
                    coll_id: self.collection_id,
                    doc_id,
                    field_id: self.field_id,
                    op_sender: sender.clone(),
                },
            });
        }

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Embedding(
            OramaModelSerializable(self.model),
            self.document_fields.clone(),
        )
    }
}
