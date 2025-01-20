use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    sync::Arc,
};

use anyhow::Result;
use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast::Sender;

use crate::{
    collection_manager::dto::{DocumentFields, FieldId},
    indexes::number::Number,
    metrics::{StringCalculationLabels, STRING_CALCULATION_METRIC},
    nlp::{locales::Locale, TextParser},
    types::{CollectionId, DocumentId, FlattenDocument, ValueType},
};

use super::{
    embedding::{EmbeddingCalculationRequest, EmbeddingCalculationRequestInput},
    CollectionWriteOperation, DocumentFieldIndexOperation, Term, TermStringField, WriteOperation,
};

pub type FieldsToIndex = DashMap<String, (ValueType, Arc<Box<dyn FieldIndexer>>)>;

#[derive(Debug, Serialize, Deserialize)]
pub enum SerializedFieldIndexer {
    Number,
    Bool,
    String(Locale),
    Embedding(String, DocumentFields),
}

#[async_trait]
pub trait FieldIndexer: Sync + Send + Debug {
    async fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: String,
        field_id: FieldId,
        doc: &FlattenDocument,
        sender: Sender<WriteOperation>,
    ) -> Result<()>;

    fn serialized(&self) -> SerializedFieldIndexer;
}

#[derive(Debug)]
pub struct NumberField {}

impl Default for NumberField {
    fn default() -> Self {
        Self::new()
    }
}

impl NumberField {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl FieldIndexer for NumberField {
    async fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: String,
        field_id: FieldId,
        doc: &FlattenDocument,
        sender: Sender<WriteOperation>,
    ) -> Result<()> {
        let value = doc.get(&field_name).and_then(|v| Number::try_from(v).ok());

        let value = match value {
            None => return Ok(()),
            Some(value) => value,
        };

        let op = WriteOperation::Collection(
            coll_id,
            CollectionWriteOperation::Index(
                doc_id,
                field_id,
                DocumentFieldIndexOperation::IndexNumber { value },
            ),
        );

        sender.send(op)?;

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Number
    }
}

#[derive(Debug)]
pub struct BoolField {}

impl Default for BoolField {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolField {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl FieldIndexer for BoolField {
    async fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: String,
        field_id: FieldId,
        doc: &FlattenDocument,
        sender: Sender<WriteOperation>,
    ) -> Result<()> {
        let value = doc.get(&field_name);

        let value = match value {
            None => return Ok(()),
            Some(value) => match value.as_bool() {
                None => return Ok(()),
                Some(value) => value,
            },
        };

        let op = WriteOperation::Collection(
            coll_id,
            CollectionWriteOperation::Index(
                doc_id,
                field_id,
                DocumentFieldIndexOperation::IndexBoolean { value },
            ),
        );

        sender.send(op)?;

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Bool
    }
}

#[derive(Debug)]
pub struct StringField {
    parser: Arc<TextParser>,
}
impl StringField {
    pub fn new(parser: Arc<TextParser>) -> Self {
        Self { parser }
    }
}

#[async_trait]
impl FieldIndexer for StringField {
    async fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: String,
        field_id: FieldId,
        doc: &FlattenDocument,
        sender: Sender<WriteOperation>,
    ) -> Result<()> {
        let metric = STRING_CALCULATION_METRIC.create(StringCalculationLabels {
            collection: coll_id.0.clone(),
            field: field_name.to_string(),
        });

        let value = doc.get(&field_name);

        let data = match value {
            None => return Ok(()),
            Some(value) => match value.as_str() {
                None => return Ok(()),
                Some(value) => self.parser.tokenize_and_stem(value),
            },
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
            coll_id,
            CollectionWriteOperation::Index(
                doc_id,
                field_id,
                DocumentFieldIndexOperation::IndexString {
                    field_length,
                    terms,
                },
            ),
        );

        sender.send(op)?;

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::String(self.parser.locale())
    }
}

#[derive(Debug)]
pub struct EmbeddingField {
    model_name: String,
    document_fields: DocumentFields,
    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
}

impl EmbeddingField {
    pub fn new(
        model_name: String,
        document_fields: DocumentFields,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
    ) -> Self {
        Self {
            model_name,
            document_fields,
            embedding_sender,
        }
    }
}

#[async_trait]
impl FieldIndexer for EmbeddingField {
    async fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        _field_name: String,
        field_id: FieldId,
        doc: &FlattenDocument,
        sender: Sender<WriteOperation>,
    ) -> Result<()> {
        let input: String = match &self.document_fields {
            DocumentFields::Properties(v) => v
                .iter()
                .filter_map(|field_name| {
                    let value = doc.get(field_name).and_then(|v| v.as_str());
                    value
                })
                .collect(),
            _ => unreachable!(),
        };

        // The input could be:
        // - empty: we should skip this (???)
        // - "normal": it is ok
        // - "too long": we should chunk it in a smart way
        // TODO: implement that logic

        self.embedding_sender
            .send(EmbeddingCalculationRequest {
                model_name: self.model_name.clone(),
                input: EmbeddingCalculationRequestInput {
                    text: input,
                    coll_id,
                    doc_id,
                    field_id,
                    op_sender: sender,
                },
            })
            .await?;

        Ok(())
    }

    fn serialized(&self) -> SerializedFieldIndexer {
        SerializedFieldIndexer::Embedding(self.model_name.clone(), self.document_fields.clone())
    }
}
