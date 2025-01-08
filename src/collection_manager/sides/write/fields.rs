use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    sync::Arc,
};

use anyhow::Result;
use dashmap::DashMap;

use crate::{
    collection_manager::dto::FieldId,
    embeddings::LoadedModel,
    indexes::number::Number,
    metrics::{
        EmbeddingCalculationLabels, StringCalculationLabels, EMBEDDING_CALCULATION_METRIC,
        STRING_CALCULATION_METRIC,
    },
    nlp::TextParser,
    types::{CollectionId, DocumentId, FlattenDocument, ValueType},
};

use super::{
    CollectionWriteOperation, DocumentFieldIndexOperation, Term, TermStringField, WriteOperation,
};

pub type FieldsToIndex = DashMap<String, (ValueType, Arc<Box<dyn FieldIndexer>>)>;

pub trait FieldIndexer: Sync + Send + Debug {
    fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: &str,
        field_id: FieldId,
        doc: &FlattenDocument,
    ) -> Result<Vec<WriteOperation>>;
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

impl FieldIndexer for NumberField {
    fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: &str,
        field_id: FieldId,
        doc: &FlattenDocument,
    ) -> Result<Vec<WriteOperation>> {
        let value = doc.get(field_name).and_then(|v| Number::try_from(v).ok());

        let value = match value {
            None => return Ok(vec![]),
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

        Ok(vec![op])
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

impl FieldIndexer for BoolField {
    fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: &str,
        field_id: FieldId,
        doc: &FlattenDocument,
    ) -> Result<Vec<WriteOperation>> {
        let value = doc.get(field_name);

        let value = match value {
            None => return Ok(vec![]),
            Some(value) => match value.as_bool() {
                None => return Ok(vec![]),
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

        Ok(vec![op])
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

impl FieldIndexer for StringField {
    fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        field_name: &str,
        field_id: FieldId,
        doc: &FlattenDocument,
    ) -> Result<Vec<WriteOperation>> {
        let metric = STRING_CALCULATION_METRIC.create(StringCalculationLabels {
            collection: coll_id.0.clone(),
            field: field_name.to_string(),
        });

        let value = doc.get(field_name);

        let data = match value {
            None => return Ok(vec![]),
            Some(value) => match value.as_str() {
                None => return Ok(vec![]),
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

        Ok(vec![op])
    }
}

#[derive(Debug)]
pub struct EmbeddingField {
    model: Arc<LoadedModel>,
    document_fields: Vec<String>,
}

impl EmbeddingField {
    pub fn new(model: Arc<LoadedModel>, document_fields: Vec<String>) -> Self {
        Self {
            model,
            document_fields,
        }
    }
}

impl FieldIndexer for EmbeddingField {
    fn get_write_operations(
        &self,
        coll_id: CollectionId,
        doc_id: DocumentId,
        _field_name: &str,
        field_id: FieldId,
        doc: &FlattenDocument,
    ) -> Result<Vec<WriteOperation>> {
        let input: String = self
            .document_fields
            .iter()
            .filter_map(|field_name| {
                let value = doc.get(field_name).and_then(|v| v.as_str());
                value
            })
            .collect();

        let metric = EMBEDDING_CALCULATION_METRIC.create(EmbeddingCalculationLabels {
            collection: coll_id.0.clone(),
            model: self.model.model_name().to_string(),
        });
        // The input could be:
        // - empty: we should skip this (???)
        // - "normal": it is ok
        // - "too long": we should chunk it in a smart way
        // TODO: implement that logic
        let mut output = self.model.embed(vec![input], None)?;
        let output = output.remove(0);
        drop(metric);

        Ok(vec![WriteOperation::Collection(
            coll_id.clone(),
            CollectionWriteOperation::Index(
                doc_id,
                field_id,
                DocumentFieldIndexOperation::IndexEmbedding { value: output },
            ),
        )])
    }
}
