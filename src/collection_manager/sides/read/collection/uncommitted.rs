use std::collections::{HashMap, HashSet};

use anyhow::Result;
use bool::BoolField;
use number::NumberField;
use string::StringField;
use tracing::trace;
use vector::VectorField;

use crate::{
    collection_manager::{dto::FieldId, sides::DocumentFieldIndexOperation},
    indexes::{
        number::NumberFilter,
        string::{BM25Scorer, GlobalInfo},
    },
    types::DocumentId,
};

pub mod bool;
pub mod number;
pub mod string;
pub mod vector;

pub mod fields {
    pub use super::bool::BoolField;
    pub use super::number::NumberField;
    pub use super::string::StringField;
    pub use super::vector::VectorField;
}

pub use string::{Positions, TotalDocumentsWithTermInField};

#[derive(Debug)]
pub struct UncommittedCollection {
    pub number_index: HashMap<FieldId, NumberField>,
    pub bool_index: HashMap<FieldId, BoolField>,
    pub string_index: HashMap<FieldId, StringField>,
    pub vector_index: HashMap<FieldId, VectorField>,
}

impl UncommittedCollection {
    pub fn new() -> Self {
        Self {
            number_index: HashMap::new(),
            bool_index: HashMap::new(),
            string_index: HashMap::new(),
            vector_index: HashMap::new(),
        }
    }

    pub fn global_info(&self, field_id: &FieldId) -> GlobalInfo {
        self.string_index
            .get(field_id)
            .map(StringField::global_info)
            .unwrap_or_default()
    }

    pub fn vector_search(
        &self,
        target: &[f32],
        properties: &[FieldId],
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        for vector_field in properties {
            let vector_field = match self.vector_index.get(vector_field) {
                Some(vector_field) => vector_field,
                None => {
                    trace!("Vector field not found");
                    continue;
                }
            };
            vector_field.search(target, filtered_doc_ids, output)?;
        }

        Ok(())
    }

    pub fn fulltext_search(
        &self,
        tokens: &Vec<String>,
        properties: Vec<FieldId>,
        boost: &HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        scorer: &mut BM25Scorer<DocumentId>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        for field_id in properties {
            let index = match self.string_index.get(&field_id) {
                Some(index) => index,
                // If the field is not indexed, we skip it
                // This could be:
                // - a field id is not a string field (this should not happen)
                // - is not yet committed
                None => continue,
            };

            let field_boost = boost.get(&field_id).copied().unwrap_or(1.0);

            index.search(tokens, field_boost, scorer, filtered_doc_ids, global_info)?;
        }

        Ok(())
    }

    pub fn calculate_number_filter<'s, 'iter>(
        &'s self,
        field_id: FieldId,
        filter_number: &NumberFilter,
    ) -> Result<Option<impl Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let number_index = match self.number_index.get(&field_id) {
            Some(index) => index,
            None => return Ok(None),
        };
        Ok(Some(number_index.filter(filter_number)))
    }

    pub fn calculate_bool_filter<'s, 'iter>(
        &'s self,
        field_id: FieldId,
        value: bool,
    ) -> Result<Option<impl Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let bool_index = match self.bool_index.get(&field_id) {
            Some(index) => index,
            None => return Ok(None),
        };

        Ok(Some(bool_index.filter(value)))
    }

    pub fn insert(
        &mut self,
        field_id: FieldId,
        doc_id: DocumentId,
        op: DocumentFieldIndexOperation,
    ) -> Result<()> {
        match op {
            DocumentFieldIndexOperation::IndexBoolean { value } => {
                self.bool_index
                    .entry(field_id)
                    .or_insert_with(BoolField::empty)
                    .insert(doc_id, value);
            }
            DocumentFieldIndexOperation::IndexNumber { value } => {
                self.number_index
                    .entry(field_id)
                    .or_insert_with(NumberField::empty)
                    .insert(doc_id, value);
            }
            DocumentFieldIndexOperation::IndexString {
                field_length,
                terms,
            } => {
                self.string_index
                    .entry(field_id)
                    .or_insert_with(StringField::empty)
                    .insert(doc_id, field_length, terms);
            }
            DocumentFieldIndexOperation::IndexEmbedding { value } => {
                self.vector_index
                    .entry(field_id)
                    .or_insert_with(|| VectorField::empty(value.len()))
                    .insert(doc_id, vec![value])?;
            }
        };

        Ok(())
    }

    pub fn get_number_fields(&self) -> impl Iterator<Item = &FieldId> {
        self.number_index.keys()
    }

    pub fn get_string_fields(&self) -> impl Iterator<Item = &FieldId> {
        self.string_index.keys()
    }

    pub fn get_bool_fields(&self) -> impl Iterator<Item = &FieldId> {
        self.bool_index.keys()
    }

    pub fn get_vector_fields(&self) -> impl Iterator<Item = &FieldId> {
        self.vector_index.keys()
    }
}
