use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use bool::BoolField;
use number::NumberField;
use string::StringField;
use vector::VectorField;

use crate::{
    collection_manager::{dto::FieldId, sides::DocumentFieldIndexOperation},
    indexes::{
        number::NumberFilter,
        string::{BM25Scorer, GlobalInfo},
    },
    types::DocumentId,
};

mod bool;
mod number;
mod string;
mod vector;

pub use string::{Positions, TotalDocumentsWithTermInField};

#[derive(Debug)]
pub struct UncommittedCollection {
    pub number_index: HashMap<FieldId, NumberField>,
    pub bool_index: HashMap<FieldId, BoolField>,
    pub string_index: HashMap<FieldId, StringField>,
    pub vector_index: HashMap<FieldId, VectorField>,
}

impl UncommittedCollection {
    pub fn vector_search(
        &self,
        target: &[f32],
        properties: Vec<FieldId>,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        for vector_field in properties {
            let vector_field = self
                .vector_index
                .get(&vector_field)
                .context("Field is not a vector field")?;
            vector_field.search(target, output)?;
        }

        Ok(())
    }

    pub fn fulltext_search<'s, 'boost, 'scorer>(
        &'s self,
        tokens: &Vec<String>,
        properties: Vec<FieldId>,
        boost: &'boost HashMap<FieldId, f32>,
        filtered_doc_ids: Option<HashSet<DocumentId>>,
        scorer: &'scorer mut BM25Scorer<DocumentId>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        for field_id in properties {
            let index = self
                .string_index
                .get(&field_id)
                .expect("Field is not a string field");

            let field_boost = boost.get(&field_id).copied().unwrap_or(1.0);

            index.search(
                &tokens,
                field_boost,
                scorer,
                filtered_doc_ids.as_ref(),
                global_info,
            )?;
        }

        Ok(())
    }

    pub fn calculate_number_filter<'s, 'iter>(
        &'s self,
        field_id: FieldId,
        filter_number: NumberFilter,
    ) -> Result<impl Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        let number_index = self
            .number_index
            .get(&field_id)
            .context("Field is not a number field")?;
        Ok(number_index.filter(filter_number))
    }

    pub fn calculate_bool_filter<'s, 'iter>(
        &'s self,
        field_id: FieldId,
        value: bool,
    ) -> Result<impl Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        let bool_index = self
            .bool_index
            .get(&field_id)
            .context("Field is not a bool field")?;
        Ok(bool_index.filter(value))
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
                    .or_insert_with(VectorField::empty)
                    .insert(doc_id, vec![value]);
            }
        };

        Ok(())
    }
}
