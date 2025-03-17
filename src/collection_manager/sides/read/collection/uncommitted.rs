use std::collections::{HashMap, HashSet};

use anyhow::Result;
use bool::{BoolField, BoolUncommittedFieldStats};
use number::{NumberField, NumberUncommittedFieldStats};
use string::{StringField, StringUncommittedFieldStats};
use tracing::trace;
use vector::{VectorField, VectorUncommittedFieldStats};

use crate::{
    collection_manager::{
        dto::{BM25Scorer, FieldId, GlobalInfo, NumberFilter},
        sides::DocumentFieldIndexOperation,
    },
    types::DocumentId,
};

mod bool;
mod number;
mod string;
mod vector;

pub mod fields {
    pub use super::bool::{BoolField, BoolUncommittedFieldStats};
    pub use super::number::{NumberField, NumberUncommittedFieldStats};
    pub use super::string::{StringField, StringUncommittedFieldStats};
    pub use super::vector::{VectorField, VectorUncommittedFieldStats};
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

    pub fn get_infos(&self) -> UncommittedInfo {
        trace!(
            "Getting uncommitted info. vector: {:?}, number {:?}, string {:?}, bool {:?}",
            self.vector_index
                .iter()
                .map(|(k, v)| (k, v.len()))
                .collect::<Vec<_>>(),
            self.number_index
                .iter()
                .map(|(k, v)| (k, v.len()))
                .collect::<Vec<_>>(),
            self.string_index
                .iter()
                .map(|(k, v)| (k, v.len()))
                .collect::<Vec<_>>(),
            self.bool_index
                .iter()
                .map(|(k, v)| (k, v.len()))
                .collect::<Vec<_>>(),
        );

        UncommittedInfo {
            number_fields: self
                .number_index
                .iter()
                .filter(|(_, v)| v.len() > 0)
                .map(|(k, _)| *k)
                .collect(),
            string_fields: self
                .string_index
                .iter()
                .filter(|(_, v)| v.len() > 0)
                .map(|(k, _)| *k)
                .collect(),
            bool_fields: self
                .bool_index
                .iter()
                .filter(|(_, v)| v.len() > 0)
                .map(|(k, _)| *k)
                .collect(),
            vector_fields: self
                .vector_index
                .iter()
                .filter(|(_, v)| v.len() > 0)
                .map(|(k, _)| *k)
                .collect(),
        }
    }

    pub fn get_bool_stats(&self) -> HashMap<FieldId, BoolUncommittedFieldStats> {
        let mut ret = HashMap::new();
        for (field_id, field) in &self.bool_index {
            let field_stats = field.get_stats();
            ret.insert(*field_id, field_stats);
        }
        ret
    }

    pub fn get_number_stats(&self) -> HashMap<FieldId, NumberUncommittedFieldStats> {
        let mut ret = HashMap::new();
        for (field_id, field) in &self.number_index {
            let field_stats = field.get_stats();
            ret.insert(*field_id, field_stats);
        }
        ret
    }

    pub fn get_string_stats(&self) -> HashMap<FieldId, StringUncommittedFieldStats> {
        let mut ret = HashMap::new();
        for (field_id, field) in &self.string_index {
            let field_stats = field.get_stats();
            ret.insert(*field_id, field_stats);
        }
        ret
    }

    pub fn get_vector_stats(&self) -> HashMap<FieldId, VectorUncommittedFieldStats> {
        let mut ret = HashMap::new();
        for (field_id, field) in &self.vector_index {
            let field_stats = field.get_stats();
            ret.insert(*field_id, field_stats);
        }
        ret
    }

    pub fn vector_search(
        &self,
        target: &[f32],
        similarity: f32,
        properties: &[FieldId],
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        for vector_field in properties {
            let vector_field = match self.vector_index.get(vector_field) {
                Some(vector_field) => vector_field,
                None => {
                    trace!("Vector field not found");
                    continue;
                }
            };
            vector_field.search(
                target,
                similarity,
                filtered_doc_ids,
                output,
                uncommitted_deleted_documents,
            )?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fulltext_search(
        &self,
        tokens: &[String],
        properties: Vec<FieldId>,
        boost: &HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        scorer: &mut BM25Scorer<DocumentId>,
        global_info: &GlobalInfo,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
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

            index.search(
                tokens,
                field_boost,
                scorer,
                filtered_doc_ids,
                global_info,
                uncommitted_deleted_documents,
            )?;
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
                    .insert(doc_id, value.0);
            }
            DocumentFieldIndexOperation::IndexStringFilter { value } => {
                panic!("Not yet implemented");
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
}

#[derive(Debug)]
pub struct UncommittedInfo {
    pub number_fields: HashSet<FieldId>,
    pub string_fields: HashSet<FieldId>,
    pub bool_fields: HashSet<FieldId>,
    pub vector_fields: HashSet<FieldId>,
}

impl UncommittedInfo {
    pub fn is_empty(&self) -> bool {
        self.number_fields.is_empty()
            && self.string_fields.is_empty()
            && self.bool_fields.is_empty()
            && self.vector_fields.is_empty()
    }
}
