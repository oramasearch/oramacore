use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
pub use bool::{BoolFieldInfo, CommittedBoolField, CommittedBoolFieldStats};
pub use number::{CommittedNumberField, CommittedNumberFieldStats, NumberFieldInfo};
pub use string::{CommittedStringField, CommittedStringFieldStats, StringFieldInfo};
pub use string_filter::{
    CommittedStringFilterField, CommittedStringFilterFieldStats, StringFilterFieldInfo,
};
pub use vector::{CommittedVectorField, CommittedVectorFieldStats, VectorFieldInfo};

use crate::{
    collection_manager::{bm25::BM25Scorer, global_info::GlobalInfo},
    types::{DocumentId, FieldId, NumberFilter},
};

mod bool;
mod number;
mod string;
mod string_filter;
mod vector;

#[derive(Debug)]
pub struct CommittedCollection {
    pub number_index: HashMap<FieldId, CommittedNumberField>,
    pub bool_index: HashMap<FieldId, CommittedBoolField>,
    pub string_filter_index: HashMap<FieldId, CommittedStringFilterField>,
    pub string_index: HashMap<FieldId, CommittedStringField>,
    pub vector_index: HashMap<FieldId, CommittedVectorField>,
}

impl CommittedCollection {
    pub fn empty() -> Self {
        Self {
            number_index: Default::default(),
            bool_index: Default::default(),
            string_filter_index: Default::default(),
            string_index: Default::default(),
            vector_index: Default::default(),
        }
    }

    pub fn try_load(
        number_field_infos: Vec<(FieldId, NumberFieldInfo)>,
        bool_field_infos: Vec<(FieldId, BoolFieldInfo)>,
        string_filter_field_infos: Vec<(FieldId, StringFilterFieldInfo)>,
        string_field_infos: Vec<(FieldId, StringFieldInfo)>,
        vector_field_infos: Vec<(FieldId, VectorFieldInfo)>,
    ) -> Result<Self> {
        let mut number_index: HashMap<FieldId, CommittedNumberField> = Default::default();
        let mut bool_index: HashMap<FieldId, CommittedBoolField> = Default::default();
        let mut string_filter_index: HashMap<FieldId, CommittedStringFilterField> =
            Default::default();
        let mut string_index: HashMap<FieldId, CommittedStringField> = Default::default();
        let mut vector_index: HashMap<FieldId, CommittedVectorField> = Default::default();

        for (field_id, info) in number_field_infos {
            let number_field = CommittedNumberField::load(info)
                .with_context(|| format!("Cannot load number {:?} field", field_id))?;
            number_index.insert(field_id, number_field);
        }
        for (field_id, info) in bool_field_infos {
            let bool_field = CommittedBoolField::load(info)
                .with_context(|| format!("Cannot load bool {:?} field", field_id))?;
            bool_index.insert(field_id, bool_field);
        }
        for (field_id, info) in string_field_infos {
            let string_field = CommittedStringField::load(info)
                .with_context(|| format!("Cannot load string {:?} field", field_id))?;
            string_index.insert(field_id, string_field);
        }
        for (field_id, info) in vector_field_infos {
            let vector_field = CommittedVectorField::load(info)
                .with_context(|| format!("Cannot load vector {:?} field", field_id))?;
            vector_index.insert(field_id, vector_field);
        }
        for (field_id, info) in string_filter_field_infos {
            let vector_field = CommittedStringFilterField::load(info)
                .with_context(|| format!("Cannot load string filter  {:?} field", field_id))?;
            string_filter_index.insert(field_id, vector_field);
        }

        Ok(Self {
            number_index,
            bool_index,
            string_filter_index,
            string_index,
            vector_index,
        })
    }

    pub fn global_info(&self, field_id: &FieldId) -> GlobalInfo {
        self.string_index
            .get(field_id)
            .map(CommittedStringField::global_info)
            .unwrap_or_default()
    }

    pub fn get_keys(&self) -> CommittedKeys {
        CommittedKeys {
            number_fields: self.number_index.keys().copied().collect(),
            string_fields: self.string_index.keys().copied().collect(),
            bool_fields: self.bool_index.keys().copied().collect(),
            vector_fields: self.vector_index.keys().copied().collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vector_search(
        &self,
        target: &[f32],
        similarity: f32,
        properties: &[FieldId],
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        limit: usize,
        output: &mut HashMap<DocumentId, f32>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        for field_id in properties {
            let vector_field = match self.vector_index.get(field_id) {
                Some(index) => index,
                // If the field is not indexed, we skip it
                // This could be:
                // - a field id is not a string field (this should not happen)
                // - is not yet committed
                None => continue,
            };
            vector_field.search(
                target,
                similarity,
                limit,
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
        let field = match self.number_index.get(&field_id) {
            Some(field) => field,
            None => return Ok(None),
        };

        field.filter(filter_number).map(Some)
    }

    pub fn calculate_string_filter<'s, 'iter>(
        &'s self,
        field_id: FieldId,
        filter_string: &str,
    ) -> Result<Option<impl Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let field = match self.string_filter_index.get(&field_id) {
            Some(field) => field,
            None => return Ok(None),
        };

        Ok(Some(field.filter(filter_string)))
    }

    pub fn calculate_bool_filter<'s, 'iter>(
        &'s self,
        field_id: FieldId,
        filter_bool: bool,
    ) -> Result<Option<impl Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let bool_index = match self.bool_index.get(&field_id) {
            Some(field) => field,
            None => return Ok(None),
        };
        bool_index.filter(filter_bool).map(Some)
    }

    pub fn get_string_values<'s, 'iter>(
        &'s self,
        field_id: FieldId,
    ) -> Result<Option<impl Iterator<Item = &'s str> + 'iter>>
    where
        's: 'iter,
    {
        let field = match self.string_filter_index.get(&field_id) {
            Some(field) => field,
            None => return Ok(None),
        };

        Ok(Some(field.get_string_value()))
    }
}

#[derive(Debug)]
pub struct CommittedKeys {
    pub number_fields: HashSet<FieldId>,
    pub string_fields: HashSet<FieldId>,
    pub bool_fields: HashSet<FieldId>,
    pub vector_fields: HashSet<FieldId>,
}
