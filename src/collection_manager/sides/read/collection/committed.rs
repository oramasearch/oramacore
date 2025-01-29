use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use bool::{BoolField, BoolWrapper};
use number::NumberField;
use string::StringField;
use vector::VectorField;

use crate::{
    collection_manager::dto::FieldId,
    indexes::{
        number::{NumberFilter, SerializableNumber},
        string::{BM25Scorer, GlobalInfo},
    },
    merger::MergedIterator,
    types::DocumentId,
};

use super::uncommitted::UncommittedCollection;

mod bool;
mod number;
mod string;
mod vector;

pub mod fields {
    pub use super::bool::BoolField;
    pub use super::number::NumberField;
    pub use super::string::StringField;
    pub use super::vector::VectorField;

    pub use super::bool::BoolWrapper;
}

#[derive(Debug)]
pub struct CommittedCollection {
    pub number_index: HashMap<FieldId, NumberField>,
    pub bool_index: HashMap<FieldId, BoolField>,
    pub string_index: HashMap<FieldId, StringField>,
    pub vector_index: HashMap<FieldId, VectorField>,
}

impl CommittedCollection {
    pub fn new() -> Self {
        Self {
            number_index: HashMap::new(),
            bool_index: HashMap::new(),
            string_index: HashMap::new(),
            vector_index: HashMap::new(),
        }
    }

    pub fn load(&mut self, data_dir: PathBuf) -> Result<()> {
        self.number_index = NumberField::load_all(&data_dir)?;
        self.bool_index = BoolField::load_all(&data_dir)?;
        self.string_index = StringField::load_all(&data_dir)?;
        self.vector_index = VectorField::load_all(&data_dir)?;

        Ok(())
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
        limit: usize,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        for vector_field in properties {
            let vector_field = self
                .vector_index
                .get(vector_field)
                .context("Field is not a vector field")?;
            vector_field.search(target, limit, filtered_doc_ids, output)?;
        }

        Ok(())
    }

    pub fn fulltext_search<'s, 'boost, 'scorer>(
        &'s self,
        tokens: &Vec<String>,
        properties: Vec<FieldId>,
        boost: &'boost HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        scorer: &'scorer mut BM25Scorer<DocumentId>,
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

            index.search(&tokens, field_boost, scorer, filtered_doc_ids, global_info)?;
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
}
