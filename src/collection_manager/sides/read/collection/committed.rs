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
    collection_manager::dto::FieldId, indexes::{
        number::{NumberFilter, SerializableNumber},
        string::{BM25Scorer, GlobalInfo},
    }, merger::MergedIterator, types::DocumentId
};

use super::uncommitted::UncommittedCollection;

mod bool;
mod number;
mod string;
mod vector;

#[derive(Debug)]
pub struct CommittedCollection {
    number_index: HashMap<FieldId, NumberField>,
    bool_index: HashMap<FieldId, BoolField>,
    string_index: HashMap<FieldId, StringField>,
    vector_index: HashMap<FieldId, VectorField>,
}

impl CommittedCollection {
    pub fn vector_search(
        &self,
        target: &[f32],
        properties: Vec<FieldId>,
        limit: usize,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        for vector_field in properties {
            let vector_field = self
                .vector_index
                .get(&vector_field)
                .context("Field is not a vector field")?;
            vector_field.search(target, limit, output)?;
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
        number_index.filter(filter_number)
    }

    pub fn calculate_bool_filter<'s, 'iter>(
        &'s self,
        field_id: FieldId,
        filter_bool: bool,
    ) -> Result<impl Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        let bool_index = self
            .bool_index
            .get(&field_id)
            .context("Field is not a bool field")?;
        bool_index.filter(filter_bool)
    }

    pub fn merge_uncommitted(
        &mut self,
        uncommitted: &mut UncommittedCollection,
        data_dir: PathBuf,
    ) -> Result<()> {
        let data_dir = data_dir.join("fields");

        let number_dir = data_dir.join("numbers");
        for (field_id, field) in &uncommitted.number_index {
            let committed = self.number_index.get(field_id);

            let field_dir = number_dir.join(format!("field-{}", field_id.0));

            let new_committed_field = match committed {
                None => {
                    let iter = field.iter().map(|(n, v)| (SerializableNumber(n), v));
                    NumberField::from_iter(iter, field_dir)?
                }
                Some(committed) => {
                    let uncommitted_iter = field.iter().map(|(n, v)| (SerializableNumber(n), v));
                    let committed_iter = committed.iter();

                    let iter = MergedIterator::new(
                        committed_iter,
                        uncommitted_iter,
                        |_, v| v,
                        |_, mut v1, v2| {
                            v1.extend(v2);
                            v1
                        },
                    );
                    NumberField::from_iter(iter, field_dir)?
                }
            };

            self.number_index.insert(*field_id, new_committed_field);
        }

        let bool_dir = data_dir.join("bools");
        for (field_id, field) in &uncommitted.bool_index {
            let committed = self.bool_index.get(field_id);

            let field_dir = bool_dir.join(format!("field-{}", field_id.0));

            let new_committed_field = match committed {
                None => {
                    let (true_docs, false_docs) = field.clone_inner();
                    let iter = vec![
                        (BoolWrapper::False, false_docs),
                        (BoolWrapper::True, true_docs),
                    ].into_iter();

                    BoolField::from_iter(iter, field_dir)?
                },
                Some(committed) => {
                    let (uncommitted_true_docs, uncommitted_false_docs) = field.clone_inner();
                    let (mut committed_true_docs, mut committed_false_docs) = committed.clone_inner()?;

                    committed_true_docs.extend(uncommitted_true_docs);
                    committed_false_docs.extend(uncommitted_false_docs);
                    
                    BoolField::from_data(
                        committed_true_docs,
                        committed_false_docs,
                        field_dir,
                    )?
                }
            };

            self.bool_index.insert(*field_id, new_committed_field);
        }

        let strings_dir = data_dir.join("strings");
        for (field_id, field) in &uncommitted.string_index {
            let committed = self.string_index.get(field_id);

            let field_dir = strings_dir.join(format!("field-{}", field_id.0));

            let new_committed_field = match committed {
                None => {
                    let field_length_per_doc = field.field_length_per_doc();
                    let iter = field.iter().map(|(n, v)| (n, v.clone()));
                    StringField::from_iter(iter, field_length_per_doc, field_dir)?
                }
                Some(committed) => {
                    let uncommitted_iter = field.iter().map(|(n, v)| (n, v.clone()));
                    let committed_iter = committed.iter();

                    let iter = MergedIterator::new(
                        committed_iter,
                        uncommitted_iter,
                        |_, v| v.clone(),
                        |_, mut v1, v2| {
                            v1.extend(v2);
                            v1
                        },
                    );
                    StringField::from_iter(iter, field_dir)?
                }
            };

            self.string_index.insert(*field_id, new_committed_field);
        }

        Ok(())
    }
}
