use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::collection_manager::dto::SerializableNumber;
use crate::file_utils::create_if_not_exists;
use crate::merger::MergedIterator;
use crate::types::DocumentId;

use super::committed::fields as committed_fields;
use super::uncommitted::fields as uncommitted_fields;

pub fn merge_number_field(
    uncommitted: Option<&uncommitted_fields::NumberField>,
    committed: Option<&committed_fields::NumberField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<committed_fields::NumberField> {
    let committed = match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed number fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            let committed_iter = committed.iter().map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            });

            committed_fields::NumberField::from_iter(committed_iter, data_dir)?
        }
        (Some(uncommitted), None) => {
            let iter = uncommitted
                .iter()
                .map(|(n, v)| (SerializableNumber(n), v))
                .map(|(k, mut d)| {
                    d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                    (k, d)
                });
            committed_fields::NumberField::from_iter(iter, data_dir)?
        }
        (Some(uncommitted), Some(committed)) => {
            let uncommitted_iter = uncommitted.iter().map(|(n, v)| (SerializableNumber(n), v));
            let committed_iter = committed.iter();

            let iter = MergedIterator::new(
                committed_iter,
                uncommitted_iter,
                |_, v| v,
                |_, mut v1, v2| {
                    v1.extend(v2);
                    v1
                },
            )
            .map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            });
            committed_fields::NumberField::from_iter(iter, data_dir)?
        }
    };

    Ok(committed)
}

pub fn merge_bool_field(
    uncommitted: Option<&uncommitted_fields::BoolField>,
    committed: Option<&committed_fields::BoolField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<committed_fields::BoolField> {
    let new_committed_field = match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed bool fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            let (true_docs, false_docs) = committed.clone_inner()?;
            let iter = vec![
                (committed_fields::BoolWrapper::False, false_docs),
                (committed_fields::BoolWrapper::True, true_docs),
            ]
            .into_iter()
            .map(|(k, mut v)| {
                v.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, v)
            });

            committed_fields::BoolField::from_iter(iter, data_dir)?
        }
        (Some(uncommitted), None) => {
            let (true_docs, false_docs) = uncommitted.clone_inner();
            let iter = vec![
                (committed_fields::BoolWrapper::False, false_docs),
                (committed_fields::BoolWrapper::True, true_docs),
            ]
            .into_iter()
            .map(|(k, mut v)| {
                v.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, v)
            });

            committed_fields::BoolField::from_iter(iter, data_dir)?
        }
        (Some(uncommitted), Some(committed)) => {
            let (uncommitted_true_docs, uncommitted_false_docs) = uncommitted.clone_inner();
            let (mut committed_true_docs, mut committed_false_docs) = committed.clone_inner()?;

            committed_true_docs.extend(uncommitted_true_docs);
            committed_false_docs.extend(uncommitted_false_docs);

            committed_fields::BoolField::from_data(
                committed_true_docs,
                committed_false_docs,
                data_dir,
            )?
        }
    };

    Ok(new_committed_field)
}

pub fn merge_string_field(
    uncommitted: Option<&uncommitted_fields::StringField>,
    committed: Option<&committed_fields::StringField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<committed_fields::StringField> {
    let new_committed_field = match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed string fields are None. Never should happen");
        }
        (None, Some(committed)) => committed_fields::StringField::from_committed(
            committed,
            data_dir,
            uncommitted_document_deletions,
        )?,
        (Some(uncommitted), None) => {
            let length_per_documents = uncommitted.field_length_per_doc();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));
            let mut entries: Vec<_> = iter.collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));

            committed_fields::StringField::from_iter(
                entries.into_iter(),
                length_per_documents,
                data_dir,
                uncommitted_document_deletions,
            )?
        }
        (Some(uncommitted), Some(committed)) => {
            let length_per_documents = uncommitted.field_length_per_doc();
            // let uncommitted_iter = uncommitted.iter();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));
            let mut entries: Vec<_> = iter.collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));

            committed_fields::StringField::from_iter_and_committed(
                entries.into_iter(),
                committed,
                length_per_documents,
                data_dir,
                uncommitted_document_deletions,
            )
            .context("Failed to merge string field")?
        }
    };

    Ok(new_committed_field)
}

pub fn merge_vector_field(
    uncommitted: Option<&uncommitted_fields::VectorField>,
    committed: Option<&committed_fields::VectorField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<committed_fields::VectorField> {
    create_if_not_exists(&data_dir).context("Failed to create data directory for vector field")?;

    let new_committed_field = match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed vector fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            committed
                .clone_to(&data_dir)
                .context("Failed to copy hnsw file")?;
            committed_fields::VectorField::from_dump_and_iter(
                data_dir,
                std::iter::empty(),
                uncommitted_document_deletions,
            )?
        }
        (Some(uncommitted), None) => committed_fields::VectorField::from_iter(
            uncommitted
                .iter()
                .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id)),
            uncommitted.dimension(),
            data_dir,
        )?,
        (Some(uncommitted), Some(committed)) => {
            committed
                .clone_to(&data_dir)
                .context("Failed to copy hnsw file")?;
            committed_fields::VectorField::from_dump_and_iter(
                data_dir,
                uncommitted.iter(),
                uncommitted_document_deletions,
            )?
        }
    };

    Ok(new_committed_field)
}

/*
pub fn merge_uncommitted(
    committed: &mut CommittedCollection,
    uncommitted: &mut UncommittedCollection,
    data_dir: PathBuf,
) -> Result<()> {
    let data_dir = data_dir.join("fields");

    let number_dir = data_dir.join("numbers");
    for (field_id, field) in &uncommitted.number_index {
        let committed = committed.number_index.get(field_id);

        let field_dir = number_dir.join(format!("field-{}", field_id.0));

        let new_committed_field = merge_number_field(field, committed, field_dir)?;

        committed.number_index.insert(*field_id, new_committed_field);
    }

    let bool_dir = data_dir.join("bools");
    for (field_id, field) in &uncommitted.bool_index {
        let committed = committed.bool_index.get(field_id);

        let field_dir = bool_dir.join(format!("field-{}", field_id.0));

        let new_committed_field = merge_bool_field(uncommitted, committed, field_dir)?;

        committed.bool_index.insert(*field_id, new_committed_field);
    }

    let strings_dir = data_dir.join("strings");
    for (field_id, field) in &uncommitted.string_index {
        let committed = committed.string_index.get(field_id);

        let field_dir = strings_dir.join(format!("field-{}", field_id.0));

        let new_committed_field = merge_string_field(uncommitted, committed, field_dir)?;


        committed.string_index.insert(*field_id, new_committed_field);
    }

    Ok(())
}
*/
