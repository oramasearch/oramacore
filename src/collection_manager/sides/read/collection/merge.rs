use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::indexes::number::SerializableNumber;
use crate::merger::MergedIterator;

use super::committed::fields as committed_fields;
use super::uncommitted::fields as uncommitted_fields;

pub fn merge_number_field(
    uncommitted: &uncommitted_fields::NumberField,
    committed: Option<&committed_fields::NumberField>,
    data_dir: PathBuf,
) -> Result<committed_fields::NumberField> {
    let committed = match committed {
        None => {
            let iter = uncommitted.iter().map(|(n, v)| (SerializableNumber(n), v));
            committed_fields::NumberField::from_iter(iter, data_dir)?
        }
        Some(committed) => {
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
            );
            committed_fields::NumberField::from_iter(iter, data_dir)?
        }
    };

    Ok(committed)
}

pub fn merge_bool_field(
    uncommitted: &uncommitted_fields::BoolField,
    committed: Option<&committed_fields::BoolField>,
    data_dir: PathBuf,
) -> Result<committed_fields::BoolField> {
    let new_committed_field = match committed {
        None => {
            let (true_docs, false_docs) = uncommitted.clone_inner();
            let iter = vec![
                (committed_fields::BoolWrapper::False, false_docs),
                (committed_fields::BoolWrapper::True, true_docs),
            ]
            .into_iter();

            committed_fields::BoolField::from_iter(iter, data_dir)?
        }
        Some(committed) => {
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
    uncommitted: &uncommitted_fields::StringField,
    committed: Option<&committed_fields::StringField>,
    data_dir: PathBuf,
) -> Result<committed_fields::StringField> {
    let new_committed_field = match committed {
        None => {
            let length_per_documents = uncommitted.field_length_per_doc();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));
            let mut entries: Vec<_> = iter.collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));

            committed_fields::StringField::from_iter(
                entries.into_iter(),
                length_per_documents,
                data_dir,
            )?
        }
        Some(committed) => {
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
            )
            .context("Failed to merge string field")?
        }
    };

    Ok(new_committed_field)
}

pub fn merge_vector_field(
    uncommitted: &uncommitted_fields::VectorField,
    committed: Option<&committed_fields::VectorField>,
    data_dir: PathBuf,
) -> Result<committed_fields::VectorField> {
    let file_path = data_dir.join("data.hnsw");
    let new_committed_field = match committed {
        None => committed_fields::VectorField::from_iter(
            uncommitted.iter(),
            uncommitted.dimension(),
            file_path,
        )?,
        Some(committed) => {
            std::fs::copy(committed.file_path(), &file_path).context("Failed to copy hnsw file")?;

            let mut vector_index = committed_fields::VectorField::load(file_path)?;
            vector_index
                .add_and_dump(uncommitted.iter())
                .context("Cannot add new element to vector index")?;

            vector_index
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
