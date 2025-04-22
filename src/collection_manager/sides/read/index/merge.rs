use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::file_utils::create_if_not_exists;
use crate::merger::MergedIterator;
use crate::types::{DocumentId, SerializableNumber};

use super::committed::fields as committed_fields;
use super::uncommitted::fields as uncommitted_fields;

pub fn merge_number_field(
    uncommitted: Option<&uncommitted_fields::UncommittedNumberField>,
    committed: Option<&committed_fields::CommittedNumberField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<committed_fields::CommittedNumberField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed number fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            if uncommitted_document_deletions.is_empty() {
                // No changes
                return Ok(None);
            }

            let committed_iter = committed.iter().map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            });

            Ok(Some(committed_fields::CommittedNumberField::from_iter(
                committed_iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), None) => {
            let iter = uncommitted
                .iter()
                .map(|(n, v)| (SerializableNumber(n), v))
                .map(|(k, mut d)| {
                    d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                    (k, d)
                });
            Ok(Some(committed_fields::CommittedNumberField::from_iter(
                iter, data_dir,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.len() == 0 {
                return Ok(None);
            }

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
            Ok(Some(committed_fields::CommittedNumberField::from_iter(
                iter, data_dir,
            )?))
        }
    }
}

pub fn merge_string_filter_field(
    uncommitted: Option<&uncommitted_fields::UncommittedStringFilterField>,
    committed: Option<&committed_fields::CommittedStringFilterField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<committed_fields::CommittedStringFilterField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed number fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            if uncommitted_document_deletions.is_empty() {
                // No changes
                return Ok(None);
            }

            let committed_iter = committed.iter().map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            });

            Ok(Some(committed_fields::CommittedStringFilterField::from_iter(
                committed_iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), None) => {
            let iter = uncommitted.iter().map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            });
            Ok(Some(committed_fields::CommittedStringFilterField::from_iter(
                iter, data_dir,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.len() == 0 {
                return Ok(None);
            }

            let uncommitted_iter = uncommitted.iter();
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
            Ok(Some(committed_fields::CommittedStringFilterField::from_iter(
                iter, data_dir,
            )?))
        }
    }
}

pub fn merge_bool_field(
    uncommitted: Option<&uncommitted_fields::UncommittedBoolField>,
    committed: Option<&committed_fields::CommittedBoolField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<committed_fields::CommittedBoolField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed bool fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            if uncommitted_document_deletions.is_empty() {
                // No changes
                return Ok(None);
            }

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

            Ok(Some(committed_fields::CommittedBoolField::from_iter(
                iter, data_dir,
            )?))
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

            Ok(Some(committed_fields::CommittedBoolField::from_iter(
                iter, data_dir,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.len() == 0 {
                return Ok(None);
            }

            let (uncommitted_true_docs, uncommitted_false_docs) = uncommitted.clone_inner();
            let (mut committed_true_docs, mut committed_false_docs) = committed.clone_inner()?;

            committed_true_docs.extend(uncommitted_true_docs);
            committed_false_docs.extend(uncommitted_false_docs);

            Ok(Some(committed_fields::CommittedBoolField::from_data(
                committed_true_docs,
                committed_false_docs,
                data_dir,
            )?))
        }
    }
}

pub fn merge_string_field(
    uncommitted: Option<&uncommitted_fields::UncommittedStringField>,
    committed: Option<&committed_fields::CommittedStringField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<committed_fields::CommittedStringField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed string fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            if uncommitted_document_deletions.is_empty() {
                // No changes
                return Ok(None);
            }

            Ok(Some(committed_fields::CommittedStringField::from_committed(
                committed,
                data_dir,
                uncommitted_document_deletions,
            )?))
        }
        (Some(uncommitted), None) => {
            let length_per_documents = uncommitted.field_length_per_doc();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));
            let mut entries: Vec<_> = iter.collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));

            Ok(Some(committed_fields::CommittedStringField::from_iter(
                entries.into_iter(),
                length_per_documents,
                data_dir,
                uncommitted_document_deletions,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.len() == 0 {
                return Ok(None);
            }

            let length_per_documents = uncommitted.field_length_per_doc();
            // let uncommitted_iter = uncommitted.iter();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));
            let mut entries: Vec<_> = iter.collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));

            Ok(Some(
                committed_fields::CommittedStringField::from_iter_and_committed(
                    entries.into_iter(),
                    committed,
                    length_per_documents,
                    data_dir,
                    uncommitted_document_deletions,
                )
                .context("Failed to merge string field")?,
            ))
        }
    }
}

pub fn merge_vector_field(
    uncommitted: Option<&uncommitted_fields::UncommittedEmbeddingField>,
    committed: Option<&committed_fields::CommittedEmbeddingField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<committed_fields::CommittedEmbeddingField>> {
    create_if_not_exists(&data_dir).context("Failed to create data directory for vector field")?;

    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed vector fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            if uncommitted_document_deletions.is_empty() {
                // No changes
                Ok(None)
            } else {
                let info = committed.get_field_info();
                let new_field = committed_fields::CommittedEmbeddingField::from_dump_and_iter(
                    info.data_dir,
                    std::iter::empty(),
                    uncommitted_document_deletions,
                    data_dir,
                )?;
                Ok(Some(new_field))
            }
        }
        (Some(uncommitted), None) => {
            let new_field = committed_fields::CommittedEmbeddingField::from_iter(
                uncommitted
                    .iter()
                    .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id)),
                uncommitted.dimension(),
                data_dir,
            )?;
            Ok(Some(new_field))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.len() == 0 {
                return Ok(None);
            }

            let info = committed.get_field_info();
            let new_field = committed_fields::CommittedEmbeddingField::from_dump_and_iter(
                info.data_dir,
                uncommitted.iter(),
                uncommitted_document_deletions,
                data_dir,
            )?;
            Ok(Some(new_field))
        }
    }
}
