use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::file_utils::create_if_not_exists;
use crate::merger::MergedIterator;
use crate::types::{DocumentId, SerializableNumber};

use super::committed_field::*;
use super::uncommitted_field::*;

pub fn merge_number_field(
    uncommitted: Option<&UncommittedNumberField>,
    committed: Option<&CommittedNumberField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<CommittedNumberField>> {
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

            Ok(Some(CommittedNumberField::from_iter(
                committed.field_path().to_vec().into_boxed_slice(),
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
            Ok(Some(CommittedNumberField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
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

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path(),
                "Uncommitted and committed field paths should be the same",
            );

            Ok(Some(CommittedNumberField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
    }
}

pub fn merge_string_filter_field(
    uncommitted: Option<&UncommittedStringFilterField>,
    committed: Option<&CommittedStringFilterField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<CommittedStringFilterField>> {
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

            Ok(Some(CommittedStringFilterField::from_iter(
                committed.field_path().to_vec().into_boxed_slice(),
                committed_iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), None) => {
            let iter = uncommitted.iter().map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            });
            Ok(Some(CommittedStringFilterField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
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

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path(),
                "Uncommitted and committed field paths should be the same",
            );

            Ok(Some(CommittedStringFilterField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
    }
}

pub fn merge_bool_field(
    uncommitted: Option<&UncommittedBoolField>,
    committed: Option<&CommittedBoolField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<CommittedBoolField>> {
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
                (BoolWrapper::False, false_docs),
                (BoolWrapper::True, true_docs),
            ]
            .into_iter()
            .map(|(k, mut v)| {
                v.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, v)
            });

            Ok(Some(CommittedBoolField::from_iter(
                committed.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), None) => {
            let (true_docs, false_docs) = uncommitted.clone_inner();
            let iter = vec![
                (BoolWrapper::False, false_docs),
                (BoolWrapper::True, true_docs),
            ]
            .into_iter()
            .map(|(k, mut v)| {
                v.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, v)
            });

            Ok(Some(CommittedBoolField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
                return Ok(None);
            }

            let (uncommitted_true_docs, uncommitted_false_docs) = uncommitted.clone_inner();
            let (mut committed_true_docs, mut committed_false_docs) = committed.clone_inner()?;

            committed_true_docs.extend(uncommitted_true_docs);
            committed_false_docs.extend(uncommitted_false_docs);

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path(),
                "Uncommitted and committed field paths should be the same",
            );

            Ok(Some(CommittedBoolField::from_data(
                committed.field_path().to_vec().into_boxed_slice(),
                committed_true_docs,
                committed_false_docs,
                data_dir,
            )?))
        }
    }
}

pub fn merge_string_field(
    uncommitted: Option<&UncommittedStringField>,
    committed: Option<&CommittedStringField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<CommittedStringField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed string fields are None. Never should happen");
        }
        (None, Some(committed)) => {
            if uncommitted_document_deletions.is_empty() {
                // No changes
                return Ok(None);
            }

            Ok(Some(CommittedStringField::from_committed(
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

            Ok(Some(CommittedStringField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                entries.into_iter(),
                length_per_documents,
                data_dir,
                uncommitted_document_deletions,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
                return Ok(None);
            }

            let length_per_documents = uncommitted.field_length_per_doc();
            // let uncommitted_iter = uncommitted.iter();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));
            let mut entries: Vec<_> = iter.collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path(),
                "Uncommitted and committed field paths should be the same",
            );

            Ok(Some(
                CommittedStringField::from_iter_and_committed(
                    uncommitted.field_path().to_vec().into_boxed_slice(),
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
    uncommitted: Option<&UncommittedVectorField>,
    committed: Option<&CommittedVectorField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
) -> Result<Option<CommittedVectorField>> {
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
                let new_field = CommittedVectorField::from_dump_and_iter(
                    committed.field_path().to_vec().into_boxed_slice(),
                    info.data_dir,
                    std::iter::empty(),
                    uncommitted_document_deletions,
                    info.model.0,
                    data_dir,
                )?;
                Ok(Some(new_field))
            }
        }
        (Some(uncommitted), None) => {
            let new_field = CommittedVectorField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                uncommitted
                    .iter()
                    .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id)),
                uncommitted.get_model(),
                data_dir,
            )?;
            Ok(Some(new_field))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
                return Ok(None);
            }

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path(),
                "Uncommitted and committed field paths should be the same",
            );

            let info = committed.get_field_info();

            // uncommitted and committed model has to be the same
            debug_assert_eq!(
                uncommitted.get_model(),
                info.model.0,
                "Uncommitted and committed models should be the same",
            );

            let new_field = CommittedVectorField::from_dump_and_iter(
                info.field_path,
                info.data_dir,
                uncommitted.iter(),
                uncommitted_document_deletions,
                uncommitted.get_model(),
                data_dir,
            )?;
            Ok(Some(new_field))
        }
    }
}
