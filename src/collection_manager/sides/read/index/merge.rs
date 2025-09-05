use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;

use crate::merger::MergedIterator;
use crate::types::{DocumentId, SerializableNumber};
use oramacore_lib::fs::create_if_not_exists;

use super::super::OffloadFieldConfig;
use super::committed_field::*;
use super::uncommitted_field::*;

pub fn merge_number_field(
    uncommitted: Option<&UncommittedNumberField>,
    committed: Option<&CommittedNumberField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
    is_promoted: bool,
) -> Result<Option<CommittedNumberField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed number fields are None. Never should happen");
        }
        (None, Some(_)) => {
            bail!("Both uncommitted field is None. Never should happen");
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
                if is_promoted {
                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let mut info = committed.get_field_info();
                    debug_assert_ne!(
                        info.data_dir,
                        data_dir,
                        "when promoting, data_dir should be different from the one in the field info"
                    );

                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let old_dir = info.data_dir;

                    // Copy the data from the old directory to the new one
                    let options = CopyOptions::new()
                        // BAD: this is bad because if a crash happens during the copy,
                        // the data will be corrupted
                        // Instead we should... ???? WHAT?
                        // TODO: check if this is the right way to do it
                        .overwrite(true);
                    copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                    // And move the field to the new directory
                    info.data_dir = data_dir;

                    return Ok(Some(
                        CommittedNumberField::try_load(info)
                            .context("Failed to load committed string field")?,
                    ));
                }

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

pub fn merge_date_field(
    uncommitted: Option<&UncommittedDateFilterField>,
    committed: Option<&CommittedDateField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
    is_promoted: bool,
) -> Result<Option<CommittedDateField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed number fields are None. Never should happen");
        }
        (None, Some(_)) => {
            bail!("Both uncommitted field is None. Never should happen");
        }
        (Some(uncommitted), None) => {
            let iter = uncommitted
                .iter()
                // .map(|(n, v)| (SerializableNumber(n), v))
                .map(|(k, mut d)| {
                    d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                    (k, d)
                });
            Ok(Some(CommittedDateField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
                if is_promoted {
                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let mut info = committed.get_field_info();
                    debug_assert_ne!(
                        info.data_dir,
                        data_dir,
                        "when promoting, data_dir should be different from the one in the field info"
                    );

                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let old_dir = info.data_dir;

                    // Copy the data from the old directory to the new one
                    let options = CopyOptions::new()
                        // BAD: this is bad because if a crash happens during the copy,
                        // the data will be corrupted
                        // Instead we should... ???? WHAT?
                        // TODO: check if this is the right way to do it
                        .overwrite(true);
                    copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                    // And move the field to the new directory
                    info.data_dir = data_dir;

                    return Ok(Some(
                        CommittedDateField::try_load(info)
                            .context("Failed to load committed string field")?,
                    ));
                }

                return Ok(None);
            }

            let uncommitted_iter = uncommitted.iter(); // .map(|(n, v)| (SerializableNumber(n), v));
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

            Ok(Some(CommittedDateField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                data_dir,
            )?))
        }
    }
}

pub fn merge_geopoint_field(
    uncommitted: Option<&UncommittedGeoPointFilterField>,
    committed: Option<&CommittedGeoPointField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
    is_promoted: bool,
) -> Result<Option<CommittedGeoPointField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed number fields are None. Never should happen");
        }
        (None, Some(_)) => {
            bail!("Uncommitted field is None. Never should happen");
        }
        (Some(uncommitted), None) => {
            let mut tree = uncommitted.inner();
            tree.delete(uncommitted_document_deletions);
            let committed = CommittedGeoPointField::from_raw(
                tree,
                uncommitted.field_path().to_vec().into_boxed_slice(),
                data_dir,
            )?;
            Ok(Some(committed))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
                if is_promoted {
                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let mut info = committed.get_field_info();
                    debug_assert_ne!(
                        info.data_dir,
                        data_dir,
                        "when promoting, data_dir should be different from the one in the field info"
                    );

                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let old_dir = info.data_dir;

                    // Copy the data from the old directory to the new one
                    let options = CopyOptions::new()
                        // BAD: this is bad because if a crash happens during the copy,
                        // the data will be corrupted
                        // Instead we should... ???? WHAT?
                        // TODO: check if this is the right way to do it
                        .overwrite(true);
                    copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                    // And move the field to the new directory
                    info.data_dir = data_dir;

                    return Ok(Some(
                        CommittedGeoPointField::try_load(info)
                            .context("Failed to load committed string field")?,
                    ));
                }

                return Ok(None);
            }

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path(),
                "Uncommitted and committed field paths should be the same",
            );

            let info = committed.get_field_info();
            let mut field = CommittedGeoPointField::try_load(info)
                .context("Failed to load committed string field")?;

            field.update(uncommitted.iter(), uncommitted_document_deletions, data_dir)?;

            Ok(Some(field))
        }
    }
}

pub fn merge_string_filter_field(
    uncommitted: Option<&UncommittedStringFilterField>,
    committed: Option<&CommittedStringFilterField>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
    is_promoted: bool,
) -> Result<Option<CommittedStringFilterField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed number fields are None. Never should happen");
        }
        (None, Some(_)) => {
            bail!("Both uncommitted field is None. Never should happen");
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
                if is_promoted {
                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let mut info = committed.get_field_info();
                    debug_assert_ne!(
                        info.data_dir,
                        data_dir,
                        "when promoting, data_dir should be different from the one in the field info"
                    );

                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let old_dir = info.data_dir;

                    // Copy the data from the old directory to the new one
                    let options = CopyOptions::new()
                        // BAD: this is bad because if a crash happens during the copy,
                        // the data will be corrupted
                        // Instead we should... ???? WHAT?
                        // TODO: check if this is the right way to do it
                        .overwrite(true);
                    copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                    // And move the field to the new directory
                    info.data_dir = data_dir;

                    return Ok(Some(
                        CommittedStringFilterField::try_load(info)
                            .context("Failed to load committed string field")?,
                    ));
                }

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
    is_promoted: bool,
) -> Result<Option<CommittedBoolField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed bool fields are None. Never should happen");
        }
        (None, Some(_)) => {
            bail!("Both uncommitted field is None. Never should happen");
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
                if is_promoted {
                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let mut info = committed.get_field_info();
                    debug_assert_ne!(
                        info.data_dir,
                        data_dir,
                        "when promoting, data_dir should be different from the one in the field info"
                    );

                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let old_dir = info.data_dir;

                    // Copy the data from the old directory to the new one
                    let options = CopyOptions::new()
                        // BAD: this is bad because if a crash happens during the copy,
                        // the data will be corrupted
                        // Instead we should... ???? WHAT?
                        // TODO: check if this is the right way to do it
                        .overwrite(true);
                    copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                    // And move the field to the new directory
                    info.data_dir = data_dir;

                    return Ok(Some(
                        CommittedBoolField::try_load(info)
                            .context("Failed to load committed string field")?,
                    ));
                }

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
    is_promoted: bool,
    offload_config: &OffloadFieldConfig,
) -> Result<Option<CommittedStringField>> {
    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed string fields are None. Never should happen");
        }
        (None, Some(_)) => {
            bail!("Both uncommitted field is None. Never should happen");
        }
        (Some(uncommitted), None) => {
            let length_per_documents = uncommitted.field_length_per_doc();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));

            Ok(Some(CommittedStringField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                iter,
                length_per_documents,
                data_dir,
                uncommitted_document_deletions,
                *offload_config,
            )?))
        }
        (Some(uncommitted), Some(committed)) => {
            if uncommitted.is_empty() {
                if is_promoted {
                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let mut info = committed.get_field_info();
                    debug_assert_ne!(
                        info.data_dir,
                        data_dir,
                        "when promoting, data_dir should be different from the one in the field info"
                    );

                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let old_dir = info.data_dir;

                    // Copy the data from the old directory to the new one
                    let options = CopyOptions::new()
                        // BAD: this is bad because if a crash happens during the copy,
                        // the data will be corrupted
                        // Instead we should... ???? WHAT?
                        // TODO: check if this is the right way to do it
                        .overwrite(true);
                    copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                    // And move the field to the new directory
                    info.data_dir = data_dir;

                    return Ok(Some(
                        CommittedStringField::try_load(info, *offload_config)
                            .context("Failed to load committed string field")?,
                    ));
                }
                return Ok(None);
            }

            let length_per_documents = uncommitted.field_length_per_doc();
            let iter = uncommitted.iter().map(|(n, v)| (n, v.clone()));

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path().as_ref(),
                "Uncommitted and committed field paths should be the same",
            );

            Ok(Some(
                CommittedStringField::from_iter_and_committed(
                    uncommitted.field_path().to_vec().into_boxed_slice(),
                    iter,
                    committed,
                    length_per_documents,
                    data_dir,
                    uncommitted_document_deletions,
                    *offload_config,
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
    is_promoted: bool,
    offload_config: &OffloadFieldConfig,
) -> Result<Option<CommittedVectorField>> {
    create_if_not_exists(&data_dir).context("Failed to create data directory for vector field")?;

    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed vector fields are None. Never should happen");
        }
        (None, Some(_)) => {
            // Uncommitted field is alway present in the index if there's a committed one
            bail!("Both uncommitted field is None. Never should happen");
        }
        (Some(uncommitted), None) => {
            let new_field = CommittedVectorField::from_iter(
                uncommitted.field_path().to_vec().into_boxed_slice(),
                uncommitted
                    .iter()
                    .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id)),
                uncommitted.get_model(),
                data_dir,
                *offload_config,
            )?;
            Ok(Some(new_field))
        }
        (Some(uncommitted), Some(committed)) => {
            let mut info = committed.get_field_info();
            if uncommitted.is_empty() {
                if is_promoted {
                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    debug_assert_ne!(
                        info.data_dir,
                        data_dir,
                        "when promoting, data_dir should be different from the one in the field info"
                    );

                    create_if_not_exists(&data_dir)
                        .context("Failed to create data directory for vector field")?;

                    let old_dir = info.data_dir;

                    // Copy the data from the old directory to the new one
                    let options = CopyOptions::new()
                        // BAD: this is bad because if a crash happens during the copy,
                        // the data will be corrupted
                        // Instead we should... ???? WHAT?
                        // TODO: check if this is the right way to do it
                        .overwrite(true);
                    copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                    // And move the field to the new directory
                    info.data_dir = data_dir;

                    return Ok(Some(
                        CommittedVectorField::try_load(info, *offload_config)
                            .context("Failed to load committed vector field")?,
                    ));
                }

                return Ok(None);
            }

            // uncommitted and committed field_path has to be the same
            debug_assert_eq!(
                uncommitted.field_path(),
                committed.field_path().as_ref(),
                "Uncommitted and committed field paths should be the same",
            );

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
                *offload_config,
            )?;
            Ok(Some(new_field))
        }
    }
}
