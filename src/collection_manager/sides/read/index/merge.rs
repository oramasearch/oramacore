use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;

use crate::types::DocumentId;
use oramacore_lib::fs::create_if_not_exists;

use super::super::OffloadFieldConfig;

// pub use super::filter::Filterable;

pub trait CommittedFieldMetadata {
    fn data_dir(&self) -> &PathBuf;
    fn set_data_dir(&mut self, data_dir: PathBuf);
    fn field_path(&self) -> &Box<[String]>;
}

pub trait CommittedField: Field + Sized {
    type Uncommitted;
    type FieldMetadata;

    fn from_uncommitted(
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self>;

    fn try_load(metadata: Self::FieldMetadata, offload_config: OffloadFieldConfig) -> Result<Self>;

    fn add_uncommitted(
        &self,
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self>;

    fn metadata(&self) -> Self::FieldMetadata;
}
pub trait UncommittedField: Field {
    fn is_empty(&self) -> bool;

    fn clear(&mut self);
}

pub trait Field {
    type FieldStats;

    fn field_path(&self) -> &Box<[String]>;

    fn stats(&self) -> Self::FieldStats;
}

pub fn merge_field<
    CommittedMetadata: CommittedFieldMetadata,
    Uncommitted: UncommittedField,
    Committed: CommittedField<Uncommitted = Uncommitted, FieldMetadata = CommittedMetadata>,
>(
    uncommitted: Option<&Uncommitted>,
    committed: Option<&Committed>,
    data_dir: PathBuf,
    uncommitted_document_deletions: &HashSet<DocumentId>,
    is_promoted: bool,
    offload_config: &OffloadFieldConfig,
) -> Result<Option<Committed>> {
    create_if_not_exists(&data_dir).context("Failed to create data directory for field")?;

    match (uncommitted, committed) {
        (None, None) => {
            bail!("Both uncommitted and committed fields are None. Never should happen");
        }
        (None, Some(_)) => {
            // Uncommitted field is alway present in the index if there's a committed one
            bail!("Both uncommitted field is None. Never should happen");
        }
        (Some(uncommitted), None) => {
            let new_field = CommittedField::from_uncommitted(
                uncommitted,
                data_dir,
                uncommitted_document_deletions,
                *offload_config,
            )?;
            Ok(Some(new_field))
        }
        (Some(uncommitted), Some(committed)) => {
            // uncommitted and committed field_path has to be the same
            let committed_metadata = committed.metadata();
            let uncommitted_field_path = uncommitted.field_path();
            debug_assert_eq!(
                uncommitted_field_path,
                committed_metadata.field_path(),
                "Uncommitted and committed field paths should be the same",
            );

            if uncommitted.is_empty() && !is_promoted {
                // Nothing to merge, return None
                return Ok(None);
            }

            let mut info = committed.metadata();
            if uncommitted.is_empty() {
                if !is_promoted {
                    // Nothing to merge, return None
                    return Ok(None);
                }

                debug_assert_ne!(
                    info.data_dir(),
                    &data_dir,
                    "when promoting, data_dir should be different from the one in the field info"
                );

                let old_dir = info.data_dir().clone();

                // Copy the data from the old directory to the new one
                let options = CopyOptions::new()
                    // BAD: this is bad because if a crash happens during the copy,
                    // the data will be corrupted
                    // Instead we should... ???? WHAT?
                    // TODO: check if this is the right way to do it
                    .overwrite(true);
                copy_items(&[old_dir], data_dir.parent().unwrap(), &options)?;
                // And move the field to the new directory
                info.set_data_dir(data_dir);

                return Ok(Some(
                    CommittedField::try_load(info, *offload_config)
                        .context("Failed to load committed vector field")?,
                ));
            }

            let new_field = committed.add_uncommitted(
                uncommitted,
                data_dir,
                uncommitted_document_deletions,
                *offload_config,
            )?;
            Ok(Some(new_field))
        }
    }
}
