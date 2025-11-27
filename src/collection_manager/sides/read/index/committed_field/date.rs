use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::read::index::{
        committed_field::number::get_iter,
        merge::{CommittedField, FieldMetadata},
        uncommitted_field::UncommittedDateFilterField,
    },
    merger::MergedIterator,
    types::{DateFilter, DocumentId, OramaDate},
};
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};

#[derive(Debug)]
pub struct CommittedDateField {
    field_path: Box<[String]>,
    vec: Vec<(i64, HashSet<DocumentId>)>,
    data_dir: PathBuf,
}

impl CommittedDateField {
    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (i64, HashSet<DocumentId>)>,
    {
        let vec: Vec<_> = iter.collect();
        let s = Self {
            field_path,
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit date field")?;

        Ok(s)
    }

    fn commit(&self) -> Result<()> {
        // Ensure the data directory exists
        create_if_not_exists(&self.data_dir).context("Failed to create data directory")?;

        BufferedFile::create_or_overwrite(self.data_dir.join("date_vec.bin"))
            .context("Failed to create date_vec.bin")?
            .write_bincode_data(&self.vec)
            .context("Failed to serialize date vec")?;

        Ok(())
    }

    pub fn get_field_info(&self) -> DateFieldInfo {
        DateFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn stats(&self) -> Result<CommittedDateFieldStats> {
        let min = self.vec.first().map(|(num, _)| *num);
        let max = self.vec.last().map(|(num, _)| *num);

        let (Some(min), Some(max)) = (min, max) else {
            return Ok(CommittedDateFieldStats {
                min: None,
                max: None,
            });
        };

        let min = OramaDate::try_from_i64(min);
        let max = OramaDate::try_from_i64(max);

        Ok(CommittedDateFieldStats { min, max })
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_date: &DateFilter,
    ) -> Result<impl Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        Ok(match filter_date {
            DateFilter::Equal(value) => {
                get_iter(&self.vec, (true, value.as_i64()), (true, value.as_i64()))
            }
            DateFilter::Between((min, max)) => {
                get_iter(&self.vec, (true, min.as_i64()), (true, max.as_i64()))
            }
            DateFilter::GreaterThan(min) => {
                get_iter(&self.vec, (false, min.as_i64()), (false, i64::MAX))
            }
            DateFilter::GreaterThanOrEqual(min) => {
                get_iter(&self.vec, (true, min.as_i64()), (false, i64::MAX))
            }
            DateFilter::LessThan(max) => {
                get_iter(&self.vec, (false, i64::MIN), (false, max.as_i64()))
            }
            DateFilter::LessThanOrEqual(max) => {
                get_iter(&self.vec, (false, i64::MIN), (true, max.as_i64()))
            }
        })
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (i64, HashSet<DocumentId>)> + '_ {
        self.vec.iter().cloned()
    }
}

impl CommittedField for CommittedDateField {
    type FieldMetadata = DateFieldInfo;
    type Uncommitted = UncommittedDateFilterField;

    fn from_uncommitted(
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: crate::collection_manager::sides::read::OffloadFieldConfig,
    ) -> Result<Self> {
        let iter = uncommitted
            .iter()
            // .map(|(n, v)| (SerializableNumber(n), v))
            .map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            });
        CommittedDateField::from_iter(
            uncommitted.field_path().to_vec().into_boxed_slice(),
            iter,
            data_dir,
        )
    }

    #[allow(deprecated)]
    fn try_load(
        metadata: Self::FieldMetadata,
        offload_config: crate::collection_manager::sides::read::OffloadFieldConfig,
    ) -> Result<Self> {
        let data_dir = metadata.data_dir;
        // Try to load from the new format first, and track if we need to commit (migrate from old format)
        let (vec, needs_commit) = match BufferedFile::open(data_dir.join("date_vec.bin"))
            .and_then(|f| f.read_bincode_data::<Vec<(i64, HashSet<DocumentId>)>>())
        {
            Ok(vec) => {
                // Successfully loaded from new format, no commit needed
                (vec, false)
            }
            Err(_) => {
                // Failed to load from new format, try to migrate from old format
                use oramacore_lib::data_structures::ordered_key::OrderedKeyIndex;
                let inner = OrderedKeyIndex::<i64, DocumentId>::load(data_dir.clone())?;

                let items = inner
                    .get_items((true, i64::MIN), (true, i64::MAX))
                    .context("Cannot get items for date field")?;

                let mut vec: Vec<_> = items.map(|item| (item.key, item.values)).collect();
                // ensure the order is by key. This should not be necessary, but we do it to ensure consistency.
                vec.sort_by_key(|(key, _)| *key);
                // Need to commit to save in new format
                (vec, true)
            }
        };

        let s = Self {
            field_path: metadata.field_path,
            vec,
            data_dir,
        };

        // Only commit if we migrated from old format
        if needs_commit {
            s.commit().context("Failed to commit date field")?;
        }

        Ok(s)
    }

    fn add_uncommitted(
        &self,
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: crate::collection_manager::sides::read::OffloadFieldConfig,
    ) -> Result<Self> {
        let uncommitted_iter = uncommitted.iter();
        let committed_iter = self.iter();

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
            self.field_path(),
            "Uncommitted and committed field paths should be the same",
        );

        CommittedDateField::from_iter(
            uncommitted.field_path().to_vec().into_boxed_slice(),
            iter,
            data_dir,
        )
    }

    fn metadata(&self) -> Self::FieldMetadata {
        self.get_field_info()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DateFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct CommittedDateFieldStats {
    pub min: Option<OramaDate>,
    pub max: Option<OramaDate>,
}

impl FieldMetadata for DateFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }
}
