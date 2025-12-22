use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::read::{
        index::{
            committed_field::number::get_iter,
            filter::Filterable,
            merge::{CommittedField, CommittedFieldMetadata, Field},
            uncommitted_field::UncommittedDateFilterField,
        },
        OffloadFieldConfig,
    },
    merger::MergedIterator,
    types::{DateFilter, DocumentId, OramaDate},
};
use oramacore_lib::fs::BufferedFile;

#[derive(Debug)]
pub struct CommittedDateField {
    field_path: Box<[String]>,
    vec: Vec<(i64, HashSet<DocumentId>)>,
    data_dir: PathBuf,
}

impl CommittedDateField {
    fn commit(&self) -> Result<()> {
        BufferedFile::create_or_overwrite(self.data_dir.join("date_vec.bin"))
            .context("Failed to create date_vec.bin")?
            .write_bincode_data(&self.vec)
            .context("Failed to serialize date vec")?;
        Ok(())
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
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let vec: Vec<_> = uncommitted
            .iter()
            .map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            })
            .collect();

        let s = Self {
            field_path: uncommitted.field_path().into(),
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit date field")?;
        Ok(s)
    }

    #[allow(deprecated)]
    fn try_load(
        metadata: Self::FieldMetadata,
        _offload_config: OffloadFieldConfig,
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
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let uncommitted_iter = uncommitted.iter();
        let committed_iter = self.iter();

        let vec: Vec<_> = MergedIterator::new(
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
        })
        .collect();

        let s = Self {
            field_path: uncommitted.field_path().into(),
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit date field")?;
        Ok(s)
    }

    fn metadata(&self) -> Self::FieldMetadata {
        DateFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

impl Field for CommittedDateField {
    type FieldStats = CommittedDateFieldStats;

    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }

    fn stats(&self) -> CommittedDateFieldStats {
        let min = self.vec.first().map(|(num, _)| *num);
        let max = self.vec.last().map(|(num, _)| *num);

        let (Some(min), Some(max)) = (min, max) else {
            return CommittedDateFieldStats {
                min: None,
                max: None,
            };
        };

        let min = OramaDate::try_from_i64(min);
        let max = OramaDate::try_from_i64(max);

        CommittedDateFieldStats { min, max }
    }
}

impl Filterable for CommittedDateField {
    type FilterParam = DateFilter;

    fn filter<'s, 'iter>(
        &'s self,
        filter_param: &Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let iter = match filter_param {
            DateFilter::Equal(value) => {
                let timestamp = value.as_i64();
                get_iter(&self.vec, (true, timestamp), (true, timestamp))
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
        };
        Ok(Box::new(iter))
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

impl CommittedFieldMetadata for DateFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }
    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }
}
