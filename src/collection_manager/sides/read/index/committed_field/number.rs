use core::f32;
use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use oramacore_lib::data_structures::ordered_key::BoundedValue;
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::read::index::{
        merge::{CommittedField, CommittedFieldMetadata, Field, Filterable},
        uncommitted_field::UncommittedNumberField,
    },
    merger::MergedIterator,
    types::{DocumentId, Number, NumberFilter, SerializableNumber},
};
use oramacore_lib::fs::BufferedFile;

#[derive(Debug)]
pub struct CommittedNumberField {
    field_path: Box<[String]>,
    vec: Vec<(SerializableNumber, HashSet<DocumentId>)>,
    data_dir: PathBuf,
}

impl CommittedNumberField {
    fn commit(&self) -> Result<()> {
        BufferedFile::create_or_overwrite(self.data_dir.join("number_vec.bin"))
            .context("Failed to create number_vec.bin")?
            .write_bincode_data(&self.vec)
            .context("Failed to serialize number vec")?;

        Ok(())
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_number: &NumberFilter,
    ) -> Result<impl Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        Ok(match filter_number {
            NumberFilter::Equal(value) => get_iter(
                &self.vec,
                (true, SerializableNumber(*value)),
                (true, SerializableNumber(*value)),
            ),
            NumberFilter::Between((min, max)) => get_iter(
                &self.vec,
                (true, SerializableNumber(*min)),
                (true, SerializableNumber(*max)),
            ),
            NumberFilter::GreaterThan(min) => get_iter(
                &self.vec,
                (false, SerializableNumber(*min)),
                (false, SerializableNumber::max_value()),
            ),
            NumberFilter::GreaterThanOrEqual(min) => get_iter(
                &self.vec,
                (true, SerializableNumber(*min)),
                (false, SerializableNumber::max_value()),
            ),
            NumberFilter::LessThan(max) => get_iter(
                &self.vec,
                (false, SerializableNumber::min_value()),
                (false, SerializableNumber(*max)),
            ),
            NumberFilter::LessThanOrEqual(max) => get_iter(
                &self.vec,
                (false, SerializableNumber::min_value()),
                (true, SerializableNumber(*max)),
            ),
        })
    }

    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = (SerializableNumber, HashSet<DocumentId>)> + '_ {
        self.vec.iter().cloned()
    }
}

impl CommittedField for CommittedNumberField {
    type FieldMetadata = NumberFieldInfo;
    type Uncommitted = UncommittedNumberField;

    fn from_uncommitted(
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: crate::collection_manager::sides::read::OffloadFieldConfig,
    ) -> Result<Self> {
        let vec: Vec<_> = uncommitted
            .iter()
            .map(|(n, v)| (SerializableNumber(n), v))
            .map(|(k, mut d)| {
                d.retain(|doc_id| !uncommitted_document_deletions.contains(doc_id));
                (k, d)
            })
            .collect();

        let s = Self {
            field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    #[allow(deprecated)]
    fn try_load(
        metadata: Self::FieldMetadata,
        _offload_config: crate::collection_manager::sides::read::OffloadFieldConfig,
    ) -> Result<Self> {
        let data_dir = metadata.data_dir;
        // Try to load from the new format first, and track if we need to commit (migrate from old format)
        let (vec, needs_commit) = match BufferedFile::open(data_dir.join("number_vec.bin"))
            .and_then(|f| f.read_bincode_data::<Vec<(SerializableNumber, HashSet<DocumentId>)>>())
        {
            Ok(vec) => {
                // Successfully loaded from new format, no commit needed
                (vec, false)
            }
            Err(_) => {
                // Failed to load from new format, try to migrate from old format
                use oramacore_lib::data_structures::ordered_key::OrderedKeyIndex;
                let inner =
                    OrderedKeyIndex::<SerializableNumber, DocumentId>::load(data_dir.clone())?;

                let items = inner
                    .get_items(
                        (true, SerializableNumber::min_value()),
                        (true, SerializableNumber::max_value()),
                    )
                    .context("Cannot get items for number field")?;

                let mut vec: Vec<_> = items.map(|item| (item.key, item.values)).collect();
                // ensure the order is by key. This should not be necessary, but we do it to ensure consistency.
                vec.sort_by_key(|(key, _)| key.0);
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
            s.commit().context("Failed to commit number field")?;
        }

        Ok(s)
    }

    fn add_uncommitted(
        &self,
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: crate::collection_manager::sides::read::OffloadFieldConfig,
    ) -> Result<Self> {
        let uncommitted_iter = uncommitted.iter().map(|(n, v)| (SerializableNumber(n), v));
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
            field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    fn metadata(&self) -> Self::FieldMetadata {
        NumberFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

impl Field for CommittedNumberField {
    type FieldStats = CommittedNumberFieldStats;

    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }

    fn stats(&self) -> CommittedNumberFieldStats {
        let min = self.vec.first().map(|(num, _)| num.0);
        let max = self.vec.last().map(|(num, _)| num.0);

        let (Some(min), Some(max)) = (min, max) else {
            return CommittedNumberFieldStats {
                min: Number::F32(f32::INFINITY),
                max: Number::F32(f32::NEG_INFINITY),
            };
        };

        CommittedNumberFieldStats { min, max }
    }
}

impl Filterable for CommittedNumberField {
    type FilterParam = NumberFilter;

    fn filter<'s, 'iter>(
        &'s self,
        filter_param: Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        // Reuse the existing filter logic by calling the existing method
        let iter = match &filter_param {
            NumberFilter::Equal(value) => get_iter(
                &self.vec,
                (true, SerializableNumber(*value)),
                (true, SerializableNumber(*value)),
            ),
            NumberFilter::Between((min, max)) => get_iter(
                &self.vec,
                (true, SerializableNumber(*min)),
                (true, SerializableNumber(*max)),
            ),
            NumberFilter::GreaterThan(min) => get_iter(
                &self.vec,
                (false, SerializableNumber(*min)),
                (false, SerializableNumber::max_value()),
            ),
            NumberFilter::GreaterThanOrEqual(min) => get_iter(
                &self.vec,
                (true, SerializableNumber(*min)),
                (false, SerializableNumber::max_value()),
            ),
            NumberFilter::LessThan(max) => get_iter(
                &self.vec,
                (false, SerializableNumber::min_value()),
                (false, SerializableNumber(*max)),
            ),
            NumberFilter::LessThanOrEqual(max) => get_iter(
                &self.vec,
                (false, SerializableNumber::min_value()),
                (true, SerializableNumber(*max)),
            ),
        };
        Ok(Box::new(iter))
    }
}

impl BoundedValue for SerializableNumber {
    fn max_value() -> Self {
        SerializableNumber(Number::F32(f32::INFINITY))
    }

    fn min_value() -> Self {
        SerializableNumber(Number::F32(f32::NEG_INFINITY))
    }
}

pub fn get_iter<K: Ord + Eq>(
    vec: &[(K, HashSet<DocumentId>)],
    min: (bool, K),
    max: (bool, K),
) -> impl Iterator<Item = DocumentId> + '_ {
    let min_index = match vec.binary_search_by_key(&&min.1, |(num, _)| num) {
        Ok(index) => {
            // vec[index] is == min.1
            // If we want to include the min value, we take the index as is.
            // Otherwise, we want to exclude it, taking the next index.
            if min.0 {
                index
            } else {
                index + 1
            }
        }
        Err(index) => {
            // vec[index] is > min.1
            // So, we don't need extra check for the value inclusion.
            index
        }
    };

    let max_index = match vec.binary_search_by_key(&&max.1, |(num, _)| num) {
        Ok(index) => {
            // vec[index] is == max.1
            // If we want to include the max value, we need to +1 due to the exclusive nature of the range.
            // Otherwise, we take the index as is.
            if max.0 {
                index + 1
            } else {
                index
            }
        }
        Err(index) => {
            // vec[index] is > max.1
            // So, we don't need extra check for the value inclusion.
            index
        }
    };

    vec[min_index..max_index]
        .iter()
        .flat_map(move |(_, ids)| ids.iter().cloned())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NumberFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct CommittedNumberFieldStats {
    pub min: Number,
    pub max: Number,
}

impl CommittedFieldMetadata for NumberFieldInfo {
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

#[cfg(test)]
mod tests_number {
    use std::time::Duration;

    use oramacore_lib::fs::create_if_not_exists;

    use crate::{
        collection_manager::sides::read::OffloadFieldConfig, tests::utils::generate_new_path,
    };

    use super::*;

    #[test]
    fn test_lt_gt() {
        let path = generate_new_path();
        create_if_not_exists(&path).unwrap();

        let mut uncommitted = UncommittedNumberField::empty(
            vec!["a".to_string(), "b".to_string()].into_boxed_slice(),
        );
        uncommitted.insert(DocumentId(1), Number::F32(1.0));
        uncommitted.insert(DocumentId(2), Number::F32(1.0));
        uncommitted.insert(DocumentId(3), Number::F32(2.0));
        uncommitted.insert(DocumentId(4), Number::F32(2.0));

        let field = CommittedNumberField::from_uncommitted(
            &uncommitted,
            path,
            &HashSet::new(),
            OffloadFieldConfig {
                unload_window: Duration::from_secs(30 * 60).into(),
                slot_count_exp: 8,
                slot_size_exp: 4,
            },
        )
        .unwrap();

        let output = field
            .filter(&NumberFilter::GreaterThan(Number::F32(1.0)))
            .unwrap()
            .collect::<HashSet<_>>();
        assert_eq!(output, HashSet::from([DocumentId(3), DocumentId(4)]));

        let output = field
            .filter(&NumberFilter::GreaterThanOrEqual(Number::F32(1.0)))
            .unwrap()
            .collect::<HashSet<_>>();
        assert_eq!(
            output,
            HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3), DocumentId(4)])
        );

        let output = field
            .filter(&NumberFilter::LessThan(Number::F32(2.0)))
            .unwrap()
            .collect::<HashSet<_>>();
        assert_eq!(output, HashSet::from([DocumentId(1), DocumentId(2)]));

        let output = field
            .filter(&NumberFilter::LessThanOrEqual(Number::F32(2.0)))
            .unwrap()
            .collect::<HashSet<_>>();
        assert_eq!(
            output,
            HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3), DocumentId(4)])
        );
    }
}
