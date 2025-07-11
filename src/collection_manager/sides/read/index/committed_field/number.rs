use core::f32;
use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    indexes::ordered_key::BoundedValue,
    types::{DocumentId, Number, NumberFilter, SerializableNumber},
};
use fs::create_if_not_exists;

#[derive(Debug)]
pub struct CommittedNumberField {
    field_path: Box<[String]>,
    vec: Vec<(SerializableNumber, HashSet<DocumentId>)>,
    data_dir: PathBuf,
}

impl CommittedNumberField {
    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (SerializableNumber, HashSet<DocumentId>)>,
    {
        let vec: Vec<_> = iter.collect();
        let s = Self {
            field_path,
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    #[allow(deprecated)]
    pub fn try_load(info: NumberFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let vec = match std::fs::File::open(data_dir.join("number_vec.bin")) {
            Ok(file) => {
                bincode::deserialize_from::<_, Vec<(SerializableNumber, HashSet<DocumentId>)>>(file)
                    .context("Failed to deserialize number_vec.bin")?
            }
            Err(_) => {
                use crate::indexes::ordered_key::OrderedKeyIndex;
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
                vec
            }
        };

        let s = Self {
            field_path: info.field_path,
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    fn commit(&self) -> Result<()> {
        // Ensure the data directory exists
        create_if_not_exists(&self.data_dir).context("Failed to create data directory")?;

        let file_path = self.data_dir.join("number_vec.bin");
        let file = std::fs::File::create(&file_path).context("Failed to create number_vec.bin")?;
        bincode::serialize_into(file, &self.vec).context("Failed to serialize number vec")?;

        Ok(())
    }

    pub fn get_field_info(&self) -> NumberFieldInfo {
        NumberFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn stats(&self) -> Result<CommittedNumberFieldStats> {
        let min = self.vec.first().map(|(num, _)| num.0);
        let max = self.vec.last().map(|(num, _)| num.0);

        let (Some(min), Some(max)) = (min, max) else {
            return Ok(CommittedNumberFieldStats {
                min: Number::F32(f32::INFINITY),
                max: Number::F32(f32::NEG_INFINITY),
            });
        };

        Ok(CommittedNumberFieldStats { min, max })
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

    pub fn iter(&self) -> impl Iterator<Item = (SerializableNumber, HashSet<DocumentId>)> + '_ {
        self.vec.iter().cloned()
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

pub fn get_iter<'s, K: Ord + Eq>(
    vec: &'s Vec<(K, HashSet<DocumentId>)>,
    min: (bool, K),
    max: (bool, K),
) -> impl Iterator<Item = DocumentId> + 's {
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

#[cfg(test)]
mod tests_number {
    use crate::tests::utils::generate_new_path;

    use super::*;

    #[test]
    fn test_lt_gt() {
        let path = generate_new_path();
        let iter = vec![
            (
                SerializableNumber(Number::F32(1.0)),
                HashSet::from([DocumentId(1), DocumentId(2)]),
            ),
            (
                SerializableNumber(Number::F32(2.0)),
                HashSet::from([DocumentId(3), DocumentId(4)]),
            ),
        ]
        .into_iter();
        let field = CommittedNumberField::from_iter(
            vec!["a".to_string(), "b".to_string()].into_boxed_slice(),
            iter,
            path,
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
