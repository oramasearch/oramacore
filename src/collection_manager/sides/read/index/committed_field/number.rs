use core::f32;
use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    indexes::ordered_key::{BoundedValue, OrderedKeyIndex},
    types::{DocumentId, Number, NumberFilter, SerializableNumber},
};

#[derive(Debug)]
pub struct CommittedNumberField {
    field_path: Box<[String]>,
    inner: OrderedKeyIndex<SerializableNumber, DocumentId>,
    data_dir: PathBuf,
}

impl CommittedNumberField {
    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (SerializableNumber, HashSet<DocumentId>)>,
    {
        let inner = OrderedKeyIndex::from_iter(iter, data_dir.clone())?;
        Ok(Self {
            field_path,
            inner,
            data_dir,
        })
    }

    pub fn try_load(info: NumberFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let inner = OrderedKeyIndex::load(data_dir.clone())?;
        Ok(Self {
            inner,
            field_path: info.field_path,
            data_dir,
        })
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
        let s = self
            .inner
            .min_max()
            .context("Cannot get min madn max key for number index")?;
        let Some((min, max)) = s else {
            return Ok(CommittedNumberFieldStats {
                min: Number::F32(f32::INFINITY),
                max: Number::F32(f32::NEG_INFINITY),
            });
        };

        Ok(CommittedNumberFieldStats {
            min: min.0,
            max: max.0,
        })
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_number: &NumberFilter,
    ) -> Result<impl Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        let (min, max) = match filter_number {
            NumberFilter::Equal(value) => (
                (true, SerializableNumber(*value)),
                (true, SerializableNumber(*value)),
            ),
            NumberFilter::Between((min, max)) => (
                (true, SerializableNumber(*min)),
                (true, SerializableNumber(*max)),
            ),
            NumberFilter::GreaterThan(min) => (
                (false, SerializableNumber(*min)),
                (true, SerializableNumber::max_value()),
            ),
            NumberFilter::GreaterThanOrEqual(min) => (
                (true, SerializableNumber(*min)),
                (true, SerializableNumber::max_value()),
            ),
            NumberFilter::LessThan(max) => (
                (true, SerializableNumber::min_value()),
                (false, SerializableNumber(*max)),
            ),
            NumberFilter::LessThanOrEqual(max) => (
                (true, SerializableNumber::min_value()),
                (true, SerializableNumber(*max)),
            ),
        };

        let items = self
            .inner
            .get_items(min, max)
            .context("Cannot get items for number index")?;

        Ok(items.flat_map(|item| item.values))
    }

    pub fn iter(&self) -> impl Iterator<Item = (SerializableNumber, HashSet<DocumentId>)> + '_ {
        self.inner.iter()
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
mod tests {
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
        assert_eq!(output.len(), 2);
        assert_eq!(output, HashSet::from([DocumentId(3), DocumentId(4)]));

        let output = field
            .filter(&NumberFilter::GreaterThanOrEqual(Number::F32(1.0)))
            .unwrap()
            .collect::<HashSet<_>>();
        assert_eq!(output.len(), 4);

        let output = field
            .filter(&NumberFilter::LessThan(Number::F32(2.0)))
            .unwrap()
            .collect::<HashSet<_>>();
        assert_eq!(output.len(), 2);
        assert_eq!(output, HashSet::from([DocumentId(1), DocumentId(2)]));

        let output = field
            .filter(&NumberFilter::LessThanOrEqual(Number::F32(2.0)))
            .unwrap()
            .collect::<HashSet<_>>();
        assert_eq!(output.len(), 4);
    }
}
