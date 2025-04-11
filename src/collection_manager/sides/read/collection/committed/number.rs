use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    indexes::ordered_key::{BoundedValue, OrderedKeyIndex},
    types::{DocumentId, Number, NumberFilter, SerializableNumber},
};

#[derive(Debug)]
pub struct NumberField {
    inner: OrderedKeyIndex<SerializableNumber, DocumentId>,
    data_dir: PathBuf,
}

impl NumberField {
    pub fn from_iter<I>(iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (SerializableNumber, HashSet<DocumentId>)>,
    {
        let inner = OrderedKeyIndex::from_iter(iter, data_dir.clone())?;
        Ok(Self { inner, data_dir })
    }

    pub fn load(info: NumberFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let inner = OrderedKeyIndex::load(data_dir.clone())?;
        Ok(Self { inner, data_dir })
    }

    pub fn get_field_info(&self) -> NumberFieldInfo {
        NumberFieldInfo {
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn get_stats(&self) -> Result<NumberCommittedFieldStats> {
        let (min, max) = self
            .inner
            .min_max()
            .context("Cannot get min madn max key for number index")?;
        Ok(NumberCommittedFieldStats {
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
            NumberFilter::Equal(value) => (SerializableNumber(*value), SerializableNumber(*value)),
            NumberFilter::Between((min, max)) => {
                (SerializableNumber(*min), SerializableNumber(*max))
            }
            NumberFilter::GreaterThan(min) => {
                (SerializableNumber(*min), SerializableNumber::max_value())
            }
            NumberFilter::GreaterThanOrEqual(min) => {
                (SerializableNumber(*min), SerializableNumber::max_value())
            }
            NumberFilter::LessThan(max) => {
                (SerializableNumber::min_value(), SerializableNumber(*max))
            }
            NumberFilter::LessThanOrEqual(max) => {
                (SerializableNumber::min_value(), SerializableNumber(*max))
            }
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
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct NumberCommittedFieldStats {
    pub min: Number,
    pub max: Number,
}
