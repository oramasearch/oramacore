use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    indexes::ordered_key::{BoundedValue, OrderedKeyIndex},
    types::{DateFilter, DocumentId, OramaDate},
};

#[derive(Debug)]
pub struct CommittedDateField {
    field_path: Box<[String]>,
    inner: OrderedKeyIndex<i64, DocumentId>,
    data_dir: PathBuf,
}

impl CommittedDateField {
    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (i64, HashSet<DocumentId>)>,
    {
        let inner = OrderedKeyIndex::from_iter(iter, data_dir.clone())?;
        Ok(Self {
            field_path,
            inner,
            data_dir,
        })
    }

    pub fn try_load(info: DateFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let inner = OrderedKeyIndex::load(data_dir.clone())?;
        Ok(Self {
            inner,
            field_path: info.field_path,
            data_dir,
        })
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
        let s = self
            .inner
            .min_max()
            .context("Cannot get min and max key for date index")?;
        let Some((min, max)) = s else {
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
        let (min, max) = match filter_date {
            DateFilter::Equal(value) => ((true, value.as_i64()), (true, value.as_i64())),
            DateFilter::Between((min, max)) => ((true, min.as_i64()), (true, max.as_i64())),
            DateFilter::GreaterThan(min) => ((false, min.as_i64()), (true, i64::MAX)),
            DateFilter::GreaterThanOrEqual(min) => ((true, min.as_i64()), (true, i64::MAX)),
            DateFilter::LessThan(max) => ((true, i64::MIN), (false, max.as_i64())),
            DateFilter::LessThanOrEqual(max) => ((true, i64::MIN), (true, max.as_i64())),
        };

        let items = self
            .inner
            .get_items(min, max)
            .context("Cannot get items for date index")?;

        Ok(items.flat_map(|item| item.values))
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, HashSet<DocumentId>)> + '_ {
        self.inner.iter()
    }
}

impl BoundedValue for i64 {
    fn max_value() -> Self {
        i64::MAX
    }

    fn min_value() -> Self {
        i64::MIN
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
