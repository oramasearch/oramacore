use std::{
    collections::{BTreeMap, HashSet},
    ops::Bound,
};

use serde::Serialize;

use anyhow::Result;

use crate::{
    collection_manager::sides::read::index::merge::{Field, Filterable, UncommittedField},
    types::{DateFilter, DocumentId, OramaDate},
};

#[derive(Debug)]
pub struct UncommittedDateFilterField {
    field_path: Box<[String]>,
    inner: BTreeMap<i64, HashSet<DocumentId>>,
}

impl UncommittedDateFilterField {
    pub fn empty(field_path: Box<[String]>) -> Self {
        Self {
            field_path,
            inner: BTreeMap::new(),
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn insert(&mut self, doc_id: DocumentId, value: i64) {
        self.inner.entry(value).or_default().insert(doc_id);
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_date: &DateFilter,
    ) -> impl Iterator<Item = DocumentId> + 'iter
    where
        's: 'iter,
    {
        inner_filter(&self.inner, filter_date)
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (i64, HashSet<DocumentId>)> + '_ {
        self.inner
            .iter()
            .map(|(number, doc_ids)| (*number, doc_ids.clone()))
    }
}

impl UncommittedField for UncommittedDateFilterField {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn clear(&mut self) {
        self.inner = Default::default();
    }
}

impl Field for UncommittedDateFilterField {
    type FieldStats = UncommittedDateFieldStats;

    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }

    fn stats(&self) -> UncommittedDateFieldStats {
        if self.inner.is_empty() {
            return UncommittedDateFieldStats {
                min: None,
                max: None,
                count: 0,
            };
        }

        let (Some((min, _)), Some((max, _))) =
            (self.inner.first_key_value(), self.inner.last_key_value())
        else {
            return UncommittedDateFieldStats {
                min: None,
                max: None,
                count: 0,
            };
        };

        let min = OramaDate::try_from_i64(*min);
        let max = OramaDate::try_from_i64(*max);

        UncommittedDateFieldStats {
            min,
            max,
            count: self.len(),
        }
    }
}

#[inline]
fn inner_filter<'tree, 'iter>(
    tree: &'tree BTreeMap<i64, HashSet<DocumentId>>,
    filter: &DateFilter,
) -> impl Iterator<Item = DocumentId> + 'iter
where
    'tree: 'iter,
{
    fn flat_doc<'tree>(
        (_, doc_ids): (&'tree i64, &'tree HashSet<DocumentId>),
    ) -> impl Iterator<Item = DocumentId> + 'tree {
        doc_ids.iter().copied()
    }

    match filter {
        DateFilter::Equal(value) => tree
            .range((
                Bound::Included(value.as_i64()),
                Bound::Included(value.as_i64()),
            ))
            .flat_map(flat_doc),
        DateFilter::LessThan(value) => tree
            .range((Bound::Unbounded, Bound::Excluded(value.as_i64())))
            .flat_map(flat_doc),
        DateFilter::LessThanOrEqual(value) => tree
            .range((Bound::Unbounded, Bound::Included(value.as_i64())))
            .flat_map(flat_doc),
        DateFilter::GreaterThan(value) => tree
            .range((Bound::Excluded(value.as_i64()), Bound::Unbounded))
            .flat_map(flat_doc),
        DateFilter::GreaterThanOrEqual(value) => tree
            .range((Bound::Included(value.as_i64()), Bound::Unbounded))
            .flat_map(flat_doc),
        DateFilter::Between((min, max)) => tree
            .range((Bound::Included(min.as_i64()), Bound::Included(max.as_i64())))
            .flat_map(flat_doc),
    }
}

#[derive(Serialize, Debug)]
pub struct UncommittedDateFieldStats {
    pub min: Option<OramaDate>,
    pub max: Option<OramaDate>,
    pub count: usize,
}

impl Filterable for UncommittedDateFilterField {
    type FilterParam = DateFilter;

    fn filter<'s, 'iter>(
        &'s self,
        filter_param: Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        // Reuse the existing inner_filter function
        let iter = inner_filter(&self.inner, &filter_param);
        Ok(Box::new(iter))
    }
}
