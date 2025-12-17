use core::f32;
use std::{
    collections::{BTreeMap, HashSet},
    ops::Bound,
};

use serde::Serialize;

use anyhow::Result;

use crate::{
    collection_manager::sides::read::index::merge::{Field, Filterable, UncommittedField},
    types::{DocumentId, Number, NumberFilter},
};

#[derive(Debug)]
pub struct UncommittedNumberField {
    field_path: Box<[String]>,
    inner: BTreeMap<Number, HashSet<DocumentId>>,
}

impl UncommittedNumberField {
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

    pub fn insert(&mut self, doc_id: DocumentId, value: Number) {
        self.inner.entry(value).or_default().insert(doc_id);
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_number: &NumberFilter,
    ) -> impl Iterator<Item = DocumentId> + 'iter
    where
        's: 'iter,
    {
        inner_filter(&self.inner, filter_number)
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + '_ {
        self.inner
            .iter()
            .map(|(number, doc_ids)| (*number, doc_ids.clone()))
    }
}

#[inline]
fn inner_filter<'tree, 'iter>(
    tree: &'tree BTreeMap<Number, HashSet<DocumentId>>,
    filter: &NumberFilter,
) -> impl Iterator<Item = DocumentId> + 'iter
where
    'tree: 'iter,
{
    fn flat_doc<'tree>(
        (_, doc_ids): (&'tree Number, &'tree HashSet<DocumentId>),
    ) -> impl Iterator<Item = DocumentId> + 'tree {
        doc_ids.iter().copied()
    }

    match filter {
        NumberFilter::Equal(value) => tree
            .range((Bound::Included(value), Bound::Included(value)))
            .flat_map(flat_doc),
        NumberFilter::LessThan(value) => tree
            .range((Bound::Unbounded, Bound::Excluded(value)))
            .flat_map(flat_doc),
        NumberFilter::LessThanOrEqual(value) => tree
            .range((Bound::Unbounded, Bound::Included(value)))
            .flat_map(flat_doc),
        NumberFilter::GreaterThan(value) => tree
            .range((Bound::Excluded(value), Bound::Unbounded))
            .flat_map(flat_doc),
        NumberFilter::GreaterThanOrEqual(value) => tree
            .range((Bound::Included(value), Bound::Unbounded))
            .flat_map(flat_doc),
        NumberFilter::Between((min, max)) => tree
            .range((Bound::Included(min), Bound::Included(max)))
            .flat_map(flat_doc),
    }
}

#[derive(Serialize, Debug)]
pub struct UncommittedNumberFieldStats {
    pub min: Number,
    pub max: Number,
    pub count: usize,
}

impl UncommittedField for UncommittedNumberField {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn clear(&mut self) {
        self.inner = Default::default();
    }
}

impl Field for UncommittedNumberField {
    type FieldStats = UncommittedNumberFieldStats;

    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }

    fn stats(&self) -> UncommittedNumberFieldStats {
        if self.inner.is_empty() {
            return UncommittedNumberFieldStats {
                min: Number::F32(f32::INFINITY),
                max: Number::F32(f32::NEG_INFINITY),
                count: 0,
            };
        }

        let (Some((min, _)), Some((max, _))) =
            (self.inner.first_key_value(), self.inner.last_key_value())
        else {
            return UncommittedNumberFieldStats {
                min: Number::F32(f32::INFINITY),
                max: Number::F32(f32::NEG_INFINITY),
                count: 0,
            };
        };

        UncommittedNumberFieldStats {
            min: *min,
            max: *max,
            count: self.len(),
        }
    }
}

impl Filterable for UncommittedNumberField {
    type FilterParam = NumberFilter;

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
