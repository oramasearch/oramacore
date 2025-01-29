use std::{
    collections::{BTreeMap, HashSet},
    ops::Bound,
};

use crate::{
    indexes::number::{Number, NumberFilter},
    types::DocumentId,
};

#[derive(Debug)]
pub struct NumberField {
    inner: BTreeMap<Number, HashSet<DocumentId>>,
}

impl NumberField {
    pub fn empty() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, doc_id: DocumentId, value: Number) {
        self.inner
            .entry(value)
            .or_insert_with(HashSet::new)
            .insert(doc_id);
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

    pub fn iter(&self) -> impl Iterator<Item = (Number, HashSet<DocumentId>)> + '_ {
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
        doc_ids.iter().map(move |d| *d)
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
