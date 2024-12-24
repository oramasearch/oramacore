use std::{collections::{BTreeMap, HashSet}, ops::Bound};

use crate::document_storage::DocumentId;

use super::{Number, NumberFilter};

#[derive(Debug, Default)]
pub struct UncommittedNumberFieldIndex {
    inner: BTreeMap<Number, HashSet<DocumentId>>,
}

impl UncommittedNumberFieldIndex {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, value: Number, document_id: DocumentId) {
        let doc_ids = self.inner.entry(value).or_default();
        doc_ids.insert(document_id);
    }

    pub fn filter(&self, filter: NumberFilter, doc_ids: &mut HashSet<DocumentId>) {
        match filter {
            NumberFilter::Equal(value) => {
                if let Some(d) = self.inner.get(&value) {
                    doc_ids.extend(d.iter().cloned());
                }
            }
            NumberFilter::LessThan(value) => doc_ids.extend(
                self.inner
                    .range((Bound::Unbounded, Bound::Excluded(&value)))
                    .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
            ),
            NumberFilter::LessThanOrEqual(value) => doc_ids.extend(
                self.inner
                    .range((Bound::Unbounded, Bound::Included(&value)))
                    .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
            ),
            NumberFilter::GreaterThan(value) => doc_ids.extend(
                self.inner
                    .range((Bound::Excluded(&value), Bound::Unbounded))
                    .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
            ),
            NumberFilter::GreaterThanOrEqual(value) => doc_ids.extend(
                self.inner
                    .range((Bound::Included(&value), Bound::Unbounded))
                    .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
            ),
            NumberFilter::Between((min, max)) => doc_ids.extend(
                self.inner
                    .range((Bound::Included(&min), Bound::Included(&max)))
                    .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
            ),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Number, HashSet<DocumentId>)> + '_ {
        self.inner.iter()
            .map(|(k, v)| (*k, v.clone()))
    }
}
