use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::Bound,
};

use serde::Serialize;

use crate::{
    collection_manager::dto::{Number, NumberFilter},
    types::DocumentId,
};

#[derive(Debug)]
pub struct StringFilterField {
    inner: HashMap<String, HashSet<DocumentId>>,
}

impl StringFilterField {
    pub fn empty() -> Self {
        Self {
            inner: Default::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn insert(&mut self, doc_id: DocumentId, value: String) {
        self.inner.entry(value).or_default().insert(doc_id);
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter: &String,
    ) -> impl Iterator<Item = DocumentId> + 'iter
    where
        's: 'iter,
    {
        self.inner.get(filter).map(|doc_ids| doc_ids.iter().cloned()).unwrap_or_default()
    }

    pub fn iter(&self) -> impl Iterator<Item = (String, HashSet<DocumentId>)> + '_ {
        self.inner
            .iter()
            .map(|(k, doc_ids)| (k.clone(), doc_ids.clone()))
    }

    pub fn get_stats(&self) -> StringFilterUncommittedFieldStats {
        if self.inner.is_empty() {
            return StringFilterUncommittedFieldStats {
                variant_count: 0,
            };
        }

        StringFilterUncommittedFieldStats {
            variant_count: self.inner.len(),
        }
    }
}

#[derive(Serialize, Debug)]
pub struct StringFilterUncommittedFieldStats {
    pub variant_count: usize
}
