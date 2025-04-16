use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::types::DocumentId;

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

    pub fn filter<'s, 'iter>(&'s self, filter: &str) -> impl Iterator<Item = DocumentId> + 'iter
    where
        's: 'iter,
    {
        self.inner
            .get(filter)
            .map(|doc_ids| doc_ids.iter().cloned())
            .unwrap_or_default()
    }

    pub fn get_string_values<'s, 'iter>(&'s self) -> impl Iterator<Item = &'s str> + 'iter
    where
        's: 'iter,
    {
        self.inner.keys().map(|k| k.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (String, HashSet<DocumentId>)> + '_ {
        self.inner
            .iter()
            .map(|(k, doc_ids)| (k.clone(), doc_ids.clone()))
    }

    pub fn get_stats(&self) -> StringFilterUncommittedFieldStats {
        let doc_count = self.inner.values().map(|v| v.len()).sum();

        StringFilterUncommittedFieldStats {
            variant_count: self.inner.len(),
            doc_count,
        }
    }
}

#[derive(Serialize, Debug)]
pub struct StringFilterUncommittedFieldStats {
    pub variant_count: usize,
    pub doc_count: usize,
}
