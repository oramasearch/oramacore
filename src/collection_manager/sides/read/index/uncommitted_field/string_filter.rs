use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::types::DocumentId;

#[derive(Debug)]
pub struct UncommittedStringFilterField {
    field_path: Box<[String]>,
    inner: HashMap<String, HashSet<DocumentId>>,
}

impl UncommittedStringFilterField {
    pub fn empty(field_path: Box<[String]>) -> Self {
        Self {
            field_path,
            inner: Default::default(),
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.inner = Default::default();
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

    pub fn stats(&self) -> UncommittedStringFilterFieldStats {
        let doc_count = self.inner.values().map(|v| v.len()).sum();

        UncommittedStringFilterFieldStats {
            key_count: self.inner.len(),
            document_count: doc_count,
        }
    }
}

#[derive(Serialize, Debug)]
pub struct UncommittedStringFilterFieldStats {
    pub key_count: usize,
    pub document_count: usize,
}
