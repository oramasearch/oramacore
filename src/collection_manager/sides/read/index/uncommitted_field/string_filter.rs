use std::collections::{HashMap, HashSet};

use serde::Serialize;

use anyhow::Result;

use crate::{
    collection_manager::sides::read::index::{
        filter::Filterable,
        merge::{Field, UncommittedField},
    },
    types::DocumentId,
};

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
}

#[derive(Serialize, Debug)]
pub struct UncommittedStringFilterFieldStats {
    pub key_count: usize,
    pub document_count: usize,
    pub keys: Option<Vec<String>>,
}

impl UncommittedField for UncommittedStringFilterField {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn clear(&mut self) {
        self.inner = Default::default();
    }
}

impl Field for UncommittedStringFilterField {
    type FieldStats = UncommittedStringFilterFieldStats;

    fn field_path(&self) -> &[String] {
        self.field_path.as_ref()
    }

    fn stats(&self) -> UncommittedStringFilterFieldStats {
        let doc_count = self.inner.values().map(|v| v.len()).sum();
        let keys = Some(self.inner.keys().cloned().collect());

        UncommittedStringFilterFieldStats {
            key_count: self.inner.len(),
            document_count: doc_count,
            keys,
        }
    }
}

impl Filterable for UncommittedStringFilterField {
    type FilterParam = String;

    fn filter<'s, 'iter>(
        &'s self,
        filter_param: &Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let iter = self
            .inner
            .get(filter_param.as_str())
            .map(|doc_ids| {
                Box::new(doc_ids.iter().copied()) as Box<dyn Iterator<Item = DocumentId>>
            })
            .unwrap_or_else(|| Box::new(std::iter::empty()));
        Ok(iter)
    }
}
