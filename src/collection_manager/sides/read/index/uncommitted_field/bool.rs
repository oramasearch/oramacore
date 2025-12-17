use std::collections::HashSet;

use serde::Serialize;

use anyhow::Result;

use crate::{
    collection_manager::sides::read::index::merge::{Field, Filterable, UncommittedField},
    types::DocumentId,
};

#[derive(Debug)]
pub struct UncommittedBoolField {
    field_path: Box<[String]>,
    inner: (HashSet<DocumentId>, HashSet<DocumentId>),
}

impl UncommittedBoolField {
    pub fn empty(field_path: Box<[String]>) -> Self {
        Self {
            field_path,
            inner: (HashSet::new(), HashSet::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.0.len() + self.inner.1.len()
    }

    pub fn insert(&mut self, id: DocumentId, value: bool) {
        if value {
            self.inner.0.insert(id);
        } else {
            self.inner.1.insert(id);
        }
    }

    pub fn filter<'s, 'iter>(&'s self, value: bool) -> impl Iterator<Item = DocumentId> + 'iter
    where
        's: 'iter,
    {
        if value {
            self.inner.0.iter().map(f)
        } else {
            self.inner.1.iter().map(f)
        }
    }

    pub fn clone_inner(&self) -> (HashSet<DocumentId>, HashSet<DocumentId>) {
        self.inner.clone()
    }
}

fn f(d: &DocumentId) -> DocumentId {
    *d
}

impl UncommittedField for UncommittedBoolField {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn clear(&mut self) {
        self.inner.0 = HashSet::new();
        self.inner.1 = HashSet::new();
    }
}

impl Field for UncommittedBoolField {
    type FieldStats = UncommittedBoolFieldStats;

    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }

    fn stats(&self) -> Self::FieldStats {
        UncommittedBoolFieldStats {
            false_count: self.inner.1.len(),
            true_count: self.inner.0.len(),
        }
    }
}

#[derive(Serialize, Debug)]
pub struct UncommittedBoolFieldStats {
    pub false_count: usize,
    pub true_count: usize,
}

impl Filterable for UncommittedBoolField {
    type FilterParam = bool;

    fn filter<'s, 'iter>(
        &'s self,
        filter_param: Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        // Use the existing filter logic: true docs are in inner.0, false docs in inner.1
        let iter: Box<dyn Iterator<Item = DocumentId> + 'iter> = if filter_param {
            Box::new(self.inner.0.iter().copied())
        } else {
            Box::new(self.inner.1.iter().copied())
        };
        Ok(iter)
    }
}
