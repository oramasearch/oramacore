use std::collections::HashSet;

use serde::Serialize;

use crate::{collection_manager::sides::read::index::merge::UncommittedField, types::DocumentId};

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

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn len(&self) -> usize {
        self.inner.0.len() + self.inner.1.len()
    }

    pub fn clear(&mut self) {
        self.inner.0 = HashSet::new();
        self.inner.1 = HashSet::new();
    }

    pub fn insert(&mut self, id: DocumentId, value: bool) {
        if value {
            self.inner.0.insert(id);
        } else {
            self.inner.1.insert(id);
        }
    }

    pub fn stats(&self) -> UncommittedBoolFieldStats {
        UncommittedBoolFieldStats {
            false_count: self.inner.1.len(),
            true_count: self.inner.0.len(),
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
}

#[derive(Serialize, Debug)]
pub struct UncommittedBoolFieldStats {
    pub false_count: usize,
    pub true_count: usize,
}
