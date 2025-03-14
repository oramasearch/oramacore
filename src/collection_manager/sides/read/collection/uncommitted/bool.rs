use std::collections::HashSet;

use serde::Serialize;

use crate::types::DocumentId;

#[derive(Debug)]
pub struct BoolField {
    inner: (HashSet<DocumentId>, HashSet<DocumentId>),
}

impl BoolField {
    pub fn empty() -> Self {
        Self {
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

    pub fn get_stats(&self) -> BoolUncommittedFieldStats {
        BoolUncommittedFieldStats {
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

#[derive(Serialize, Debug)]
pub struct BoolUncommittedFieldStats {
    pub false_count: usize,
    pub true_count: usize,
}
