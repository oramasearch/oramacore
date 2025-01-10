use std::{
    collections::{BTreeMap, HashSet},
    ops::Bound,
    sync::RwLock,
};

use anyhow::Result;

use crate::types::DocumentId;

use super::{Number, NumberFilter};

#[derive(Debug, Default)]
struct InnerUncommittedNumberFieldIndex {
    left: BTreeMap<Number, HashSet<DocumentId>>,
    right: BTreeMap<Number, HashSet<DocumentId>>,
    state: bool,
}

impl InnerUncommittedNumberFieldIndex {
    fn insert(&mut self, value: Number, document_id: DocumentId) -> Result<()> {
        let doc_ids = if self.state {
            &mut self.left
        } else {
            &mut self.right
        };
        let doc_ids = doc_ids.entry(value).or_default();
        doc_ids.insert(document_id);
        Ok(())
    }

    fn filter(&self, filter: NumberFilter, doc_ids: &mut HashSet<DocumentId>) -> Result<()> {
        inner_filter(&self.left, doc_ids, &filter);
        inner_filter(&self.right, doc_ids, &filter);
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct UncommittedNumberFieldIndex {
    inner: RwLock<InnerUncommittedNumberFieldIndex>,
}

impl UncommittedNumberFieldIndex {
    pub fn insert(&self, value: Number, document_id: DocumentId) -> Result<()> {
        let mut inner = match self.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        inner.insert(value, document_id)?;
        Ok(())
    }

    pub fn filter(&self, filter: NumberFilter, doc_ids: &mut HashSet<DocumentId>) -> Result<()> {
        let inner = match self.inner.read() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        inner.filter(filter, doc_ids)?;
        Ok(())
    }

    pub fn take(&self) -> Result<UncommittedNumberFieldIndexTaken> {
        // There is a subtle bug here: this method should be called only once at time
        // If this method is called twice at the same time, the result will be wrong
        // This is because:
        // - the state is flipped
        // - the tree is cleared on `UncommittedNumberFieldIndexTaken` drop (which happens at the end of the commit)
        // So, if we flipped the state twice, the tree will be cleared twice and we could lose data
        // TODO: investigate how to make this method safe to be called multiple times

        let mut inner = match self.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        let tree: BTreeMap<Number, HashSet<DocumentId>> = if inner.state {
            inner.left.clone()
        } else {
            inner.right.clone()
        };
        let current_state = inner.state;

        // Route the writes to the other side
        inner.state = !current_state;
        drop(inner);

        let tree_len = tree.len();

        Ok(UncommittedNumberFieldIndexTaken {
            tree: Box::new(tree.into_iter()),
            tree_len,
            index: self,
            state: current_state,
        })
    }
}

pub struct UncommittedNumberFieldIndexTaken<'index> {
    tree: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>>,
    tree_len: usize,
    state: bool,
    index: &'index UncommittedNumberFieldIndex,
}
impl Iterator for UncommittedNumberFieldIndexTaken<'_> {
    type Item = (Number, HashSet<DocumentId>);

    fn next(&mut self) -> Option<Self::Item> {
        self.tree.next()
    }
}
impl Drop for UncommittedNumberFieldIndexTaken<'_> {
    fn drop(&mut self) {
        let mut lock = self.index.inner.write().unwrap();
        let tree = if self.state {
            &mut lock.left
        } else {
            &mut lock.right
        };
        tree.clear();
    }
}
impl UncommittedNumberFieldIndexTaken<'_> {
    pub fn len(&self) -> usize {
        self.tree_len
    }
}

fn inner_filter(
    tree: &BTreeMap<Number, HashSet<DocumentId>>,
    doc_ids: &mut HashSet<DocumentId>,
    filter: &NumberFilter,
) {
    match filter {
        NumberFilter::Equal(value) => {
            if let Some(d) = tree.get(value) {
                doc_ids.extend(d.iter().cloned());
            }
        }
        NumberFilter::LessThan(value) => doc_ids.extend(
            tree.range((Bound::Unbounded, Bound::Excluded(value)))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
        ),
        NumberFilter::LessThanOrEqual(value) => doc_ids.extend(
            tree.range((Bound::Unbounded, Bound::Included(value)))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
        ),
        NumberFilter::GreaterThan(value) => doc_ids.extend(
            tree.range((Bound::Excluded(value), Bound::Unbounded))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
        ),
        NumberFilter::GreaterThanOrEqual(value) => doc_ids.extend(
            tree.range((Bound::Included(value), Bound::Unbounded))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
        ),
        NumberFilter::Between((min, max)) => doc_ids.extend(
            tree.range((Bound::Included(min), Bound::Included(max)))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexes_number_uncommitted() -> Result<()> {
        let index = UncommittedNumberFieldIndex::default();

        index.insert(Number::I32(1), DocumentId(1))?;
        index.insert(Number::I32(2), DocumentId(2))?;
        index.insert(Number::I32(3), DocumentId(3))?;

        let taken = index.take()?;

        let collected: Vec<_> = taken.into_iter().collect();
        assert_eq!(
            collected,
            vec![
                (Number::I32(1), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(2), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(3), HashSet::from_iter([DocumentId(3)])),
            ]
        );

        Ok(())
    }
}
