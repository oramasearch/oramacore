use std::{
    collections::{BTreeMap, HashSet},
    ops::Bound,
};

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::warn;

use crate::{collection_manager::sides::Offset, offset_storage::OffsetStorage, types::DocumentId};

use super::{merger::DataToCommit, Number, NumberFilter};

#[derive(Debug, Clone, Copy)]
pub enum State {
    Left,
    Right,
}

#[derive(Debug)]
pub struct InnerUncommittedNumberFieldIndex {
    pub left: BTreeMap<Number, HashSet<DocumentId>>,
    pub right: BTreeMap<Number, HashSet<DocumentId>>,
    pub state: State,
}

impl InnerUncommittedNumberFieldIndex {
    fn new() -> Self {
        Self {
            left: BTreeMap::new(),
            right: BTreeMap::new(),
            state: State::Left,
        }
    }

    fn insert(&mut self, value: Number, document_id: DocumentId) -> Result<()> {
        let doc_ids = match self.state {
            State::Left => &mut self.left,
            State::Right => &mut self.right,
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

#[derive(Debug)]
pub struct UncommittedNumberFieldIndex {
    pub inner: RwLock<(OffsetStorage, InnerUncommittedNumberFieldIndex)>,
}

impl UncommittedNumberFieldIndex {
    pub fn new(offset: Offset) -> Self {
        let offset_storage = OffsetStorage::new();
        offset_storage.set_offset(offset);
        Self {
            inner: RwLock::new((offset_storage, InnerUncommittedNumberFieldIndex::new())),
        }
    }

    pub async fn insert(
        &self,
        offset: Offset,
        value: Number,
        document_id: DocumentId,
    ) -> Result<()> {
        let mut inner = self.inner.write().await;
        // Ignore inserts with lower offset
        if offset <= inner.0.get_offset() {
            warn!("Skip insert number with lower offset");
            return Ok(());
        }
        inner.1.insert(value, document_id)?;
        inner.0.set_offset(offset);
        Ok(())
    }

    pub async fn filter(
        &self,
        filter: NumberFilter,
        doc_ids: &mut HashSet<DocumentId>,
    ) -> Result<()> {
        let inner = self.inner.read().await;
        inner.1.filter(filter, doc_ids)?;
        Ok(())
    }

    pub async fn take(&self) -> Result<DataToCommit<'_>> {
        // There is a subtle bug here: this method should be called only once at time
        // If this method is called twice at the same time, the result will be wrong
        // This is because:
        // - the state is flipped
        // - the tree is cleared on `DataToCommit` drop (which happens at the end of the commit)
        // So, if we flipped the state twice, the tree will be cleared twice and we could lose data
        // TODO: investigate how to make this method safe to be called multiple times

        let mut inner = self.inner.write().await;
        let tree: BTreeMap<Number, HashSet<DocumentId>> = match inner.1.state {
            State::Left => inner.1.left.clone(),
            State::Right => inner.1.right.clone(),
        };
        let state_to_clear = inner.1.state;

        // It is safe to put here because we are holding the lock
        let current_offset = inner.0.get_offset();

        // Route the writes to the other side
        inner.1.state = match state_to_clear {
            State::Left => State::Right,
            State::Right => State::Left,
        };
        drop(inner);

        Ok(DataToCommit {
            tree,
            index: self,
            state_to_clear,
            current_offset,
        })
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

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_number_uncommitted() -> Result<()> {
        let index = UncommittedNumberFieldIndex::new(Offset(0));

        index
            .insert(Offset(1), Number::I32(1), DocumentId(1))
            .await?;
        index
            .insert(Offset(2), Number::I32(2), DocumentId(2))
            .await?;
        index
            .insert(Offset(3), Number::I32(3), DocumentId(3))
            .await?;

        let taken = index.take().await?;

        let collected: Vec<_> = taken.iter().collect();
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
