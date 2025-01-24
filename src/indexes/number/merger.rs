use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};

use crate::{collection_manager::sides::Offset, merger::MergedIterator, types::DocumentId};

use super::{
    committed::CommittedNumberFieldIndex,
    uncommitted::{State, UncommittedNumberFieldIndex},
    Number,
};

pub struct DataToCommit<'uncommitted> {
    pub index: &'uncommitted UncommittedNumberFieldIndex,
    pub state_to_clear: State,
    pub tree: BTreeMap<Number, HashSet<DocumentId>>,
    pub current_offset: Offset,
}

impl DataToCommit<'_> {
    pub fn get_offset(&self) -> Offset {
        self.current_offset
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Number, HashSet<DocumentId>)> + '_ {
        self.tree.iter().map(|(k, v)| (*k, v.clone()))
    }

    pub async fn done(self) {
        let mut lock = self.index.inner.write().await;
        let tree = match self.state_to_clear {
            State::Left => &mut lock.1.left,
            State::Right => &mut lock.1.right,
        };
        tree.clear();
    }
}

pub async fn merge(
    offset: Offset,
    data_to_commit: DataToCommit<'_>,
    committed: &CommittedNumberFieldIndex,
    new_data_dir: PathBuf,
) -> Result<CommittedNumberFieldIndex> {
    let uncommitted_iter = data_to_commit.iter();

    let iter = MergedIterator::new(
        committed.iter(),
        uncommitted_iter,
        |_, v| v,
        |_, mut v1, v2| {
            v1.extend(v2);
            v1
        },
    );

    let committed: Result<CommittedNumberFieldIndex> = tokio::task::block_in_place(|| {
        let committed = CommittedNumberFieldIndex::from_iter(offset, iter, new_data_dir.clone())?;
        committed.commit(new_data_dir)?;
        anyhow::Result::Ok(committed)
    });
    let committed = committed
        .context("Failed to create committed index")?;

    data_to_commit.done().await;

    Ok(committed)
}

pub async fn create(
    offset: Offset,
    data_to_commit: DataToCommit<'_>,
    new_data_dir: PathBuf,
) -> Result<CommittedNumberFieldIndex> {
    let uncommitted_iter = data_to_commit.iter();

    let committed: Result<CommittedNumberFieldIndex> = tokio::task::block_in_place(|| {
        let committed = CommittedNumberFieldIndex::from_iter(offset, uncommitted_iter, new_data_dir.clone())?;
        committed.commit(new_data_dir)?;
        anyhow::Result::Ok(committed)
    });
    let committed = committed
        .context("Failed to create committed index")?;

    data_to_commit.done().await;

    Ok(committed)
}

#[cfg(test)]
mod test {
    use crate::test_utils::generate_new_path;

    use super::*;

    async fn create_uncommitted_data(v: Vec<(i32, Vec<u64>)>) -> UncommittedNumberFieldIndex {
        let uncommitted = UncommittedNumberFieldIndex::new(Offset(0));
        let mut offset = 1;
        for (k, v) in v {
            for id in v {
                offset +=1;
                uncommitted.insert(Offset(offset), Number::I32(k), DocumentId(id)).await.unwrap();
            }
        }

        uncommitted
    }

    fn create_committed_data(v: Vec<(i32, Vec<u64>)>) -> CommittedNumberFieldIndex {
        let iter = v.into_iter().map(|(k, v)| {
            (
                k.into(),
                v.into_iter()
                    .map(|id| DocumentId(id))
                    .collect::<HashSet<_>>(),
            )
        });
        CommittedNumberFieldIndex::from_iter(Offset(0), iter, generate_new_path()).unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_number_merge() {
        let uncommitted_data = create_uncommitted_data(vec![(1, vec![1]), (2, vec![2]), (3, vec![3])]).await;
        let committed_data = create_committed_data(vec![(1, vec![1]), (2, vec![2]), (3, vec![3])]);

        let merged = merge(
            Offset(0),
            uncommitted_data.take().await.unwrap(),
            &committed_data,
            generate_new_path(),
        )
        .await.unwrap();

        assert_eq!(
            vec![
                (Number::I32(1), HashSet::from_iter([DocumentId(1), DocumentId(1)])),
                (Number::I32(2), HashSet::from_iter([DocumentId(2), DocumentId(2)])),
                (Number::I32(3), HashSet::from_iter([DocumentId(3), DocumentId(3)]))
            ],
            merged.iter().collect::<Vec<_>>()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_number_merge_2() {
        let uncommitted_data = create_uncommitted_data(vec![(1, vec![1]), (3, vec![2]), (5, vec![3])]).await;
        let committed_data = create_committed_data(vec![(2, vec![1]), (4, vec![2]), (6, vec![3])]);

        let merged = merge(
            Offset(0),
            uncommitted_data.take().await.unwrap(),
            &committed_data,
            generate_new_path(),
        )
        .await.unwrap();

        assert_eq!(
            vec![
                (Number::I32(1), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(2), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(3), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(4), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(5), HashSet::from_iter([DocumentId(3)])),
                (Number::I32(6), HashSet::from_iter([DocumentId(3)]))
            ],
            merged.iter().collect::<Vec<_>>()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_number_merge_3() {
        let uncommitted_data = create_uncommitted_data(vec![(2, vec![1]), (4, vec![2]), (6, vec![3])]).await;
        let committed_data = create_committed_data(vec![(1, vec![1]), (3, vec![2]), (5, vec![3])]);

        let merged = merge(
            Offset(0),
            uncommitted_data.take().await.unwrap(),
            &committed_data,
            generate_new_path(),
        )
        .await.unwrap();

        assert_eq!(
            vec![
                (Number::I32(1), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(2), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(3), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(4), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(5), HashSet::from_iter([DocumentId(3)])),
                (Number::I32(6), HashSet::from_iter([DocumentId(3)]))
            ],
            merged.iter().collect::<Vec<_>>()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_number_merge_4() {
        let uncommitted_data = create_uncommitted_data(vec![(1, vec![1]), (2, vec![2]), (3, vec![3])]).await;
        let committed_data = create_committed_data(vec![(4, vec![1]), (5, vec![2]), (6, vec![3])]);

        let merged = merge(
            Offset(0),
            uncommitted_data.take().await.unwrap(),
            &committed_data,
            generate_new_path(),
        )
        .await.unwrap();

        assert_eq!(
            vec![
                (Number::I32(1), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(2), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(3), HashSet::from_iter([DocumentId(3)])),
                (Number::I32(4), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(5), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(6), HashSet::from_iter([DocumentId(3)]))
            ],
            merged.iter().collect::<Vec<_>>()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_number_merge_5() {
        let uncommitted_data = create_uncommitted_data(vec![(4, vec![1]), (5, vec![2]), (6, vec![3])]).await;
        let committed_data = create_committed_data(vec![(1, vec![1]), (2, vec![2]), (3, vec![3])]);

        let merged = merge(
            Offset(0),
            uncommitted_data.take().await.unwrap(),
            &committed_data,
            generate_new_path(),
        )
        .await.unwrap();

        assert_eq!(
            vec![
                (Number::I32(1), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(2), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(3), HashSet::from_iter([DocumentId(3)])),
                (Number::I32(4), HashSet::from_iter([DocumentId(1)])),
                (Number::I32(5), HashSet::from_iter([DocumentId(2)])),
                (Number::I32(6), HashSet::from_iter([DocumentId(3)]))
            ],
            merged.iter().collect::<Vec<_>>()
        );
    }
}
