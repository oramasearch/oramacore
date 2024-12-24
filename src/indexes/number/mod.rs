use std::{
    collections::HashSet,
    path::PathBuf,
};

use anyhow::Result;
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use committed::CommittedNumberFieldIndex;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::debug;
use uncommitted::UncommittedNumberFieldIndex;

use crate::{collection_manager::dto::FieldId, document_storage::DocumentId};

// mod linear;
// mod merge_iter;
// mod serializable_number;
// mod stats;
mod n;
mod uncommitted;
mod committed;

pub use n::Number;

pub struct CommitConfig {}

#[derive(Debug)]
pub struct NumberIndex {
    uncommitted: DashMap<FieldId, UncommittedNumberFieldIndex>,
    committed: DashMap<FieldId, CommittedNumberFieldIndex>,
    // committed: LinearNumberIndex,
    base_path: PathBuf,
    max_size_per_chunk: usize,
}

impl NumberIndex {
    pub fn new(base_path: PathBuf, max_size_per_chunk: usize) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;
        Ok(Self {
            uncommitted: Default::default(),
            committed: Default::default(),
            // committed: LinearNumberIndex::from_fs(base_path.clone(), max_size_per_chunk)?,
            base_path,
            max_size_per_chunk,
        })
    }

    pub fn add(&self, doc_id: DocumentId, field_id: FieldId, value: Number) {
        debug!(
            "Adding number index: doc_id: {:?}, field_id: {:?}, value: {:?}",
            doc_id, field_id, value
        );
        let mut uncommitted = self.uncommitted.entry(field_id).or_default();

        uncommitted.insert(value, doc_id);
    }

    pub fn filter(&self, field_id: FieldId, filter: NumberFilter) -> Result<HashSet<DocumentId>> {
        let mut doc_ids = if let Some(committed) = self.committed.get(&field_id) {
            committed.filter(&filter)?
        } else {
            HashSet::new()
        };

        if let Some(uncommitted) = self.uncommitted.get(&field_id) {
            uncommitted.filter(filter, &mut doc_ids);
        };

        Ok(doc_ids)
    }

    pub fn commit(&self, _config: CommitConfig) -> Result<()> {
        for entry in &self.uncommitted {
            let field_id = entry.key();
            let uncommitted = entry.value();
            
            let committed = self.committed.get(field_id);

            // 

            let a = if let Some(committed) = committed {
                panic!("Not implemented");
            } else {
                CommittedNumberFieldIndex::from_iter(uncommitted.iter())
            };
        }

        /*
        let merge_iter = MergeIter {
            iter1: committed.map(|p| p.unwrap()),
            iter2: dd.into_iter(),
            state: MergeIterState::Unstarted,
        };

        let new_linear = LinearNumberIndex::from_iter(
            merge_iter,
            FromIterConfig {
                max_size_per_chunk: self.max_size_per_chunk,
                base_path: self.base_path.clone(),
            },
        )?;

        // This is safe because `commit` method keeps **`&mut self`**.
        // So no concurrent access is possible.
        self.committed = new_linear;
        self.uncommitted.clear();
        */

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub enum NumberFilter {
    #[serde(rename = "eq")]
    Equal(#[schema(inline)] Number),
    #[serde(rename = "gt")]
    GreaterThan(#[schema(inline)] Number),
    #[serde(rename = "gte")]
    GreaterThanOrEqual(#[schema(inline)] Number),
    #[serde(rename = "lt")]
    LessThan(#[schema(inline)] Number),
    #[serde(rename = "lte")]
    LessThanOrEqual(#[schema(inline)] Number),
    #[serde(rename = "between")]
    Between(#[schema(inline)] (Number, Number)),
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::test_utils::generate_new_path;

    use super::*;

    macro_rules! test_number_filter {
        ($fn_name: ident, $b: expr) => {
            #[test]
            fn $fn_name() {
                let mut index = NumberIndex::new(generate_new_path(), 2048).unwrap();

                index.add(DocumentId(0), FieldId(0), 0.into());
                index.add(DocumentId(1), FieldId(0), 1.into());
                index.add(DocumentId(2), FieldId(0), 2.into());
                index.add(DocumentId(3), FieldId(0), 3.into());
                index.add(DocumentId(4), FieldId(0), 4.into());
                index.add(DocumentId(5), FieldId(0), 2.into());

                let a = $b;

                a(&index);

                index.commit(CommitConfig {}).unwrap();

                a(&index);
            }
        };
    }

    test_number_filter!(test_number_index_filter_eq, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );
    });
    test_number_filter!(test_number_index_filter_lt, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::LessThan(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(0), DocumentId(1)])
        );
    });
    test_number_filter!(test_number_index_filter_lt_equal, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::LessThanOrEqual(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![
                DocumentId(0),
                DocumentId(1),
                DocumentId(2),
                DocumentId(5)
            ])
        );
    });
    test_number_filter!(test_number_index_filter_gt, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::GreaterThan(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(3), DocumentId(4)])
        );
    });
    test_number_filter!(test_number_index_filter_gt_equal, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::GreaterThanOrEqual(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![
                DocumentId(3),
                DocumentId(4),
                DocumentId(2),
                DocumentId(5)
            ])
        );
    });
    test_number_filter!(test_number_index_filter_between, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::Between((2.into(), 3.into())))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(3), DocumentId(2), DocumentId(5)])
        );
    });

    #[test]
    fn test_number_commit() {
        let mut index = NumberIndex::new(generate_new_path(), 2048).unwrap();

        index.add(DocumentId(0), FieldId(0), 0.into());
        index.add(DocumentId(1), FieldId(0), 1.into());
        index.add(DocumentId(2), FieldId(0), 2.into());
        index.add(DocumentId(3), FieldId(0), 3.into());
        index.add(DocumentId(4), FieldId(0), 4.into());
        index.add(DocumentId(5), FieldId(0), 2.into());

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );

        index.commit(CommitConfig {}).unwrap();

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );
    }

    #[test]
    fn test_indexes_number_save_and_load_from_fs() -> Result<()> {
        let mut index = NumberIndex::new(generate_new_path(), 2048).unwrap();

        let iter = (0..1_000).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        for (number, (doc_id, field_id)) in iter {
            index.add(doc_id, field_id, number);
        }

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(output, HashSet::from_iter(vec![DocumentId(2)]));

        index.commit(CommitConfig {})?;

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(output, HashSet::from_iter(vec![DocumentId(2)]));

        Ok(())
    }
}
