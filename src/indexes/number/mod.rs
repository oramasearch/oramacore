use std::{
    borrow::Cow,
    collections::{BTreeMap, HashSet},
    path::PathBuf,
};

use anyhow::Result;
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use dashmap::DashMap;
use linear::{FromIterConfig, LinearNumberIndex};
use merge_iter::{MergeIter, MergeIterState};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{collection_manager::dto::FieldId, document_storage::DocumentId};

mod linear;
mod merge_iter;
mod serializable_number;
mod stats;
mod r#type;

pub use r#type::Number;

pub struct CommitConfig {}

#[derive(Debug)]
pub struct NumberIndex {
    uncommitted: DashMap<FieldId, BTreeMap<Number, HashSet<DocumentId>>>,
    committed: LinearNumberIndex,
    base_path: PathBuf,
    max_size_per_chunk: usize,
}

impl NumberIndex {
    pub fn new(base_path: PathBuf, max_size_per_chunk: usize) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;
        Ok(Self {
            uncommitted: Default::default(),
            committed: LinearNumberIndex::from_fs(base_path.clone(), max_size_per_chunk)?,
            base_path,
            max_size_per_chunk,
        })
    }

    pub fn add(&self, doc_id: DocumentId, field_id: FieldId, value: Number) {
        debug!(
            "Adding number index: doc_id: {:?}, field_id: {:?}, value: {:?}",
            doc_id, field_id, value
        );
        let mut btree = self.uncommitted.entry(field_id).or_default();
        let doc_ids = btree.entry(value).or_default();
        doc_ids.insert(doc_id);
    }

    pub fn filter(&self, field_id: FieldId, filter: NumberFilter) -> Result<HashSet<DocumentId>> {
        use std::ops::Bound;

        let mut doc_ids = self.committed.filter(field_id, &filter, 0)?;

        if let Some(btree) = self.uncommitted.get(&field_id) {
            match filter {
                NumberFilter::Equal(value) => {
                    if let Some(d) = btree.get(&value) {
                        doc_ids.extend(d.iter().cloned());
                    }
                }
                NumberFilter::LessThan(value) => doc_ids.extend(
                    btree
                        .range((Bound::Unbounded, Bound::Excluded(&value)))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::LessThanOrEqual(value) => doc_ids.extend(
                    btree
                        .range((Bound::Unbounded, Bound::Included(&value)))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::GreaterThan(value) => doc_ids.extend(
                    btree
                        .range((Bound::Excluded(&value), Bound::Unbounded))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::GreaterThanOrEqual(value) => doc_ids.extend(
                    btree
                        .range((Bound::Included(&value), Bound::Unbounded))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::Between((min, max)) => doc_ids.extend(
                    btree
                        .range((Bound::Included(&min), Bound::Included(&max)))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
            }
        };

        Ok(doc_ids)
    }

    pub fn commit(&mut self, _config: CommitConfig) -> Result<()> {
        let committed = self.committed.iter();

        let mut dd: BTreeMap<Number, Cow<Vec<(DocumentId, FieldId)>>> = BTreeMap::new();
        for e in self.uncommitted.iter() {
            let field_id = *e.key();
            let btree = e.value();
            for (number, doc_ids) in btree.iter() {
                let a = dd.entry(*number).or_default();
                for doc_id in doc_ids.iter() {
                    a.to_mut().push((*doc_id, field_id));
                }
            }
        }

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

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub enum NumberFilter {
    Equal(#[schema(inline)] Number),
    GreaterThan(#[schema(inline)] Number),
    GreaterThanOrEqual(#[schema(inline)] Number),
    LessThan(#[schema(inline)] Number),
    LessThanOrEqual(#[schema(inline)] Number),
    Between(#[schema(inline)] (Number, Number)),
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs};

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
