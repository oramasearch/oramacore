use std::{collections::HashSet, path::PathBuf};

use anyhow::Result;
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use committed::{merge::merge, CommittedNumberFieldIndex};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::debug;
use uncommitted::UncommittedNumberFieldIndex;

use crate::{
    collection_manager::{dto::FieldId, sides::read::CommitConfig},
    document_storage::DocumentId,
};

mod committed;
mod n;
mod uncommitted;

pub use n::Number;

pub struct NumberIndexConfig {}

#[derive(Debug)]
pub struct NumberIndex {
    uncommitted: DashMap<FieldId, UncommittedNumberFieldIndex>,
    committed: DashMap<FieldId, CommittedNumberFieldIndex>,
}

impl NumberIndex {
    pub fn try_new(_: NumberIndexConfig) -> Result<Self> {
        Ok(Self {
            uncommitted: Default::default(),
            committed: Default::default(),
        })
    }

    pub fn add(&self, doc_id: DocumentId, field_id: FieldId, value: Number) -> Result<()> {
        debug!(
            "Adding number index: doc_id: {:?}, field_id: {:?}, value: {:?}",
            doc_id, field_id, value
        );
        let uncommitted = self.uncommitted.entry(field_id).or_default();
        uncommitted.insert(value, doc_id)?;

        Ok(())
    }

    pub fn filter(&self, field_id: FieldId, filter: NumberFilter) -> Result<HashSet<DocumentId>> {
        let mut doc_ids = if let Some(committed) = self.committed.get(&field_id) {
            committed.filter(&filter)?
        } else {
            HashSet::new()
        };

        if let Some(uncommitted) = self.uncommitted.get(&field_id) {
            uncommitted.filter(filter, &mut doc_ids)?;
        };

        Ok(doc_ids)
    }

    pub fn commit(&self, config: &CommitConfig) -> Result<()> {
        for entry in &self.uncommitted {
            let field_id = entry.key();
            let uncommitted = entry.value();

            let committed = self.committed.get(field_id);

            let base_dir = config.folder_to_commit.join(format!("{}", field_id.0));

            let new_committed_number = if let Some(committed) = committed {
                let data = uncommitted.take()?;

                CommittedNumberFieldIndex::from_iter(
                    merge(
                        committed.iter(),
                        data.into_iter(),
                        |_, mut committed, uncommitted| {
                            committed.extend(uncommitted);
                            committed
                        },
                    ),
                    base_dir,
                )
            } else {
                let data = uncommitted.take()?;
                CommittedNumberFieldIndex::from_iter(data, base_dir)
            }?;

            self.committed.insert(*field_id, new_committed_number);
        }

        let p = config.folder_to_commit.join("committed");
        std::fs::write(p, config.epoch.to_string())?;

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
                let index = NumberIndex::try_new(NumberIndexConfig {}).unwrap();

                index.add(DocumentId(0), FieldId(0), 0.into()).unwrap();
                index.add(DocumentId(1), FieldId(0), 1.into()).unwrap();
                index.add(DocumentId(2), FieldId(0), 2.into()).unwrap();
                index.add(DocumentId(3), FieldId(0), 3.into()).unwrap();
                index.add(DocumentId(4), FieldId(0), 4.into()).unwrap();
                index.add(DocumentId(5), FieldId(0), 2.into()).unwrap();

                let a = $b;

                a(&index);

                index
                    .commit(&CommitConfig {
                        folder_to_commit: generate_new_path(),
                        epoch: 0,
                    })
                    .unwrap();

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
        let index = NumberIndex::try_new(NumberIndexConfig {}).unwrap();

        index.add(DocumentId(0), FieldId(0), 0.into()).unwrap();
        index.add(DocumentId(1), FieldId(0), 1.into()).unwrap();
        index.add(DocumentId(2), FieldId(0), 2.into()).unwrap();
        index.add(DocumentId(3), FieldId(0), 3.into()).unwrap();
        index.add(DocumentId(4), FieldId(0), 4.into()).unwrap();
        index.add(DocumentId(5), FieldId(0), 2.into()).unwrap();

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );

        index
            .commit(&CommitConfig {
                folder_to_commit: generate_new_path(),
                epoch: 0,
            })
            .unwrap();

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
        let index = NumberIndex::try_new(NumberIndexConfig {}).unwrap();

        let iter = (0..1_000).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        for (number, (doc_id, field_id)) in iter {
            index.add(doc_id, field_id, number)?;
        }

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(output, HashSet::from_iter(vec![DocumentId(2)]));

        index.commit(&CommitConfig {
            folder_to_commit: generate_new_path(),
            epoch: 0,
        })?;

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(output, HashSet::from_iter(vec![DocumentId(2)]));

        Ok(())
    }
}
