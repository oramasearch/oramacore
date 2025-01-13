use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use committed::{merge, CommittedNumberFieldIndex};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};
use uncommitted::UncommittedNumberFieldIndex;

use crate::{collection_manager::dto::FieldId, file_utils::BufferedFile, types::DocumentId};

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

    #[instrument(skip(self, data_dir))]
    pub fn commit(&self, data_dir: PathBuf) -> Result<()> {
        std::fs::create_dir_all(&data_dir)
            .context("Cannot create data directory for number index")?;

        let all_fields = self
            .uncommitted
            .iter()
            .map(|entry| *entry.key())
            .chain(self.committed.iter().map(|entry| *entry.key()))
            .collect::<HashSet<_>>();

        info!("Committing number index: {:?}", all_fields);

        BufferedFile::create(data_dir.join("info.json"))
            .context("Cannot create info.json")?
            .write_json_data(&all_fields)
            .context("Cannot serialize into info.json")?;

        for field_id in all_fields {
            let uncommitted = self.uncommitted.get(&field_id);
            let uncommitted = match uncommitted {
                Some(uncommitted) => uncommitted,
                None => {
                    debug!(?field_id, "No uncommitted data for field");
                    continue;
                }
            };

            let data = uncommitted.take()?;
            let committed = self.committed.get(&field_id);

            let base_dir = data_dir.join(format!("{}", field_id.0));

            info!(
                ?field_id,
                committed = committed.is_some(),
                uncommitted_count = data.len(),
                "Committing number index at {base_dir:?}"
            );

            let new_committed_number = if let Some(committed) = committed {
                let merged = merge(data.into_iter(), committed.iter());
                CommittedNumberFieldIndex::from_iter(merged, base_dir.clone())
            } else {
                CommittedNumberFieldIndex::from_iter(data, base_dir.clone())
            }?;

            new_committed_number.commit(base_dir)?;

            self.committed.insert(field_id, new_committed_number);
        }

        Ok(())
    }

    #[instrument(skip(self, data_dir))]
    pub fn load(&mut self, data_dir: PathBuf) -> Result<()> {
        let field_ids: HashSet<FieldId> = BufferedFile::open(data_dir.join("info.json"))
            .context("Cannot open info.json")?
            .read_json_data()
            .context("Cannot deserialize info.json")?;

        info!("Loading number index: {:?}", field_ids);

        for field_id in field_ids {
            let field_dir = data_dir.join(format!("{}", field_id.0));
            let committed =
                CommittedNumberFieldIndex::load(field_dir).context("Cannot load field")?;

            self.committed.insert(field_id, committed);
        }

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

                index.commit(generate_new_path()).unwrap();

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

        index.commit(generate_new_path()).unwrap();

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

        let iter = (0..1_000).map(|i| (Number::from(i), (DocumentId(i as u64), FieldId(0))));
        for (number, (doc_id, field_id)) in iter {
            index.add(doc_id, field_id, number)?;
        }

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(output, HashSet::from_iter(vec![DocumentId(2)]));

        index.commit(generate_new_path())?;

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(output, HashSet::from_iter(vec![DocumentId(2)]));

        Ok(())
    }

    #[test]
    fn test_indexes_number_commit_load() -> Result<()> {
        let data: Vec<(_, HashSet<_>)> = vec![
            (Number::I32(0), HashSet::from_iter([DocumentId(0)])),
            (Number::I32(1), HashSet::from_iter([DocumentId(1)])),
            (Number::I32(2), HashSet::from_iter([DocumentId(2)])),
            (Number::I32(3), HashSet::from_iter([DocumentId(3)])),
            (Number::I32(4), HashSet::from_iter([DocumentId(4)])),
            (Number::I32(5), HashSet::from_iter([DocumentId(5)])),
            (Number::I32(6), HashSet::from_iter([DocumentId(6)])),
            (Number::I32(7), HashSet::from_iter([DocumentId(7)])),
            (Number::I32(8), HashSet::from_iter([DocumentId(8)])),
            (Number::I32(9), HashSet::from_iter([DocumentId(9)])),
        ];
        let field_id = FieldId(0);

        let index = NumberIndex::try_new(NumberIndexConfig {})?;
        for d in data {
            for doc_id in d.1 {
                index.add(doc_id, field_id, d.0)?;
            }
        }

        // Query on uncommitted data
        let output = index.filter(
            field_id,
            NumberFilter::Between((Number::I32(-1), Number::I32(10))),
        )?;
        assert_eq!(
            output,
            HashSet::from_iter((0..10).map(|i| DocumentId(i as u64)))
        );

        let data_dir = generate_new_path();
        index.commit(data_dir.clone())?;

        let mut after = NumberIndex::try_new(NumberIndexConfig {})?;
        after.load(data_dir)?;

        // Query on committed data
        let output = after.filter(
            field_id,
            NumberFilter::Between((Number::I32(-1), Number::I32(10))),
        )?;
        assert_eq!(
            output,
            HashSet::from_iter((0..10).map(|i| DocumentId(i as u64)))
        );

        after.add(DocumentId(10), field_id, Number::I32(10))?;
        after.add(DocumentId(11), field_id, Number::I32(11))?;

        // Query on committed & uncommitted data
        let output = after.filter(
            field_id,
            NumberFilter::Between((Number::I32(-1), Number::I32(12))),
        )?;
        assert_eq!(
            output,
            HashSet::from_iter((0..12).map(|i| DocumentId(i as u64)))
        );

        let new_data_dir = generate_new_path();
        after.commit(new_data_dir.clone())?;

        let mut after = NumberIndex::try_new(NumberIndexConfig {})?;
        after.load(new_data_dir)?;

        // Query on committed (and merged) data
        let output = after.filter(
            field_id,
            NumberFilter::Between((Number::I32(-1), Number::I32(12))),
        )?;
        assert_eq!(
            output,
            HashSet::from_iter((0..12).map(|i| DocumentId(i as u64)))
        );

        Ok(())
    }
}
