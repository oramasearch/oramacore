use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use committed::{merge, CommittedNumberFieldIndex};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};
use uncommitted::UncommittedNumberFieldIndex;

use crate::{
    collection_manager::{dto::FieldId, sides::Offset},
    field_id_hashmap::FieldIdHashMap,
    file_utils::{create_if_not_exists, BufferedFile},
    types::DocumentId,
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

    pub fn add(
        &self,
        offset: Offset,
        doc_id: DocumentId,
        field_id: FieldId,
        value: Number,
    ) -> Result<()> {
        debug!(
            "Adding number index: doc_id: {:?}, field_id: {:?}, value: {:?}",
            doc_id, field_id, value
        );

        let uncommitted = self.uncommitted.entry(field_id).or_default();
        uncommitted.insert(offset, value, doc_id)?;

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
        create_if_not_exists(&data_dir).context("Cannot create data directory for number index")?;

        let mut fields = HashMap::new();

        let all_fields = self
            .uncommitted
            .iter()
            .map(|entry| *entry.key())
            .chain(self.committed.iter().map(|entry| *entry.key()))
            .collect::<HashSet<_>>();

        info!("Committing number index: {:?}", all_fields);

        for field_id in all_fields {
            let uncommitted = self.uncommitted.get(&field_id);
            let committed = self.committed.get(&field_id);

            fields.insert(field_id, committed.as_ref().map(|c| c.current_offset()));

            let uncommitted: dashmap::mapref::one::Ref<'_, FieldId, UncommittedNumberFieldIndex> =
                match uncommitted {
                    Some(uncommitted) => uncommitted,
                    None => {
                        info!(?field_id, "No uncommitted data for field");
                        continue;
                    }
                };

            let data = uncommitted.take()?;

            if data.is_empty() {
                info!(?field_id, "No uncommitted data for field");
                continue;
            }

            let current_offset = data.current_offset();
            fields.insert(field_id, Some(current_offset));

            let base_dir = data_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", current_offset.0));

            info!(
                ?field_id,
                committed = committed.is_some(),
                uncommitted_count = data.len(),
                "Committing number index at {base_dir:?}"
            );

            let new_committed_number = if let Some(committed) = committed {
                let merged = merge(data.into_iter(), committed.iter());
                CommittedNumberFieldIndex::from_iter(current_offset, merged, base_dir.clone())
            } else {
                CommittedNumberFieldIndex::from_iter(current_offset, data, base_dir.clone())
            }?;

            new_committed_number.commit(base_dir)?;

            self.committed.insert(field_id, new_committed_number);
        }

        let number_index_info = NumberIndexInfo::V1(NumberIndexInfoV1 {
            field_infos: fields
                .into_iter()
                .filter_map(|(k, v)| v.map(|v| (k, v)))
                .collect(),
        });
        BufferedFile::create_or_overwrite(data_dir.join("info.json"))
            .context("Cannot create info.json")?
            .write_json_data(&number_index_info)
            .context("Cannot serialize into info.json")?;

        Ok(())
    }

    #[instrument(skip(self, data_dir))]
    pub fn load(&mut self, data_dir: PathBuf) -> Result<()> {
        let number_index_info: NumberIndexInfo = BufferedFile::open(data_dir.join("info.json"))
            .context("Cannot open info.json")?
            .read_json_data()
            .context("Cannot deserialize info.json")?;
        let NumberIndexInfo::V1(number_index_info) = number_index_info;

        info!("Loading number index: {:?}", number_index_info.field_infos);

        for (field_id, offset) in number_index_info.field_infos.into_inner() {
            let field_dir = data_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            let committed =
                CommittedNumberFieldIndex::load(offset, field_dir).context("Cannot load field")?;

            self.committed.insert(field_id, committed);
            self.uncommitted
                .insert(field_id, UncommittedNumberFieldIndex::new(offset));
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "version")]
enum NumberIndexInfo {
    #[serde(rename = "1")]
    V1(NumberIndexInfoV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct NumberIndexInfoV1 {
    field_infos: FieldIdHashMap<Offset>,
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

                index
                    .add(Offset(1), DocumentId(0), FieldId(0), 0.into())
                    .unwrap();
                index
                    .add(Offset(2), DocumentId(1), FieldId(0), 1.into())
                    .unwrap();
                index
                    .add(Offset(3), DocumentId(2), FieldId(0), 2.into())
                    .unwrap();
                index
                    .add(Offset(4), DocumentId(3), FieldId(0), 3.into())
                    .unwrap();
                index
                    .add(Offset(5), DocumentId(4), FieldId(0), 4.into())
                    .unwrap();
                index
                    .add(Offset(6), DocumentId(5), FieldId(0), 2.into())
                    .unwrap();

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
        let data_dir = generate_new_path();

        let index = NumberIndex::try_new(NumberIndexConfig {}).unwrap();

        index
            .add(Offset(1), DocumentId(0), FieldId(0), 0.into())
            .unwrap();
        index
            .add(Offset(2), DocumentId(1), FieldId(0), 1.into())
            .unwrap();
        index
            .add(Offset(3), DocumentId(2), FieldId(0), 2.into())
            .unwrap();
        index
            .add(Offset(4), DocumentId(3), FieldId(0), 3.into())
            .unwrap();
        index
            .add(Offset(5), DocumentId(4), FieldId(0), 4.into())
            .unwrap();
        index
            .add(Offset(6), DocumentId(5), FieldId(0), 2.into())
            .unwrap();

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );

        index.commit(data_dir.clone()).unwrap();

        // We can continue to search after commit
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );

        // And insert
        index
            .add(Offset(7), DocumentId(6), FieldId(0), 2.into())
            .unwrap();
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5), DocumentId(6)])
        );

        // Anyway, uncommitted data are lost.
        let mut index = NumberIndex::try_new(NumberIndexConfig {}).unwrap();
        index.load(data_dir.clone()).unwrap();

        // So, after loading, we can still search on old data
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );

        // We can insert data
        index
            .add(Offset(7), DocumentId(6), FieldId(0), 2.into())
            .unwrap();
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5), DocumentId(6)])
        );

        let data_dir = generate_new_path();

        // Commit again
        index.commit(data_dir.clone()).unwrap();

        // The document 6 is still there
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5), DocumentId(6)])
        );

        // Reload again the index
        let mut index = NumberIndex::try_new(NumberIndexConfig {}).unwrap();
        index.load(data_dir.clone()).unwrap();

        // The document 6 is still there
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5), DocumentId(6)])
        );
    }
}
