use std::collections::{HashMap, HashSet};

use crate::collection_manager::dto::FieldId;
use crate::document_storage::DocumentId;
use anyhow::{anyhow, Result};
use dashmap::{DashMap, Entry};
use hora::core::metrics::Metric::Manhattan;
use hora::core::node::IdxType;
use hora::index::hnsw_idx;
use hora::{core::ann_index::ANNIndex, index::hnsw_idx::HNSWIndex};
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};
use uncommitted::UncommittedVectorIndex;

mod uncommitted;

#[derive(
    Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash, Deserialize,
)]
struct IdxID(Option<DocumentId>);

pub struct VectorIndex {
    uncommitted: UncommittedVectorIndex,
    commited: DashMap<FieldId, HNSWIndex<f32, IdxID>>,
}

pub struct VectorIndexConfig {}

impl IdxType for IdxID {}

impl VectorIndex {
    pub fn try_new(_config: VectorIndexConfig) -> Result<Self> {
        Ok(Self {
            uncommitted: UncommittedVectorIndex::new(),
            commited: Default::default(),
        })
    }

    pub fn add_field(&self, field_id: FieldId, dimension: usize) -> Result<()> {
        let entry = self.commited.entry(field_id);
        match entry {
            Entry::Occupied(_) => Err(anyhow!("Field already exists")),
            Entry::Vacant(entry) => {
                let idx = hnsw_idx::HNSWIndex::<f32, IdxID>::new(
                    dimension,
                    &hora::index::hnsw_params::HNSWParams::<f32>::default(),
                );
                entry.insert(idx);

                Ok(())
            }
        }
    }

    pub fn insert_batch(&self, data: Vec<(DocumentId, FieldId, Vec<Vec<f32>>)>) -> Result<()> {
        for (doc_id, field_id, vectors) in data {
            self.uncommitted.insert((doc_id, field_id, vectors))?;
        }
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        // This process can take time. In the meantime, we cannot:
        // - add new vectors
        // - perform searches
        // And this is not good. We should monitor how much time this process takes
        // and think about how to make it faster or non-blocking.
        // TODO: measure time and think about it.

        let uncommitted = self.uncommitted.consume();

        // Keep track of field_ids to build indexes only for them
        let mut field_ids = HashSet::new();
        for (id, field_id, vectors) in uncommitted {
            let index = self.commited.get_mut(&field_id);
            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            for (_, vector) in vectors {
                index
                    .add(&vector, IdxID(Some(id)))
                    .map_err(|e| anyhow!("Error adding vector: {:?}", e))?;
            }

            field_ids.insert(field_id);
        }

        // Rebuild indexes only for fields that have been updated
        for field_id in field_ids {
            let index = self.commited.get_mut(&field_id);
            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            index
                .build(Manhattan)
                .map_err(|e| anyhow!("Error building index: {:?}", e))?;
        }

        Ok(())
    }

    pub fn search(
        &self,
        field_ids: &Vec<FieldId>,
        target: &[f32],
        limit: usize,
    ) -> Result<HashMap<DocumentId, f32>> {
        trace!(
            "Searching for target: {:?} in fields {:?}",
            target,
            field_ids
        );

        let mut output = self.uncommitted.search(field_ids, target, limit)?;

        for field_id in field_ids {
            let index = self.commited.get(field_id);
            let index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            let search_output = index.search(target, limit).into_iter();
            for id in search_output {
                let doc_id = match id.0 {
                    Some(id) => id,
                    // ???
                    None => {
                        warn!("This should not happen");
                        continue;
                    }
                };
                let v = output.entry(doc_id).or_insert(0.0);
                *v += 1.0;
            }
        }

        trace!("VectorIndex output: {:?}", output);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexes_vector_serialize_deserialize() -> Result<()> {
        let index = VectorIndex::try_new(VectorIndexConfig {})?;
        index.add_field(FieldId(0), 3)?;

        index.insert_batch(vec![
            (DocumentId(0), FieldId(0), vec![vec![1.0, 2.0, 3.0]]),
            (DocumentId(1), FieldId(0), vec![vec![4.0, 5.0, 6.0]]),
            (DocumentId(2), FieldId(0), vec![vec![7.0, 8.0, 9.0]]),
        ])?;

        let output = index.search(&vec![FieldId(0)], &[1.0, 2.0, 3.0], 5)?;

        let keys: HashSet<_> = output.keys().cloned().collect();
        assert_eq!(
            keys,
            HashSet::from_iter([DocumentId(0), DocumentId(1), DocumentId(2)])
        );

        let hnws_index = &*index.commited.get(&FieldId(0)).unwrap();
        let dump = bincode::serialize(&hnws_index)?;

        let index_after = VectorIndex::try_new(VectorIndexConfig {})?;
        index_after.add_field(FieldId(0), 3)?;
        let hnws_index: HNSWIndex<f32, IdxID> = bincode::deserialize(&dump)?;
        index_after.commited.insert(FieldId(0), hnws_index);

        let output_after = index.search(&vec![FieldId(0)], &[1.0, 2.0, 3.0], 5)?;

        assert_eq!(output, output_after);

        Ok(())
    }

    #[test]
    fn test_indexes_vector_commit_dont_change_the_result() -> Result<()> {
        const DIM: usize = 3;
        const N: usize = 1_000;

        let mut index = VectorIndex::try_new(VectorIndexConfig {})?;
        index.add_field(FieldId(0), 3)?;

        let data = (0..N)
            .map(|i| {
                let doc_id = DocumentId(i as u32);
                let vector: Vec<_> = (0..DIM)
                    .map(|_| {
                        let x = rand::random::<i8>();
                        x as f32 / 10.0
                    })
                    .collect();
                (doc_id, FieldId(0), vec![vector])
            })
            .collect::<Vec<_>>();

        index.insert_batch(data)?;

        let output = index.search(&vec![FieldId(0)], &[1.0, 2.0, 3.0], 5)?;
        let keys = output.keys().cloned().collect::<HashSet<_>>();

        index.commit()?;

        let output_after = index.search(&vec![FieldId(0)], &[1.0, 2.0, 3.0], 5)?;
        let keys_after = output_after.keys().cloned().collect::<HashSet<_>>();

        // Equality is wrong here:
        // HWNSIndex doesn't guarantee the correct results.
        // But the 2 sets should have some common elements at least...
        assert!(!keys.is_disjoint(&keys_after));

        Ok(())
    }
}
