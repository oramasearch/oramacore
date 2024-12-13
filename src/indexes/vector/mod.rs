use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use crate::collection_manager::dto::FieldId;
use crate::document_storage::DocumentId;
use anyhow::{anyhow, Context, Result};
use committed::CommittedState;
use dashmap::{DashMap, Entry};
use tracing::trace;
use uncommitted::UncommittedVectorIndex;

mod committed;
mod uncommitted;

pub struct VectorIndex {
    uncommitted: UncommittedVectorIndex,
    commited: DashMap<FieldId, CommittedState>,
    base_path: PathBuf,
}

pub struct VectorIndexConfig {
    pub base_path: PathBuf,
}

impl VectorIndex {
    pub fn try_new(config: VectorIndexConfig) -> Result<Self> {
        Ok(Self {
            uncommitted: UncommittedVectorIndex::new(),
            commited: Default::default(),
            base_path: config.base_path,
        })
    }

    pub fn add_field(&self, field_id: FieldId, dimension: usize) -> Result<()> {
        let entry = self.commited.entry(field_id);
        match entry {
            Entry::Occupied(_) => Err(anyhow!("Field already exists")),
            Entry::Vacant(entry) => {
                entry.insert(CommittedState::new_in_memory(dimension));

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
        let mut loaded_index_field_ids = HashSet::new();
        for (id, field_id, vectors) in uncommitted {
            let index = self.commited.get_mut(&field_id);

            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            // `load_in_memory` does nothing if it is already loaded.
            let loaded_index = index.load_in_memory()?;
            for (_, vector) in vectors {
                loaded_index.add(&vector, id)?;
            }

            loaded_index_field_ids.insert(field_id);
        }

        // Rebuild indexes only for fields that have been updated
        for field_id in loaded_index_field_ids {
            let index = self.commited.get_mut(&field_id);

            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            let loaded_index = index.load_in_memory()?;
            loaded_index.build()?;

            loaded_index.save_on_fs(self.base_path.join(field_id.0.to_string()))?;
        }

        Ok(())
    }

    pub fn load_in_memory(&self, field_ids: Option<Vec<FieldId>>) -> Result<()> {
        let field_ids = match field_ids {
            Some(field_ids) => field_ids,
            None => self.commited.iter().map(|e| *e.key()).collect(),
        };

        for field_id in field_ids {
            let index = self.commited.get_mut(&field_id);

            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            // We don't use the index itself,
            // but we need to call `load_in_memory` to load the index in memory
            let _ = index.load_in_memory();
        }

        Ok(())
    }

    pub fn unload_from_memory(&self, field_ids: Option<Vec<FieldId>>) -> Result<()> {
        let field_ids = match field_ids {
            Some(field_ids) => field_ids,
            None => self.commited.iter().map(|e| *e.key()).collect(),
        };

        for field_id in field_ids {
            let index = self.commited.get_mut(&field_id);

            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            index.unload_from_memory(self.base_path.join(field_id.0.to_string()))?;
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

            index
                .search(target, limit, &mut output)
                .context("Cannot perform search on vector index")?;
        }

        trace!("VectorIndex output: {:?}", output);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[test]
    fn test_indexes_vector_commit_dont_change_the_result() -> Result<()> {
        const DIM: usize = 3;
        const N: usize = 1_000;

        let mut index = VectorIndex::try_new(VectorIndexConfig {
            base_path: generate_new_path(),
        })?;
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
