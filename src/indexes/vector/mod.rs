use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use crate::{
    collection_manager::{dto::FieldId, sides::Offset},
    file_utils::BufferedFile,
    types::DocumentId,
};
use anyhow::{anyhow, Context, Result};
use committed::CommittedVectorFieldIndex;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, trace};
use uncommitted::UncommittedVectorFieldIndex;

mod committed;
mod uncommitted;

#[derive(Debug)]
pub struct VectorIndex {
    uncommitted: DashMap<FieldId, UncommittedVectorFieldIndex>,
    committed: DashMap<FieldId, CommittedVectorFieldIndex>,
}

pub struct VectorIndexConfig {}

impl VectorIndex {
    pub fn try_new(_: VectorIndexConfig) -> Result<Self> {
        Ok(Self {
            uncommitted: Default::default(),
            committed: Default::default(),
        })
    }

    pub fn add_field(&self, offset: Offset, field_id: FieldId, dimension: usize) -> Result<()> {
        self.uncommitted
            .entry(field_id)
            .or_insert_with(|| UncommittedVectorFieldIndex::new(dimension, offset));
        self.committed
            .entry(field_id)
            .or_insert_with(|| CommittedVectorFieldIndex::new(dimension, offset));

        Ok(())
    }

    pub fn insert_batch(
        &self,
        offset: Offset,
        data: Vec<(DocumentId, FieldId, Vec<Vec<f32>>)>,
    ) -> Result<()> {
        for (doc_id, field_id, vectors) in data {
            if vectors.is_empty() {
                continue;
            }

            let uncommitted = self
                .uncommitted
                .entry(field_id)
                .or_insert_with(|| UncommittedVectorFieldIndex::new(vectors[0].len(), offset));
            uncommitted.insert(offset, (doc_id, vectors))?;
        }

        Ok(())
    }

    #[instrument(skip(self, data_dir))]
    pub fn commit(&self, data_dir: PathBuf) -> Result<()> {
        std::fs::create_dir_all(&data_dir).context("Cannot create directory for vector index")?;

        let all_field_ids = self
            .uncommitted
            .iter()
            .map(|e| *e.key())
            .chain(self.committed.iter().map(|e| *e.key()))
            .collect::<HashSet<_>>();

        let mut info = VectorIndexInfo {
            fields: self
                .committed
                .iter()
                .map(|e| {
                    let key = *e.key();
                    let offset = e.value().offset();
                    (key, offset)
                })
                .collect(),
        };

        info!("Committing vector index with fields: {:?}", all_field_ids);

        for field_id in all_field_ids {
            let uncommitted = self.uncommitted.get(&field_id);

            let uncommitted = match uncommitted {
                Some(uncommitted) => uncommitted,
                None => {
                    debug!("Empty uncommitted vector index");
                    continue;
                }
            };

            let mut taken = uncommitted.take();

            let offset = taken.current_offset();

            let mut committed = self
                .committed
                .entry(field_id)
                .or_insert_with(|| CommittedVectorFieldIndex::new(taken.dimension, offset));

            for (doc_id, vectors) in taken.data() {
                for (_, vector) in vectors {
                    committed.insert((doc_id, vector))?;
                }
            }

            taken.close();

            let data_dir = data_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            committed.commit(data_dir)?;

            info.fields.insert(field_id, offset);
        }

        BufferedFile::create_or_overwrite(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&info)
            .context("Cannot serialize info.json file")?;

        Ok(())
    }

    #[instrument(skip(self, data_dir))]
    pub fn load(&self, data_dir: PathBuf) -> Result<()> {
        let info: VectorIndexInfo = BufferedFile::open(data_dir.join("info.json"))
            .context("Cannot open info.json file")?
            .read_json_data()
            .context("Cannot deserialize info.json file")?;

        info!(
            "Loading vector index with fields: {:?}",
            info.fields.keys().collect::<Vec<_>>()
        );

        for (field_id, offset) in info.fields {
            let data_dir = data_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            let index = CommittedVectorFieldIndex::load(data_dir, offset)
                .with_context(|| format!("Cannot load vector index for field {:?}", field_id))?;

            self.committed.insert(field_id, index);
        }

        Ok(())
    }

    pub fn search(
        &self,
        field_ids: &Vec<FieldId>,
        target: &[f32],
        committed_limit: usize,
    ) -> Result<HashMap<DocumentId, f32>> {
        trace!(
            "Searching for target: {:?} in fields {:?}",
            target,
            field_ids
        );

        let mut output = HashMap::new();

        for field_id in field_ids {
            let index = self.uncommitted.get(field_id);
            let index = match index {
                Some(index) => index,
                None => continue,
            };

            index
                .search(target, &mut output)
                .context("Cannot perform search on vector index")?;
        }

        for field_id in field_ids {
            let index = self.committed.get(field_id);
            let index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            index
                .search(target, committed_limit, &mut output)
                .context("Cannot perform search on vector index")?;
        }

        trace!("VectorIndex output: {:?}", output);

        Ok(output)
    }
}

#[derive(Serialize, Deserialize)]
struct VectorIndexInfo {
    fields: HashMap<FieldId, Offset>,
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[test]
    fn test_indexes_vector_commit_dont_change_the_result() -> Result<()> {
        const DIM: usize = 3;
        const N: usize = 2;

        let index = VectorIndex::try_new(VectorIndexConfig {})?;
        index.add_field(Offset(0), FieldId(0), 3)?;

        let data = (0..N)
            .map(|i| {
                let doc_id = DocumentId(i as u64);
                let vector: Vec<_> = (0..DIM)
                    .map(|_| {
                        let x = rand::random::<i8>();
                        x as f32 / 10.0
                    })
                    .collect();
                (doc_id, FieldId(0), vec![vector])
            })
            .collect::<Vec<_>>();

        index.insert_batch(Offset(1), data)?;

        let output = index.search(&vec![FieldId(0)], &[1.0, 2.0, 3.0], 5)?;
        let uncommitted_keys = output.keys().cloned().collect::<HashSet<_>>();

        let data_dir = generate_new_path();
        index.commit(data_dir.clone())?;

        let output_after = index.search(&vec![FieldId(0)], &[1.0, 2.0, 3.0], 5)?;
        let committed_keys = output_after.keys().cloned().collect::<HashSet<_>>();

        // Equality is wrong here:
        // HWNSIndex doesn't guarantee the correct results.
        // But the 2 sets should have some common elements at least...
        assert!(!uncommitted_keys.is_disjoint(&committed_keys));

        let new_index = VectorIndex::try_new(VectorIndexConfig {})?;
        new_index.load(data_dir)?;
        let loaded_output = index.search(&vec![FieldId(0)], &[1.0, 2.0, 3.0], 5)?;
        let loaded_keys = loaded_output.keys().cloned().collect::<HashSet<_>>();

        assert_eq!(committed_keys, loaded_keys);

        Ok(())
    }
}
