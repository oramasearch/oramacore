use std::{collections::HashMap, path::PathBuf};

use anyhow::{anyhow, Result};
use hora::{
    core::{
        ann_index::{ANNIndex, SerializableIndex},
        metrics::Metric,
        node::IdxType,
    },
    index::{hnsw_idx::HNSWIndex, hnsw_params::HNSWParams},
};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::types::DocumentId;

#[derive(
    Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash, Deserialize,
)]
pub struct IdxID(Option<DocumentId>);
impl IdxType for IdxID {}

#[derive(Debug)]
pub struct CommittedVectorFieldIndex {
    index: HNSWIndex::<f32, IdxID>,
}

impl CommittedVectorFieldIndex {
    pub fn new(dimension: usize) -> Self {
        let index = HNSWIndex::<f32, IdxID>::new(dimension, &HNSWParams::<f32>::default());
        Self {
            index,
        }
    }

    pub fn search(
        &self,
        target: &[f32],
        limit: usize,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        let search_output = self.index.search(target, limit).into_iter();
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

        Ok(())
    }

    pub fn insert(&mut self, data: (DocumentId, Vec<f32>)) -> Result<()> {
        let (doc_id, vector) = data;
        self.index.add(&vector, IdxID(Some(doc_id)))
            .map_err(|e| anyhow!("Cannot add vector to index: {}", e))?;

        Ok(())
    }

    pub fn load(data_dir: PathBuf) -> Result<Self> {
        let data_dir = match data_dir.to_str() {
            Some(data_dir) => data_dir,
            None => {
                return Err(anyhow!("Cannot convert path to string"));
            }
        };

        let index = HNSWIndex::<f32, IdxID>::load(data_dir)
            .map_err(|e| anyhow!("Cannot load index: {}", e))?;
        Ok(Self {
            index,
        })
    }

    pub fn commit(&mut self, data_dir: PathBuf) -> Result<()> {
        self.index.build(Metric::Manhattan)
            .map_err(|e| anyhow!("Cannot build index: {}", e))?;

        let data_dir = match data_dir.to_str() {
            Some(data_dir) => data_dir,
            None => {
                return Err(anyhow!("Cannot convert path to string"));
            }
        };
        self.index.dump(data_dir)
            .map_err(|e| anyhow!("Cannot dump index: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[test]
    fn test_serialize_deserialize_committed_state() -> Result<()> {
        const DIM: usize = 3;
        const N: usize = 10;

        let data_dir = generate_new_path();

        
        let index = {
            let mut index = CommittedVectorFieldIndex::new(3);
            let data = (0..N)
                .map(|i| {
                    let doc_id = DocumentId(i as u64);
                    let vector: Vec<_> = (0..DIM)
                        .map(|_| {
                            let x = rand::random::<i8>();
                            x as f32 / 10.0
                        })
                        .collect();
                    (vector, doc_id)
                })
                .collect::<Vec<_>>();

            for (vector, doc_id) in data {
                index.insert((doc_id, vector))?;
            }
            index.commit(data_dir.clone())?;

            index
        };

        let deserialized = CommittedVectorFieldIndex::load(data_dir.clone())?;

        let mut output_before = HashMap::new();
        index.search(&[0.0, 0.0, 0.0], 2, &mut output_before)?;

        let mut output_after = HashMap::new();
        deserialized.search(&[0.0, 0.0, 0.0], 2, &mut output_after)?;

        assert_eq!(output_before, output_after);

        Ok(())
    }
}
