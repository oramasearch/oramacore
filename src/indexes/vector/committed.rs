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
        let params = HNSWParams::<f32>::default()
            .max_item(1_000_000_000);
        let index = HNSWIndex::<f32, IdxID>::new(dimension, &params);
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
        let search_output = self.index.search_nodes(target, limit);
        if search_output.is_empty() {
            return Ok(());
        }

        for (node, distance) in search_output {

            // `hora` returns the score as Euclidean distance.
            // That means 0.0 is the best score and the larger the score, the worse.
            // NB: because it is a distance, it is always positive.
            // NB2: the score is capped with a maximum value.
            // So, inverting the score could be a good idea.
            // NB3: we capped the score to "100".
            // TODO: put `0.01` number in config.
            let inc = 1.0 / distance.max(0.01);

            let idx = match node.idx() {
                Some(idx) => idx,
                None => {
                    warn!("This should not happen");
                    continue;
                }
            };
            let doc_id = match idx.0 {
                Some(id) => id,
                // ???
                None => {
                    warn!("This should not happen");
                    continue;
                }
            };

            let v = output.entry(doc_id).or_insert(0.0);
            *v += inc;
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
        self.index.build(Metric::Euclidean)
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
    use assert_approx_eq::assert_approx_eq;

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

        let mut output_before = HashMap::new();
        index.search(&[0.0, 0.0, 0.0], 2, &mut output_before)?;

        let deserialized = CommittedVectorFieldIndex::load(data_dir.clone())?;

        let mut output_after = HashMap::new();
        deserialized.search(&[0.0, 0.0, 0.0], 2, &mut output_after)?;

        assert_eq!(output_before, output_after);

        Ok(())
    }

    #[test]
    fn test_score() -> Result<()> {
        let mut index = CommittedVectorFieldIndex::new(3);
        index.insert((DocumentId(1), vec![1.0, 0.0, 0.0]))?;
        index.insert((DocumentId(2), vec![-1.0, 0.0, 0.0]))?;

        index.index.build(Metric::Euclidean)
            .map_err(|e| anyhow!("Cannot build index: {}", e))?;

        let mut output = HashMap::new();
        index.search(&[0.999, 0.001, -0.001], 2, &mut output)?;

        assert!(output[&DocumentId(1)] > output[&DocumentId(2)]);

        let data_dir = generate_new_path();
        index.commit(data_dir.clone())?;

        let new_index = CommittedVectorFieldIndex::load(data_dir)?;

        let mut output_after_load = HashMap::new();
        new_index.search(&[0.999, 0.001, -0.001], 2, &mut output_after_load)?;

        assert_eq!(output, output_after_load);

        Ok(())
    }

    #[test]
    fn test_with_one() -> Result<()> {
        let mut index = CommittedVectorFieldIndex::new(3);
        index.insert((DocumentId(1), vec![1.0, 0.0, 0.0]))?;

        index.index.build(Metric::Euclidean)
            .map_err(|e| anyhow!("Cannot build index: {}", e))?;

        let data_dir = generate_new_path();
        index.commit(data_dir.clone())?;

        let new_index = CommittedVectorFieldIndex::load(data_dir)?;

        let mut output_after_load = HashMap::new();
        new_index.search(&[0.999, 0.001, -1.0], 2, &mut output_after_load)?;

        assert_approx_eq!(output_after_load[&DocumentId(1)], 1.0, 0.01);

        Ok(())
    }
}
