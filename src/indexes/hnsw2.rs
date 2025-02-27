use anyhow::Result;
use hnsw2::{
    core::{
        ann_index::ANNIndex,
        metrics::{real_cosine_similarity, Metric},
        node::{IdxType, Node},
    },
    hnsw_params::HNSWParams,
    HNSWIndex,
};
use serde::{Deserialize, Serialize};

use crate::types::DocumentId;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
struct DocumentIdWrapper(DocumentId);
impl IdxType for DocumentIdWrapper {}
impl Default for DocumentIdWrapper {
    fn default() -> Self {
        panic!("DocumentIdWrapper::default() should not be called");
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HNSW2Index {
    inner: HNSWIndex<f32, DocumentIdWrapper>,
    dim: usize,
}

impl HNSW2Index {
    pub fn new(dim: usize) -> Self {
        let params = &HNSWParams::<f32>::default();
        Self {
            inner: HNSWIndex::new(dim, params),
            dim,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn into_data(self) -> impl Iterator<Item = (DocumentId, Vec<f32>)> {
        self.inner
            .into_data()
            .map(|(v, DocumentIdWrapper(id))| (id, v))
    }

    pub fn add(&mut self, point: &[f32], id: DocumentId) -> Result<()> {
        self.inner
            .add(point, DocumentIdWrapper(id))
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn build(&mut self) -> Result<()> {
        self.inner
            .build(Metric::Euclidean)
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn search(&self, target: Vec<f32>, limit: usize) -> Vec<(DocumentId, f32)> {
        assert_eq!(target.len(), self.dim);

        let v = self.inner.node_search_k(&Node::new(&target), limit);

        let mut result = Vec::new();
        for (node, _) in v {
            let n = node.vectors();

            // The cosine similarity isnt a distance in the math sense
            // https://en.wikipedia.org/wiki/Distance#Mathematical_formalization
            // Anyway, it is good for ranking purposes
            // 1 means the vectors are equal
            // 0 means the vectors are orthogonal
            let score = real_cosine_similarity(n, &target)
                .expect("real_cosine_similarity should not return an error");

            let id = match node.idx() {
                Some(DocumentIdWrapper(id)) => id,
                None => continue,
            };
            result.push((*id, score));
        }

        result
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use rand::distr::{Distribution, Uniform};

    use super::*;

    #[test]
    fn test_hnsw2() {
        let dim = 3;
        let points = [
            vec![255.0, 0.0, 0.0],
            vec![0.0, 255.0, 0.0],
            vec![0.0, 0.0, 255.0],
        ];

        let mut index = HNSW2Index::new(dim);
        for (id, point) in points.iter().enumerate() {
            let id = DocumentId(id as u64);
            index.add(point, id).unwrap();
        }
        index.build().unwrap();

        let target = vec![255.0, 0.0, 0.0];
        let v = index.search(target, 10);

        let res: HashMap<_, _> = v.into_iter().collect();

        assert_eq!(
            res,
            HashMap::from([
                (DocumentId(0), 1.0),
                (DocumentId(1), 0.0),
                (DocumentId(2), 0.0),
            ])
        )
    }

    #[test]
    fn test_hnsw2_serialize_deserialize() {
        let n = 10_000;
        let dimension = 64;

        let normal = Uniform::new(0.0, 10.0).unwrap();
        let samples = (0..n)
            .map(|_| {
                (0..dimension)
                    .map(|_| normal.sample(&mut rand::rng()))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();
        let mut index = HNSW2Index::new(dimension);
        for (i, sample) in samples.into_iter().enumerate() {
            index.add(&sample, DocumentId(i as u64)).unwrap();
        }
        index.build().unwrap();

        let a = bincode::serialize(&index).unwrap();
        let new_index: HNSW2Index = bincode::deserialize(&a).unwrap();

        let target = (0..dimension)
            .map(|_| normal.sample(&mut rand::rng()))
            .collect::<Vec<f32>>();

        let v1 = index.search(target.clone(), 10);
        let v2 = new_index.search(target.clone(), 10);

        assert_eq!(v1, v2);
    }
}
