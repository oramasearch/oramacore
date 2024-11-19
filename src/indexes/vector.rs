use crate::embeddings::OramaModels;
use anyhow::Result;
use hora::core::ann_index::ANNIndex;
use hora::core::metrics::Metric::Manhattan;
use hora::core::node::IdxType;
use hora::index::hnsw_idx;
use serde::Serialize;

use crate::types::DocumentId;

#[derive(Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash)]
struct IdxID(Option<DocumentId>);

pub struct VectorIndex {
    pub embeddings_model: OramaModels,
    pub dimensions: usize,
    idx: hnsw_idx::HNSWIndex<f32, IdxID>,
}

pub struct VectorIndexConfig {
    pub embeddings_model: OramaModels,
}

impl IdxType for IdxID {}

impl VectorIndex {
    pub fn new(config: VectorIndexConfig) -> Self {
        let dimensions = config.embeddings_model.dimensions();

        let idx = hnsw_idx::HNSWIndex::<f32, IdxID>::new(
            dimensions,
            &hora::index::hnsw_params::HNSWParams::<f32>::default(),
        );

        Self {
            idx,
            embeddings_model: config.embeddings_model,
            dimensions,
        }
    }

    pub fn insert(&mut self, id: DocumentId, vector: &[f32]) -> Result<(), &str> {
        self.idx.add(vector, IdxID(Some(id)))?;
        self.idx.build(Manhattan)
    }

    pub fn insert_batch(&mut self, data: Vec<(DocumentId, &[f32])>) -> Result<(), &str> {
        for (id, vector) in data {
            self.idx.add(vector, IdxID(Some(id)))?
        }

        self.idx.build(Manhattan)
    }

    pub fn search(&mut self, target: &[f32], k: usize) -> Vec<DocumentId> {
        self.idx
            .search(target, k)
            .into_iter()
            .map(|id| id.0.unwrap())
            .collect()
    }
}
