use anyhow::Result;
use embeddings::OramaModels;
use hora::core::ann_index::ANNIndex;
use hora::core::metrics::Metric::Euclidean;
use hora::index::hnsw_idx;

pub struct VectorIndex {
    pub embeddings_model: OramaModels,
    pub dimensions: usize,
    pub idx: hnsw_idx::HNSWIndex<f32, String>,
}

pub struct VectorIndexConfig {
    pub embeddings_model: OramaModels,
}

impl VectorIndex {
    pub fn new(config: VectorIndexConfig) -> Self {
        let dimensions = config.embeddings_model.dimensions();

        let idx = hnsw_idx::HNSWIndex::<f32, String>::new(
            dimensions,
            &hora::index::hnsw_params::HNSWParams::<f32>::default(),
        );

        Self {
            idx,
            embeddings_model: config.embeddings_model,
            dimensions,
        }
    }

    pub fn insert(&mut self, id: String, vector: &[f32]) -> Result<(), &str> {
        self.idx.add(vector, id)?;
        self.idx.build(Euclidean)
    }

    pub fn insert_batch(&mut self, data: Vec<(String, &[f32])>) -> Result<(), &str> {
        for (id, vector) in data.iter() {
            self.idx.add(vector, id.clone())?
        }

        self.idx.build(Euclidean)
    }

    pub fn search(&mut self, target: &[f32], k: usize) -> Vec<String> {
        self.idx.search(target, k)
    }
}
