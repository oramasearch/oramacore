use embeddings::OramaModels;
use hora::index::hnsw_idx;

pub struct VectorIndex {
    pub embeddings_model: OramaModels,
    pub dimensions: usize,
    pub idx: hnsw_idx::HNSWIndex<f32, usize>,
}

pub struct VectorIndexConfig {
    pub embeddings_model: OramaModels,
}

impl VectorIndex {
    pub fn new(config: VectorIndexConfig) -> Self {
        let dimensions = config.embeddings_model.dimensions();

        let idx = hnsw_idx::HNSWIndex::<f32, usize>::new(
            dimensions,
            &hora::index::hnsw_params::HNSWParams::<f32>::default(),
        );

        Self {
            idx,
            embeddings_model: config.embeddings_model,
            dimensions,
        }
    }
}
