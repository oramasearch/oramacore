use anyhow::Result;
use vector_quantizer::pq::PQ;

struct Quantizer {
    pq: PQ,
    training_threshold: usize, // The minimum number of elements required to start codebook training
    retrain_threshold: usize, // After adding an X number of documents, we'll need to retrain the codebook
    dimensions: usize,        // Embeddings dimensions. For example, BGE Small is 384 dimensions.
    m: usize,                 // Number of subspaces
    ks: usize,                // Number of centroids per subspace
}

#[derive(Clone)]
struct QuantizerConfig {
    training_threshold: Option<usize>,
    retrain_threshold: Option<usize>,
    dimensions: usize,
    m: Option<usize>,
    ks: Option<usize>,
}

impl Default for QuantizerConfig {
    fn default() -> Self {
        Self {
            training_threshold: Some(50_000),
            retrain_threshold: Some(10_000),
            dimensions: 384,
            m: Some(96),
            ks: Some(256),
        }
    }
}

impl QuantizerConfig {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    pub fn build(self) -> Result<Quantizer> {
        let pq = PQ::try_new(
            self.m.unwrap_or(self.dimensions / 4),
            self.ks.unwrap_or(256) as u32,
        )?;

        Ok(Quantizer {
            pq,
            training_threshold: self.training_threshold.unwrap_or(50_000),
            retrain_threshold: self.retrain_threshold.unwrap_or(10_000),
            dimensions: self.dimensions,
            m: self.m.unwrap_or(self.dimensions / 4),
            ks: self.ks.unwrap_or(256),
        })
    }
}
