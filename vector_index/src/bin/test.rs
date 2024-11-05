use embeddings::OramaModels;
use vector_index::{VectorIndex, VectorIndexConfig};
fn main() {
    let config = VectorIndexConfig {
        embeddings_model: OramaModels::GTESmall,
    };

    let idx = vector_index::VectorIndex::new(config);
}
