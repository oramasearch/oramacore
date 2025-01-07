use tonic::Request;

pub mod orama_ai_service {
    tonic::include_proto!("orama_ai_service");
}

use orama_ai_service::{
    calculate_embeddings_service_client::CalculateEmbeddingsServiceClient, EmbeddingRequest,
    OramaIntent, OramaModel,
};
