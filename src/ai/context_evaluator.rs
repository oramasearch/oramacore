use super::OramaModel;
use crate::{
    ai::AIService,
    collection_manager::dto::SearchResult,
    nlp::chunker::{Chunker, ChunkerConfig},
};
use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;
use tiktoken_rs::{cl100k_base, CoreBPE};

struct ContextEvaluator {
    pub ai_service: Arc<AIService>,
    pub bpe: CoreBPE,
}

impl ContextEvaluator {
    pub fn try_new(ai_service: Arc<AIService>) -> Result<Self> {
        let bpe = cl100k_base()?;

        Ok(Self { ai_service, bpe })
    }

    pub async fn evaluate(&self, query: String, context: SearchResult) -> Result<f32> {
        let query_length = self.bpe.encode_with_special_tokens(&query).len();

        // If the input query is composed of more than 50 tokens, we will use a 20% overlap
        let overlap = if query_length < 50 {
            None
        } else {
            Some(query_length * 20 / 100)
        };

        let chunker = Chunker::try_new(ChunkerConfig {
            max_tokens: query_length,
            overlap,
        })?;

        let query_embeddings_list = self
            .ai_service
            .embed_query(OramaModel::MultilingualE5Small, vec![&query])
            .await?;

        let query_embeddings = query_embeddings_list.first();

        if query_embeddings.is_none() {
            return Ok(0.0);
        }

        let hits = context.hits;
        let formatted_hits = hits
            .iter()
            .filter_map(|hit| match &hit.document {
                Some(document) => match self.convert_raw_to_value(document.inner.clone()) {
                    Ok(raw_document) => {
                        let flat_values = self.extract_flat_values(&raw_document);
                        Some(flat_values)
                    }
                    Err(_) => None,
                },
                None => None,
            })
            .collect::<Vec<String>>();

        let chunks = formatted_hits
            .iter()
            .flat_map(|hit| chunker.chunk_text(hit))
            .collect::<Vec<String>>();

        let mut chunks_embeddings = Vec::new();

        for chunk in &chunks {
            let embedding_result = self
                .ai_service
                .embed_query(OramaModel::MultilingualE5Small, vec![chunk])
                .await;

            if let Ok(embeddings) = embedding_result {
                match embeddings.first() {
                    Some(embedding) => chunks_embeddings.push(embedding.clone()),
                    None => {}
                }
            }
        }

        let mut scores = Vec::new();

        for chunk_embedding in &chunks_embeddings {
            let score = self.cosine_similarity(&query_embeddings.unwrap(), &chunk_embedding);
            scores.push(score);
        }

        let average_score = scores.iter().sum::<f32>() / scores.len() as f32;

        Ok(average_score)
    }

    fn extract_flat_values(&self, value: &Value) -> String {
        match value {
            Value::Object(map) => {
                let mut values = Vec::new();
                for val in map.values() {
                    match val {
                        Value::Object(_) | Value::Array(_) => {
                            values.push(self.extract_flat_values(val));
                        }
                        Value::String(s) => values.push(s.clone()),
                        Value::Number(n) => values.push(n.to_string()),
                        Value::Bool(b) => values.push(b.to_string()),
                        Value::Null => values.push("null".to_string()),
                    }
                }
                values.join(" ")
            }
            Value::Array(arr) => {
                let mut values = Vec::new();
                for item in arr {
                    values.push(self.extract_flat_values(item));
                }
                values.join(" ")
            }
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".to_string(),
        }
    }

    fn convert_raw_to_value(
        &self,
        raw: Box<serde_json::value::RawValue>,
    ) -> Result<Value, serde_json::Error> {
        let json_str = raw.get();

        let value: Value = serde_json::from_str(json_str)?;

        Ok(value)
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let (dot_product, magnitude_a, magnitude_b) =
            a.iter()
                .zip(b.iter())
                .fold((0.0, 0.0, 0.0), |(dot, mag_a, mag_b), (&x, &y)| {
                    (
                        dot + x * y,   // Dot product
                        mag_a + x * x, // Squared magnitude of a
                        mag_b + y * y, // Squared magnitude of b
                    )
                });

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a.sqrt() * magnitude_b.sqrt())
    }
}
