use super::OramaModel;
use crate::{
    ai::AIService,
    nlp::chunker::{Chunker, ChunkerConfig},
    types::SearchResult,
};
use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;
use tiktoken_rs::{cl100k_base, CoreBPE};

pub struct ContextEvaluator {
    pub ai_service: Arc<AIService>,
    pub bpe: CoreBPE,
    pub model: OramaModel,
}

impl ContextEvaluator {
    pub fn try_new(ai_service: Arc<AIService>) -> Result<Self> {
        let bpe = cl100k_base()?;
        let model = OramaModel::MultilingualMiniLml12v2;

        Ok(Self {
            ai_service,
            bpe,
            model,
        })
    }

    pub async fn evaluate(&self, query: String, context: SearchResult) -> Result<f32> {
        let query_length = self.bpe.encode_with_special_tokens(&query).len();

        // Use a 30% overlap when the query is > 10 tokens.
        // Otherwise, use a fixed overlap of 5 tokens.
        let overlap = if query_length < 10 {
            Some(5)
        } else {
            Some(query_length * 30 / 100)
        };

        // Use a chunk length of 100 tokens or 2x the query length, whichever is greater.
        let chunk_length = std::cmp::max(100, query_length * 2);

        let chunker = Chunker::try_new(ChunkerConfig {
            max_tokens: chunk_length,
            overlap,
        })?;

        let query_embeddings_list = self
            .ai_service
            .embed_query(self.model, vec![&query])
            .await?;

        let query_embeddings = query_embeddings_list.first();

        if query_embeddings.is_none() {
            return Ok(0.0);
        }

        let hits = context.hits;
        let formatted_hits = hits
            .iter()
            .filter_map(|hit| match &hit.document {
                Some(document) => match self.convert_raw_to_value(&document.inner) {
                    Ok(raw_document) => {
                        let flat_values = Self::extract_flat_values(&raw_document);
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
            let embedding_result = self.ai_service.embed_passage(self.model, vec![chunk]).await;

            if let Ok(embeddings) = embedding_result {
                if let Some(embedding) = embeddings.first() {
                    chunks_embeddings.push(embedding.clone())
                }
            }
        }

        let mut scores = Vec::new();

        for chunk_embedding in &chunks_embeddings {
            let score = self.cosine_similarity(query_embeddings.unwrap(), chunk_embedding);

            let final_score = if self.model == OramaModel::MultilingualE5Small
                || self.model == OramaModel::MultilingualE5Base
                || self.model == OramaModel::MultilingualE5Large
            {
                self.rescale_similarity(0.7, score)
            } else {
                score
            };

            scores.push(self.rescale_similarity(0.0, final_score));
        }

        scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top_n = scores.iter().take(3).copied().collect::<Vec<f32>>();
        let result = if !top_n.is_empty() {
            top_n.iter().sum::<f32>() / top_n.len() as f32
        } else {
            0.0
        };

        Ok(result)
    }

    fn extract_flat_values(value: &Value) -> String {
        match value {
            Value::Object(map) => {
                let mut values = Vec::new();
                for val in map.values() {
                    match val {
                        Value::Object(_) | Value::Array(_) => {
                            values.push(Self::extract_flat_values(val));
                        }
                        Value::String(s) => values.push(s.clone()),
                        Value::Number(n) => values.push(n.to_string()),
                        Value::Bool(_b) => values.push("".to_string()), // Don't add boolean values
                        Value::Null => values.push("".to_string()),     // Don't add null values
                    }
                }
                values.join(" ")
            }
            Value::Array(arr) => {
                let mut values = Vec::new();
                for item in arr {
                    values.push(Self::extract_flat_values(item));
                }
                values.join(" ")
            }
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(_b) => "".to_string(), // Don't add boolean values
            Value::Null => "".to_string(),     // Don't add null values
        }
    }

    fn convert_raw_to_value(
        &self,
        raw: &serde_json::value::RawValue,
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

    // E5-MultiLingual model outputs similarity scores in the range [0.7, 1.0].
    // We rescale this range to [0.0, 1.0] to make it consistent with the rest of the system in case we change the model.
    fn rescale_similarity(&self, base: f32, similarity: f32) -> f32 {
        // Define the original range
        let original_min = base;
        let original_max = 1.0;
        let original_range = original_max - original_min;

        // Define the target range
        let target_min = 0.0;
        let target_max = 1.0;
        let target_range = target_max - target_min;

        // Apply linear transformation:
        // new_value = target_min + (value - original_min) * (target_range / original_range)
        let rescaled = target_min + (similarity - original_min) * (target_range / original_range);

        // threat NaN values as 0.0
        if rescaled.is_nan() {
            return 0.0;
        }
        // Clamp the result to ensure it stays within [0.0, 1.0]
        rescaled.clamp(0.0, 1.0)
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        ai::{AIService, AIServiceConfig},
        tests::utils::create_grpc_server,
        types::{RawJSONDocument, SearchResultHit},
    };
    use http::uri::Scheme;
    use serde_json::value::RawValue;

    use super::ContextEvaluator;

    fn create_context_evaluator() -> ContextEvaluator {
        let ai_service = Arc::new(AIService::new(AIServiceConfig {
            api_key: None,
            host: "0.0.0.0".to_string(),
            port: 50051,
            llm: crate::ai::AIServiceLLMConfig {
                port: 8000,
                host: "0.0.0.0".to_string(),
                model: "".to_string(),
            },
            remote_llms: None,
            max_connections: 1,
            scheme: Scheme::HTTP,
        }));

        ContextEvaluator::try_new(ai_service).unwrap()
    }

    #[tokio::test]
    async fn test_context_evaluator_positive() {
        let address = create_grpc_server().await.unwrap();
        let ai_server_host = address.ip().to_string();
        let ai_server_port = address.port();

        let query = "How do I create a new branch in Git?";

        let context = SearchResult {
            hits: vec![
                SearchResultHit {
                    id: "123".to_string(),
                    score: 0.0,
                    document: Some(
                        Arc::new(RawJSONDocument {
                            id: Some("123".to_string()),
                            inner: RawValue::from_string(
                                    r#"
                                    {
                                        "title": "How to create a new branch in Git?",
                                        "content": "To create a new branch in Git, you can use the git branch command. For example, to create a new branch named my-branch, you would run git branch my-branch. You can then switch to the new branch using git checkout my-branch."
                                    }
                                    "#.to_string()
                                ).unwrap(),
                        })
                    )
                },
                SearchResultHit {
                    id: "456".to_string(),
                    score: 0.0,
                    document: Some(
                        Arc::new(RawJSONDocument {
                            id: Some("456".to_string()),
                            inner: RawValue::from_string(
                                    r#"
                                    {
                                        "title": "Things to know when creating a new branch in Git",
                                        "content": "When creating a new branch in Git, you should be aware of a few things. First, make sure you are in the correct directory where you want to create the branch. Second, give the branch a descriptive name that reflects its purpose. Third, consider the base branch from which you are creating the new branch. Finally, remember to push the new branch to the remote repository if you want to share it with others."
                                    }
                                    "#.to_string()
                                ).unwrap(),
                        })
                    )
                }
            ],
            count: 2,
            facets: None
        };

        let ai_service = Arc::new(super::AIService::new(AIServiceConfig {
            api_key: None,
            host: ai_server_host,
            port: ai_server_port,
            llm: crate::ai::AIServiceLLMConfig {
                port: 8000,
                host: "0.0.0.0".to_string(),
                model: "".to_string(),
            },
            remote_llms: None,
            max_connections: 4,
            scheme: Scheme::HTTP,
        }));

        let context_evaluator = super::ContextEvaluator::try_new(ai_service).unwrap();

        let result = context_evaluator
            .evaluate(query.to_string(), context)
            .await
            .unwrap();

        assert!(result > 0.80);
    }

    #[tokio::test]
    async fn test_context_evaluator_neutral() {
        let address = create_grpc_server().await.unwrap();
        let ai_server_host = address.ip().to_string();
        let ai_server_port = address.port();

        let query = "`git checkout main` gives not fully merged error";

        let context = SearchResult {
            hits: vec![
                SearchResultHit {
                    id: "123".to_string(),
                    score: 0.0,
                    document: Some(
                        Arc::new(RawJSONDocument {
                            id: Some("123".to_string()),
                            inner: RawValue::from_string(
                                    r#"
                                    {
                                        "title": "How to create a new branch in Git?",
                                        "content": "To create a new branch in Git, you can use the git branch command. For example, to create a new branch named my-branch, you would run git branch my-branch. You can then switch to the new branch using git checkout my-branch."
                                    }
                                    "#.to_string()
                                ).unwrap(),
                        })
                    )
                },
                SearchResultHit {
                    id: "456".to_string(),
                    score: 0.0,
                    document: Some(
                        Arc::new(RawJSONDocument {
                            id: Some("456".to_string()),
                            inner: RawValue::from_string(
                                    r#"
                                    {
                                        "title": "Things to know when creating a new branch in Git",
                                        "content": "When creating a new branch in Git, you should be aware of a few things. First, make sure you are in the correct directory where you want to create the branch. Second, give the branch a descriptive name that reflects its purpose. Third, consider the base branch from which you are creating the new branch. Finally, remember to push the new branch to the remote repository if you want to share it with others."
                                    }
                                    "#.to_string()
                                ).unwrap(),
                        })
                    )
                }
            ],
            count: 2,
            facets: None
        };

        let ai_service = Arc::new(super::AIService::new(AIServiceConfig {
            api_key: None,
            host: ai_server_host,
            port: ai_server_port,
            llm: crate::ai::AIServiceLLMConfig {
                port: 8000,
                host: "0.0.0.0".to_string(),
                model: "".to_string(),
            },
            remote_llms: None,
            max_connections: 4,
            scheme: Scheme::HTTP,
        }));

        let context_evaluator = super::ContextEvaluator::try_new(ai_service).unwrap();

        let result = context_evaluator
            .evaluate(query.to_string(), context)
            .await
            .unwrap();

        assert!(result < 0.50);
    }

    #[tokio::test]
    async fn test_context_evaluator_negative() {
        let address = create_grpc_server().await.unwrap();
        let ai_server_host = address.ip().to_string();
        let ai_server_port = address.port();

        let query = "How do I cook pasta with tomato sauce?";

        let context = SearchResult {
            hits: vec![
                SearchResultHit {
                    id: "123".to_string(),
                    score: 0.0,
                    document: Some(
                        Arc::new(RawJSONDocument {
                            id: Some("123".to_string()),
                            inner: RawValue::from_string(
                                    r#"
                                    {
                                        "title": "How to create a new branch in Git?",
                                        "content": "To create a new branch in Git, you can use the git branch command. For example, to create a new branch named my-branch, you would run git branch my-branch. You can then switch to the new branch using git checkout my-branch."
                                    }
                                    "#.to_string()
                                ).unwrap(),
                        })
                    )
                },
                SearchResultHit {
                    id: "456".to_string(),
                    score: 0.0,
                    document: Some(
                        Arc::new(RawJSONDocument {
                            id: Some("456".to_string()),
                            inner: RawValue::from_string(
                                    r#"
                                    {
                                        "title": "Things to know when creating a new branch in Git",
                                        "content": "When creating a new branch in Git, you should be aware of a few things. First, make sure you are in the correct directory where you want to create the branch. Second, give the branch a descriptive name that reflects its purpose. Third, consider the base branch from which you are creating the new branch. Finally, remember to push the new branch to the remote repository if you want to share it with others."
                                    }
                                    "#.to_string()
                                ).unwrap(),
                        })
                    )
                }
            ],
            count: 2,
            facets: None
        };

        let ai_service = Arc::new(super::AIService::new(AIServiceConfig {
            api_key: None,
            host: ai_server_host,
            port: ai_server_port,
            llm: crate::ai::AIServiceLLMConfig {
                port: 8000,
                host: "0.0.0.0".to_string(),
                model: "".to_string(),
            },
            remote_llms: None,
            max_connections: 4,
            scheme: Scheme::HTTP,
        }));

        let context_evaluator = super::ContextEvaluator::try_new(ai_service).unwrap();

        let result = context_evaluator
            .evaluate(query.to_string(), context)
            .await
            .unwrap();

        assert!(result < 0.25);
    }

    #[test]
    fn test_rescale_similarity_exact_min() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, 0.7);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_rescale_similarity_exact_max() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, 1.0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_rescale_similarity_midpoint() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, 0.85);
        // Expected: 0.5, since 0.85 is halfway between 0.7 and 1.0
        assert!(result > 0.49 && result < 0.51);
    }

    #[test]
    fn test_rescale_similarity_quarter_point() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, 0.775);
        // Expected: 0.25, since 0.775 is 1/4 of the way between 0.7 and 1.0
        assert!(result > 0.24 && result < 0.26);
    }

    #[test]
    fn test_rescale_similarity_three_quarter_point() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, 0.925);
        // Expected: 0.75, since 0.925 is 3/4 of the way between 0.7 and 1.0
        assert!(result > 0.74 && result < 0.76);
    }

    #[test]
    fn test_rescale_similarity_below_min() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, 0.6);
        // Should be clamped to 0.0
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_rescale_similarity_above_max() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, 1.1);
        // Should be clamped to 1.0
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_rescale_similarity_negative() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, -0.5);
        // Should be clamped to 0.0
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_rescale_similarity_nan_handling() {
        let evaluator = create_context_evaluator();
        let result = evaluator.rescale_similarity(0.7, f32::NAN);
        // NaN should be converted to 0.0 - you might want to handle this differently
        assert_eq!(result, 0.0);
    }
}
*/
