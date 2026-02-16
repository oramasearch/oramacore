use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
    str::FromStr,
    sync::Mutex,
};

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::ai::AIServiceConfig;

#[derive(Serialize, Deserialize, Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum Model {
    BGESmall,
    BGEBase,
    BGELarge,
    JinaEmbeddingsV2BaseCode,
    MultilingualE5Small,
    MultilingualE5Base,
    MultilingualE5Large,
    MultilingualMiniLML12V2,
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::BGESmall => write!(f, "BGESmall"),
            Model::BGEBase => write!(f, "BGEBase"),
            Model::BGELarge => write!(f, "BGELarge"),
            Model::JinaEmbeddingsV2BaseCode => write!(f, "JinaEmbeddingsV2BaseCode"),
            Model::MultilingualE5Small => write!(f, "MultilingualE5Small"),
            Model::MultilingualE5Base => write!(f, "MultilingualE5Base"),
            Model::MultilingualE5Large => write!(f, "MultilingualE5Large"),
            Model::MultilingualMiniLML12V2 => write!(f, "MultilingualMiniLML12V2"),
        }
    }
}

impl Model {
    pub fn sequence_length(&self) -> usize {
        match self {
            Model::BGESmall => 512,
            Model::BGEBase => 512,
            Model::BGELarge => 512,
            Model::JinaEmbeddingsV2BaseCode => 512,
            Model::MultilingualE5Small => 512,
            Model::MultilingualE5Base => 512,
            Model::MultilingualE5Large => 512,
            Model::MultilingualMiniLML12V2 => 128,
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            Model::BGESmall => 384,
            Model::BGEBase => 768,
            Model::BGELarge => 1024,
            Model::JinaEmbeddingsV2BaseCode => 768,
            Model::MultilingualE5Small => 384,
            Model::MultilingualE5Base => 768,
            Model::MultilingualE5Large => 1024,
            Model::MultilingualMiniLML12V2 => 384,
        }
    }

    pub fn overlap(&self) -> usize {
        self.sequence_length() * 2 / 100
    }

    // Some models need a rescale because they produce "similar" embeddings.
    #[inline]
    pub fn rescale_score(&self, score: f32) -> f32 {
        let is_e5_model = matches!(
            self,
            Model::MultilingualE5Small | Model::MultilingualE5Base | Model::MultilingualE5Large
        );

        if is_e5_model {
            // For instance, E5 models produce similarity scores that rarely go below
            // 0.7, making the effective range much narrower.
            // So, cosine similarity scores are usually in a narrow range like [0.7, 1.0],
            // instead of the full [0.0, 1.0] range.
            // This rescaling helps normalize the scores to use the full [0.0, 1.0]
            // range for better search ranking.
            const MIN: f32 = 0.7;
            const MAX: f32 = 1.0;
            const DELTA: f32 = MAX - MIN;
            let clamped_score = score.clamp(MIN, MAX);
            (clamped_score - MIN) / DELTA
        } else {
            score
        }
    }

    /// Maps this model to the corresponding fastembed `EmbeddingModel` variant.
    fn as_fastembed_model(&self) -> EmbeddingModel {
        match self {
            Model::BGESmall => EmbeddingModel::BGESmallENV15,
            Model::BGEBase => EmbeddingModel::BGEBaseENV15,
            Model::BGELarge => EmbeddingModel::BGELargeENV15,
            Model::JinaEmbeddingsV2BaseCode => EmbeddingModel::JinaEmbeddingsV2BaseCode,
            Model::MultilingualE5Small => EmbeddingModel::MultilingualE5Small,
            Model::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
            Model::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
            Model::MultilingualMiniLML12V2 => EmbeddingModel::ParaphraseMLMiniLML12V2,
        }
    }

    /// Returns the prefix to prepend to each input text for query intent.
    /// BGE models use a sentence representation prefix, E5 models use "query: ",
    /// and Jina/MiniLM models use no prefix.
    fn query_prefix(&self) -> &'static str {
        match self {
            Model::BGESmall | Model::BGEBase | Model::BGELarge => {
                "Represent this sentence for searching relevant passages: "
            }
            Model::MultilingualE5Small | Model::MultilingualE5Base | Model::MultilingualE5Large => {
                "query: "
            }
            Model::MultilingualMiniLML12V2 | Model::JinaEmbeddingsV2BaseCode => "",
        }
    }

    /// Returns the prefix to prepend to each input text for passage/document intent.
    /// E5 models use "passage: ", all other models use no prefix.
    fn passage_prefix(&self) -> &'static str {
        match self {
            Model::MultilingualE5Small | Model::MultilingualE5Base | Model::MultilingualE5Large => {
                "passage: "
            }
            _ => "",
        }
    }
}

impl FromStr for Model {
    type Err = ();

    fn from_str(model_name: &str) -> Result<Self, Self::Err> {
        match model_name {
            "BGESmall" => Ok(Model::BGESmall),
            "BGEBase" => Ok(Model::BGEBase),
            "BGELarge" => Ok(Model::BGELarge),
            "JinaEmbeddingsV2BaseCode" => Ok(Model::JinaEmbeddingsV2BaseCode),
            "MultilingualE5Small" => Ok(Model::MultilingualE5Small),
            "MultilingualE5Base" => Ok(Model::MultilingualE5Base),
            "MultilingualE5Large" => Ok(Model::MultilingualE5Large),
            "MultilingualMiniLML12V2" => Ok(Model::MultilingualMiniLML12V2),
            _ => Err(()),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum Intent {
    Passage,
    Query,
}

impl Display for Intent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Intent::Passage => write!(f, "passage"),
            Intent::Query => write!(f, "query"),
        }
    }
}

impl FromStr for Intent {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "passage" => Ok(Intent::Passage),
            "query" => Ok(Intent::Query),
            _ => Err(()),
        }
    }
}

/// Pure Rust embeddings service using the fastembed crate (ONNX-based).
/// Replaces the previous Python/PyO3 embeddings layer.
pub struct EmbeddingsService {
    models: Mutex<HashMap<Model, TextEmbedding>>,
    cache_dir: String,
}

impl EmbeddingsService {
    /// Creates a new EmbeddingsService with the given AI config.
    /// Models are loaded lazily on first use unless `dynamically_load_models` is false,
    /// in which case all models in the configured group are preloaded.
    pub fn new(config: AIServiceConfig) -> Result<Self> {
        let cache_dir = config.models_cache_dir.clone();

        // Ensure cache directory exists
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("Cannot create models cache directory: {cache_dir}"))?;

        let mut models_map: HashMap<Model, TextEmbedding> = HashMap::new();

        let embeddings_config = config.embeddings.as_ref();
        let dynamically_load = embeddings_config
            .map(|e| e.dynamically_load_models)
            .unwrap_or(true);

        if !dynamically_load {
            // Preload all models in the configured group
            let group = embeddings_config
                .map(|e| e.default_model_group.as_str())
                .unwrap_or("all");

            let models_to_load = Self::models_for_group(group);
            for model in models_to_load {
                info!(model = %model, "Preloading embedding model");
                let text_embedding = Self::load_model(model, &cache_dir)?;
                models_map.insert(model, text_embedding);
            }
        }

        Ok(Self {
            models: Mutex::new(models_map),
            cache_dir,
        })
    }

    /// Calculates embeddings for the given input texts using the specified model and intent.
    /// Prepends intent-specific prefixes to each input before embedding.
    pub fn calculate_embeddings(
        &self,
        input: Vec<String>,
        intent: Intent,
        model: Model,
    ) -> Result<Vec<Vec<f32>>> {
        // Prepend intent-specific prefix to each input string
        let prefix = match intent {
            Intent::Query => model.query_prefix(),
            Intent::Passage => model.passage_prefix(),
        };

        let prefixed_input: Vec<String> = if prefix.is_empty() {
            input
        } else {
            input.into_iter().map(|s| format!("{prefix}{s}")).collect()
        };

        // Ensure the model is loaded (lazy loading)
        {
            let mut models = self
                .models
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to acquire models lock: {e}"))?;

            if let std::collections::hash_map::Entry::Vacant(e) = models.entry(model) {
                info!(model = %model, "Dynamically loading embedding model");
                let text_embedding = Self::load_model(model, &self.cache_dir)?;
                e.insert(text_embedding);
            }
        }

        // Generate embeddings (release lock during computation for better concurrency)
        let models = self
            .models
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire models lock: {e}"))?;

        let text_embedding = models
            .get(&model)
            .expect("Model should be loaded at this point");

        let documents: Vec<&str> = prefixed_input.iter().map(|s| s.as_str()).collect();
        let embeddings = text_embedding
            .embed(documents, None)
            .with_context(|| format!("Failed to generate embeddings with model {model}"))?;

        Ok(embeddings)
    }

    /// Loads a fastembed TextEmbedding model from cache or downloads it.
    fn load_model(model: Model, cache_dir: &str) -> Result<TextEmbedding> {
        let options = InitOptions::new(model.as_fastembed_model())
            .with_cache_dir(cache_dir.into())
            .with_show_download_progress(true);

        TextEmbedding::try_new(options)
            .with_context(|| format!("Failed to load embedding model: {model}"))
    }

    /// Returns the list of models for a given model group name.
    fn models_for_group(group: &str) -> Vec<Model> {
        match group {
            "en" => vec![Model::BGEBase, Model::BGESmall, Model::BGELarge],
            "multilingual" => vec![
                Model::MultilingualE5Large,
                Model::MultilingualE5Small,
                Model::MultilingualE5Base,
                Model::MultilingualMiniLML12V2,
            ],
            "small" => vec![Model::BGESmall, Model::MultilingualE5Small],
            "code" => vec![Model::JinaEmbeddingsV2BaseCode],
            _ => vec![
                Model::BGESmall,
                Model::BGEBase,
                Model::BGELarge,
                Model::JinaEmbeddingsV2BaseCode,
                Model::MultilingualE5Small,
                Model::MultilingualE5Base,
                Model::MultilingualE5Large,
                Model::MultilingualMiniLML12V2,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::utils::{create_oramacore_config, init_log};
    use std::sync::{Arc, LazyLock};

    use super::*;

    static EMBEDDINGS_SERVICE: LazyLock<Arc<EmbeddingsService>> = LazyLock::new(|| {
        init_log();

        let config = create_oramacore_config();
        Arc::new(
            EmbeddingsService::new(config.ai_server)
                .expect("Failed to initialize EmbeddingsService for tests"),
        )
    });

    fn get_embeddings_service() -> Arc<EmbeddingsService> {
        EMBEDDINGS_SERVICE.clone()
    }

    #[test]
    fn test_calculate_embeddings_with_single_string() -> Result<()> {
        let embeddings = get_embeddings_service();

        let result = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            Intent::Passage,
            Model::BGESmall,
        )?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);

        Ok(())
    }

    #[test]
    fn test_calculate_embeddings_with_multiple_strings() -> Result<()> {
        let embeddings = get_embeddings_service();

        let result = embeddings.calculate_embeddings(
            vec![
                "Hello world".to_string(),
                "The quick brown fox jumps over the lazy dog".to_string(),
            ],
            Intent::Passage,
            Model::BGESmall,
        )?;

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 384);
        assert_eq!(result[1].len(), 384);

        Ok(())
    }

    #[test]
    fn test_calculate_embeddings_with_multiple_models() -> Result<()> {
        let embeddings = get_embeddings_service();

        let result1 = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            Intent::Passage,
            Model::BGESmall,
        )?;

        let result2 = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            Intent::Passage,
            Model::JinaEmbeddingsV2BaseCode,
        )?;

        assert_eq!(result1.len(), 1);
        assert_eq!(result1[0].len(), 384);

        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].len(), 768);

        Ok(())
    }

    #[test]
    fn test_calculate_embeddings_with_different_intent() -> Result<()> {
        let embeddings = get_embeddings_service();

        let result1 = embeddings.calculate_embeddings(
            vec!["The process of photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight energy, primarily occurring in chloroplasts through light-dependent and light-independent reactions.".to_string()],
            Intent::Passage,
            Model::MultilingualE5Small,
        )?;

        let result2 = embeddings.calculate_embeddings(
            vec!["The process of photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight energy, primarily occurring in chloroplasts through light-dependent and light-independent reactions.".to_string()],
            Intent::Query,
            Model::MultilingualE5Small,
        )?;

        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);
        assert_ne!(result1, result2);

        Ok(())
    }
}
