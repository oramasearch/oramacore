use std::{
    collections::HashMap,
    fmt::{self, Debug},
    path::PathBuf,
};

use anyhow::{anyhow, Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::Deserialize;

pub struct FastEmbedModel {
    name: String,
    model: TextEmbedding,
    dimensions: usize,
}
impl FastEmbedModel {
    pub fn model_name(&self) -> String {
        self.name.clone()
    }
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn embed(&self, input: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        self.model.embed(input, None)
    }
}
impl Debug for FastEmbedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FastEmbedModel({})", self.name)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FastEmbedModelRepoConfig {
    pub real_model_name: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FastEmbedRepoConfig {
    cache_dir: PathBuf,
}

#[derive(Debug)]
pub struct FastEmbedRepo {
    fast_embed_config: FastEmbedRepoConfig,
    model_configs: HashMap<String, FastEmbedModelRepoConfig>,
}

impl FastEmbedRepo {
    pub fn new(
        fast_embed_config: FastEmbedRepoConfig,
        model_configs: HashMap<String, FastEmbedModelRepoConfig>,
    ) -> Self {
        Self {
            fast_embed_config,
            model_configs,
        }
    }

    pub async fn load_model(&self, model_name: String) -> Result<FastEmbedModel> {
        let model_repo_config = self
            .model_configs
            .get(&model_name)
            .ok_or_else(|| anyhow!("Model not found: {}", model_name))?;

        let embedding_model = match model_repo_config.real_model_name.as_str() {
            "Xenova/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "Xenova/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "Xenova/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
            "intfloat/multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
            "intfloat/multilingual-e5-base" => EmbeddingModel::MultilingualE5Base,
            "Qdrant/multilingual-e5-large-onnx" => EmbeddingModel::MultilingualE5Large,
            _ => return Err(anyhow!("Unknown model name: {model_name}")),
        };

        let init_option = InitOptions::new(embedding_model.clone())
            .with_cache_dir(self.fast_embed_config.cache_dir.clone())
            .with_show_download_progress(false);

        let text_embedding = TextEmbedding::try_new(init_option)
            .with_context(|| format!("Failed to initialize the Fastembed: {embedding_model}"))?;

        let model = FastEmbedModel {
            name: model_name,
            model: text_embedding,
            dimensions: model_repo_config.dimensions,
        };

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[tokio::test]
    async fn test_embedding_run_fastembed() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let tmp = tempdir::TempDir::new("test_fe_download_onnx")?;
        let cache_path: PathBuf = tmp.path().into();
        std::fs::remove_dir(cache_path.clone())?;

        let rebranded_name = "my-model".to_string();

        let fast_embed_config = FastEmbedRepoConfig {
            cache_dir: generate_new_path(),
        };

        let repo = FastEmbedRepo::new(
            fast_embed_config,
            HashMap::from_iter([(
                rebranded_name.clone(),
                FastEmbedModelRepoConfig {
                    real_model_name: "Xenova/bge-small-en-v1.5".to_string(),
                    dimensions: 384,
                },
            )]),
        );
        let model = repo
            .load_model(rebranded_name)
            .await
            .expect("Failed to cache model");

        let output = model.embed(vec![&"foo".to_string()])?;

        assert_eq!(output[0].len(), 384);

        Ok(())
    }
}
