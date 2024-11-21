// pub mod custom_models;
pub mod pq;
pub mod properties_selector;

mod hf;

use anyhow::{anyhow, Context, Result};
use dashmap::{DashMap, Entry};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use hf::HuggingFaceConfiguration;
use serde::{Deserialize, Serialize};
use std::{fmt, sync::Arc};
use strum::EnumIter;
use strum_macros::{AsRefStr, Display};
use tracing::{info, instrument};

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingPreload {
    Bool(bool),
    List(Vec<OramaModel>),
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub preload: EmbeddingPreload,
    pub cache_path: String,
    pub hugging_face: Option<HuggingFaceConfiguration>,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct EmbeddingsParams {
    model: OramaModel,
    intent: EncodingIntent,
    input: Vec<String>,
}

#[derive(Deserialize, Debug, Copy, Clone)]
#[serde(rename_all = "lowercase")]
pub enum EncodingIntent {
    Query,
    Passage,
}

#[derive(Serialize)]
pub struct EmbeddingsResponse {
    dimensions: i32,
    embeddings: Vec<Vec<f32>>,
}

#[derive(
    Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, EnumIter, Display, AsRefStr,
)]
pub enum OramaFastembedModel {
    #[serde(rename = "gte-small")]
    #[strum(serialize = "gte-small")]
    GTESmall,
    #[serde(rename = "gte-base")]
    #[strum(serialize = "gte-base")]
    GTEBase,
    #[serde(rename = "gte-large")]
    #[strum(serialize = "gte-large")]
    GTELarge,
    #[serde(rename = "multilingual-e5-small")]
    #[strum(serialize = "multilingual-e5-small")]
    MultilingualE5Small,
    #[serde(rename = "multilingual-e5-base")]
    #[strum(serialize = "multilingual-e5-base")]
    MultilingualE5Base,
    #[serde(rename = "multilingual-e5-large")]
    #[strum(serialize = "multilingual-e5-large")]
    MultilingualE5Large,
}

impl OramaFastembedModel {
    pub fn dimensions(&self) -> usize {
        match self {
            OramaFastembedModel::GTESmall => 384,
            OramaFastembedModel::GTEBase => 768,
            OramaFastembedModel::GTELarge => 1024,
            OramaFastembedModel::MultilingualE5Small => 384,
            OramaFastembedModel::MultilingualE5Base => 768,
            OramaFastembedModel::MultilingualE5Large => 1024,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, Display, AsRefStr)]
#[serde(untagged)]
pub enum OramaModel {
    Fastembed(OramaFastembedModel),
    HuggingFace(String),
}

impl fmt::Display for EncodingIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodingIntent::Query => write!(f, "query"),
            EncodingIntent::Passage => write!(f, "passage"),
        }
    }
}

pub struct LoadedModel {
    text_embedding: TextEmbedding,
    model_name: String,
    max_input_tokens: usize,
    dimensions: usize,
}

impl LoadedModel {
    fn new(
        text_embedding: TextEmbedding,
        model_name: String,
        max_input_tokens: usize,
        dimensions: usize,
    ) -> Self {
        Self {
            text_embedding,
            model_name,
            max_input_tokens,
            dimensions,
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn embed(&self, input: Vec<String>, batch_size: Option<usize>) -> Result<Vec<Vec<f32>>> {
        // The following "clone" is ugly: we are cloing every input string
        // Unfortunatelly, `embed` method required `Vec` and not `&Vec`.
        // So, we can't avoid cloning here.
        // Anyway, `embed` takes `Vec<&AsRef<str>>`
        // so, in some how, we can avoid cloning.
        // TODO: avoid cloning
        self.text_embedding
            .embed(input.clone(), batch_size)
            .with_context(|| {
                format!(
                    "Model \"{}\" fails to calculate the embed for input: {:?}",
                    self.model_name, input
                )
            })
    }
}

pub struct EmbeddingService {
    loaded_models: DashMap<OramaModel, Arc<LoadedModel>>,
    builder: EmbeddingBuilder,
}

impl EmbeddingService {
    pub async fn try_new(config: EmbeddingConfig) -> Result<Self> {
        let builder = EmbeddingBuilder::try_new(config.clone())?;

        let s = Self {
            loaded_models: DashMap::new(),
            builder,
        };

        match config.preload {
            EmbeddingPreload::Bool(true) => {
                unimplemented!("Preloading \"true\" is not implemented yet");
            }
            EmbeddingPreload::Bool(false) => {
                // Do nothing
            }
            EmbeddingPreload::List(models) => {
                for model in models {
                    s.get_model(model).await?;
                }
            }
        }

        Ok(s)
    }

    pub async fn get_model(&self, model: OramaModel) -> Result<Arc<LoadedModel>> {
        let loaded_model = self.loaded_models.entry(model.clone());
        match loaded_model {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let loaded_model = self.builder.try_get(model).await?;
                let loaded_model = Arc::new(loaded_model);
                entry.insert(loaded_model.clone());
                Ok(loaded_model)
            }
        }
    }

    pub fn max_input_tokens(&self, model: OramaModel) -> Result<usize> {
        let loaded_model = self.loaded_models.get(&model);
        match loaded_model {
            Some(model) => Ok(model.max_input_tokens),
            None => Err(anyhow!(
                "Model not found in the loaded models. Try to load the model first"
            )),
        }
    }

    pub fn dimensions(&self, model: OramaModel) -> Result<usize> {
        let loaded_model = self.loaded_models.get(&model);
        match loaded_model {
            Some(model) => Ok(model.dimensions),
            None => Err(anyhow!(
                "Model not found in the loaded models. Try to load the model first"
            )),
        }
    }

    /*
    pub fn normalize_input(&self, model: OramaModel, intent: EncodingIntent, input: Vec<String>) -> Vec<String> {
        match self {
            OramaModel::Fastembed(OramaFastembedModel::MultilingualE5Small)
            | OramaModel::Fastembed(OramaFastembedModel::MultilingualE5Base)
            | OramaModel::Fastembed(OramaFastembedModel::MultilingualE5Large) => input
                .into_iter()
                .map(|text| format!("{intent}: {text}"))
                .collect(),
            _ => input,
        }
    }
    */

    pub async fn embed(
        &self,
        model: OramaModel,
        input: Vec<String>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let loaded_model = self.get_model(model).await?;
        loaded_model.embed(input, batch_size)
    }
}

pub struct EmbeddingBuilder {
    config: EmbeddingConfig,
}

impl EmbeddingBuilder {
    fn try_new(config: EmbeddingConfig) -> Result<Self> {
        let builder = Self {
            config: config.clone(),
        };

        Ok(builder)
    }

    #[instrument(skip(self), fields(orama_model = ?orama_model))]
    async fn try_get(&self, orama_model: OramaModel) -> Result<LoadedModel> {
        info!("Loading model");
        match orama_model {
            OramaModel::Fastembed(orama_embedding_model) => {
                let embedding_model = match orama_embedding_model {
                    OramaFastembedModel::GTESmall => EmbeddingModel::BGESmallENV15,
                    OramaFastembedModel::GTEBase => EmbeddingModel::BGEBaseENV15,
                    OramaFastembedModel::GTELarge => EmbeddingModel::BGELargeENV15,
                    OramaFastembedModel::MultilingualE5Small => EmbeddingModel::MultilingualE5Small,
                    OramaFastembedModel::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
                    OramaFastembedModel::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
                };

                let text_embedding = TextEmbedding::try_new(
                    InitOptions::new(embedding_model)
                        .with_show_download_progress(false)
                        .with_cache_dir(self.config.cache_path.clone().into()),
                )
                .with_context(|| {
                    format!("Failed to initialize the Fastembed: {orama_embedding_model}")
                })?;
                Ok(LoadedModel::new(
                    text_embedding,
                    orama_embedding_model.to_string(),
                    512,
                    orama_embedding_model.dimensions(),
                ))
            }
            OramaModel::HuggingFace(model_name) => {
                let hugging_face_config = self
                    .config
                    .hugging_face
                    .as_ref()
                    .ok_or_else(|| anyhow!("HuggingFace configuration is missing"))?;
                let hf_model = LoadedModel::try_from_hugging_face(
                    hugging_face_config,
                    self.config.cache_path.clone(),
                    model_name.clone(),
                )
                .await
                .with_context(move || format!("Failed to load HuggingFace model {model_name}"))?;
                Ok(hf_model)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embeddings_calculate_embedding() {
        let _ = tracing_subscriber::fmt::try_init();

        // We don't want to download the model every time we run the test
        // So we use a standard temp directory, without any cleanup logic
        let temp_dir = std::env::temp_dir().to_str().unwrap().to_string();

        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            cache_path: temp_dir.clone(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        })
        .await
        .expect("Failed to initialize the EmbeddingService");

        let output = embedding_service
            .embed(
                OramaModel::Fastembed(OramaFastembedModel::GTESmall),
                vec!["Hello, world!".to_string()],
                Some(1),
            )
            .await
            .expect("Failed to embed text");

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), OramaFastembedModel::GTESmall.dimensions());

        assert_eq!(embedding_service.loaded_models.len(), 1);
    }
}
