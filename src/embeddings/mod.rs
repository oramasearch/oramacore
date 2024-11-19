pub mod custom_models;
pub mod pq;
pub mod properties_selector;

use custom_models::{CustomModel, ModelFileConfig};
use anyhow::{anyhow, Context, Result};
use fastembed::{EmbeddingModel, InitOptions, InitOptionsUserDefined, TextEmbedding};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use strum::EnumIter;
use strum_macros::{AsRefStr, Display};

static MODELS: OnceCell<RwLock<HashMap<OramaModels, Arc<TextEmbedding>>>> = OnceCell::new();

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct EmbeddingsParams {
    model: OramaModels,
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

#[derive(Deserialize, Debug, Hash, PartialEq, Eq, Copy, Clone, EnumIter, Display, AsRefStr)]
pub enum OramaModels {
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
    #[serde(rename = "jinaai/jina-embeddings-v2-base-code")]
    #[strum(serialize = "jinaai/jina-embeddings-v2-base-code")]
    JinaV2BaseCode,
}

pub struct LoadedModels(HashMap<OramaModels, TextEmbedding>);

impl LoadedModels {
    pub fn embed(
        &self,
        model: OramaModels,
        input: Vec<String>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let text_embedding = match self.0.get(&model) {
            Some(model) => model,
            None => return Err(anyhow!("Unable to retrieve embedding model: {model:?}")),
        };

        text_embedding.embed(input, batch_size)
    }
}

impl TryInto<EmbeddingModel> for OramaModels {
    type Error = anyhow::Error;

    fn try_into(self) -> std::result::Result<EmbeddingModel, Self::Error> {
        match self {
            OramaModels::GTESmall => Ok(EmbeddingModel::BGESmallENV15),
            OramaModels::GTEBase => Ok(EmbeddingModel::BGEBaseENV15),
            OramaModels::GTELarge => Ok(EmbeddingModel::BGELargeENV15),
            OramaModels::MultilingualE5Small => Ok(EmbeddingModel::MultilingualE5Small),
            OramaModels::MultilingualE5Base => Ok(EmbeddingModel::MultilingualE5Base),
            OramaModels::MultilingualE5Large => Ok(EmbeddingModel::MultilingualE5Large),
            OramaModels::JinaV2BaseCode => Err(anyhow!("JinaV2BaseCode is a custom model")),
        }
    }
}

impl OramaModels {
    pub fn try_new(&self) -> Result<Arc<TextEmbedding>> {
        MODELS.get_or_init(|| RwLock::new(HashMap::new()));

        let mut models_map = MODELS
            .get()
            .unwrap()
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on the models map"))?;

        if let Some(existing_model) = models_map.get(self) {
            return Ok(existing_model.clone());
        }

        let new_model = if !self.is_custom_model() {
            let embedding_model = (*self).try_into()?;
            TextEmbedding::try_new(
                InitOptions::new(embedding_model).with_show_download_progress(true),
            )
            .with_context(|| format!("Failed to initialize the model: {}", self))
        } else {
            let custom_model = CustomModel::try_new(
                self.to_string(),
                self.files().ok_or_else(|| {
                    anyhow!("Missing file configuration for custom model {}", self)
                })?,
            )
            .with_context(|| format!("Unable to initialize custom model {}", self))?;

            if custom_model.exists() {
                custom_model
                    .download()
                    .with_context(|| format!("Unable to download custom model {}", self))?;
            }

            let init_model = custom_model
                .load()
                .with_context(|| format!("Unable to load local files for custom model {}", self))?;

            TextEmbedding::try_new_from_user_defined(init_model, InitOptionsUserDefined::default())
                .with_context(|| format!("Unable to initialize custom model {}", self))
        }?;

        let arc_model = Arc::new(new_model);
        models_map.insert(*self, arc_model.clone());
        Ok(arc_model)
    }

    pub fn normalize_input(self, intent: EncodingIntent, input: Vec<String>) -> Vec<String> {
        match self {
            OramaModels::MultilingualE5Small
            | OramaModels::MultilingualE5Base
            | OramaModels::MultilingualE5Large => input
                .into_iter()
                .map(|text| format!("{intent}: {text}"))
                .collect(),
            _ => input,
        }
    }

    pub fn is_custom_model(self) -> bool {
        match self {
            OramaModels::JinaV2BaseCode => true,
            _ => false,
        }
    }

    pub fn max_input_tokens(self) -> usize {
        match self {
            OramaModels::JinaV2BaseCode => 512,
            OramaModels::GTESmall => 512,
            OramaModels::GTEBase => 512,
            OramaModels::GTELarge => 512,
            OramaModels::MultilingualE5Small => 512,
            OramaModels::MultilingualE5Base => 512,
            OramaModels::MultilingualE5Large => 512,
        }
    }

    pub fn dimensions(self) -> usize {
        match self {
            OramaModels::JinaV2BaseCode => 768,
            OramaModels::GTESmall => 384,
            OramaModels::GTEBase => 768,
            OramaModels::GTELarge => 1024,
            OramaModels::MultilingualE5Small => 384,
            OramaModels::MultilingualE5Base => 768,
            OramaModels::MultilingualE5Large => 1024,
        }
    }

    pub fn files(self) -> Option<ModelFileConfig> {
        match self {
            OramaModels::JinaV2BaseCode => Some(ModelFileConfig {
                onnx_model: "onnx/model.onnx".to_string(),
                special_tokens_map: "special_tokens_map.json".to_string(),
                tokenizer: "tokenizer.json".to_string(),
                tokenizer_config: "tokenizer_config.json".to_string(),
                config: "config.json".to_string(),
            }),
            _ => None,
        }
    }
}

impl fmt::Display for EncodingIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodingIntent::Query => write!(f, "query"),
            EncodingIntent::Passage => write!(f, "passage"),
        }
    }
}
