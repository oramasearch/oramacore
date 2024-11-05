pub mod custom_models;
pub mod pq;

use crate::custom_models::{CustomModel, ModelFileConfig};
use anyhow::{anyhow, Context, Result};
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use strum::EnumIter;
use strum::IntoEnumIterator;
use strum_macros::{AsRefStr, Display};

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
#[strum(serialize_all = "kebab-case")]
pub enum OramaModels {
    #[serde(rename = "gte-small")]
    GTESmall,
    #[serde(rename = "gte-base")]
    GTEBase,
    #[serde(rename = "gte-large")]
    GTELarge,
    #[serde(rename = "multilingual-e5-small")]
    MultilingualE5Small,
    #[serde(rename = "multilingual-e5-base")]
    MultilingualE5Base,
    #[serde(rename = "multilingual-e5-large")]
    MultilingualE5Large,
    #[serde(rename = "jinaai/jina-embeddings-v2-base-code")]
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

impl Into<EmbeddingModel> for OramaModels {
    fn into(self) -> EmbeddingModel {
        match self {
            OramaModels::JinaV2BaseCode => EmbeddingModel::BGESmallENV15, // @todo: understand how to use other models
            OramaModels::GTESmall => EmbeddingModel::BGESmallENV15,
            OramaModels::GTEBase => EmbeddingModel::BGEBaseENV15,
            OramaModels::GTELarge => EmbeddingModel::BGELargeENV15,
            OramaModels::MultilingualE5Small => EmbeddingModel::MultilingualE5Small,
            OramaModels::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
            OramaModels::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
        }
    }
}

impl OramaModels {
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
                onnx_model: "onnx.model.onnx".to_string(),
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

pub fn load_models() -> LoadedModels {
    let models: Vec<_> = vec![
        OramaModels::MultilingualE5Small,
        // OramaModels::MultilingualE5Base,
        // OramaModels::MultilingualE5Large,
        OramaModels::GTESmall,
        // OramaModels::GTEBase,
        // OramaModels::GTELarge,
        OramaModels::JinaV2BaseCode,
    ];

    let model_map: HashMap<OramaModels, TextEmbedding> = models
        .into_par_iter()
        .map(|model| {
            if !model.is_custom_model() {
                let initialized_model = TextEmbedding::try_new(
                    InitOptions::new(model.into()).with_show_download_progress(true),
                )
                .unwrap();

                return (model, initialized_model);
            }

            let custom_model = CustomModel::try_new(model.to_string(), model.files().unwrap())
                .with_context(|| format!("Unable to initialize custom model {}", model.to_string()))
                .unwrap();

            if custom_model.exists() {
                custom_model
                    .download()
                    .with_context(|| {
                        format!("Unable to download custom model {}", model.to_string())
                    })
                    .unwrap();
            };

            let init_model = custom_model
                .load()
                .with_context(|| {
                    format!(
                        "Unable to load local files for custom model {}",
                        model.to_string()
                    )
                })
                .unwrap();
            let initialized_model = TextEmbedding::try_new_from_user_defined(
                init_model,
                InitOptionsUserDefined::default(),
            )
            .with_context(|| {
                format!(
                    "Unable to initialize a new TextEmbedding instance from custom model {}",
                    model.to_string()
                )
            })
            .unwrap();

            return (model, initialized_model);
        })
        .collect();

    LoadedModels(model_map)
}
