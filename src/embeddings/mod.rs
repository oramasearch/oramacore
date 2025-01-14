use anyhow::{anyhow, Ok, Result};
use dashmap::DashMap;

use hf::HuggingFaceRepoConfig;
use itertools::Itertools;
use serde::Deserialize;
use std::{collections::HashMap, fmt::Debug, hash::Hash, sync::Arc};

pub mod fe;
pub mod grpc;
pub mod hf;

#[derive(Debug)]
pub enum LoadedModel {
    HuggingFace(hf::HuggingFaceModel),
    Fastembed(fe::FastEmbedModel),
    Grpc(grpc::GrpcModel),
}

impl PartialEq for LoadedModel {
    fn eq(&self, other: &Self) -> bool {
        let model_name_eq = self.model_name() == other.model_name();
        match (self, other) {
            (Self::HuggingFace(_), Self::HuggingFace(_)) => model_name_eq,
            (Self::Fastembed(_), Self::Fastembed(_)) => model_name_eq,
            (Self::Grpc(_), Self::Grpc(_)) => model_name_eq,
            _ => false,
        }
    }
}
impl Eq for LoadedModel {}

impl Hash for LoadedModel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::HuggingFace(_) => "HuggingFace".hash(state),
            Self::Fastembed(_) => "Fastembed".hash(state),
            Self::Grpc(_) => "Grpc".hash(state),
        };
        self.model_name().hash(state);
    }
}

impl LoadedModel {
    pub fn model_name(&self) -> String {
        match self {
            LoadedModel::HuggingFace(model) => model.model_name(),
            LoadedModel::Fastembed(model) => model.model_name(),
            LoadedModel::Grpc(model) => model.model_name(),
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            LoadedModel::HuggingFace(model) => model.dimensions(),
            LoadedModel::Fastembed(model) => model.dimensions(),
            LoadedModel::Grpc(model) => model.dimensions(),
        }
    }

    pub async fn embed(&self, input: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        match self {
            LoadedModel::HuggingFace(model) => model.embed(input),
            LoadedModel::Fastembed(model) => model.embed(input),
            LoadedModel::Grpc(model) => model.embed(input).await,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ModelConfig {
    #[serde(rename = "hugging_face")]
    HuggingFace(hf::HuggingFaceModelRepoConfig),
    #[serde(rename = "fastembed")]
    Fastembed(fe::FastEmbedModelRepoConfig),
    #[serde(rename = "grpc")]
    Grpc(grpc::GrpcModelConfig),
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub preload: Vec<String>,
    pub grpc: Option<grpc::GrpcRepoConfig>,
    pub hugging_face: Option<HuggingFaceRepoConfig>,
    pub fastembed: Option<fe::FastEmbedRepoConfig>,
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Debug)]
enum Repo {
    FastEmbed,
    HuggingFace,
    Grpc,
}

#[derive(Debug)]
pub struct EmbeddingService {
    fastembed_repo: Option<fe::FastEmbedRepo>,
    hugging_face_repo: Option<hf::HuggingFaceRepo>,
    grpc_repo: Option<grpc::GrpcRepo>,

    models_repo: HashMap<String, Repo>,
    loaded_models: DashMap<String, Arc<LoadedModel>>,
}

impl EmbeddingService {
    pub async fn try_new(config: EmbeddingConfig) -> Result<Self> {
        let mut service = EmbeddingService {
            fastembed_repo: None,
            hugging_face_repo: None,
            grpc_repo: None,
            models_repo: HashMap::new(),
            loaded_models: DashMap::new(),
        };

        let EmbeddingConfig {
            fastembed,
            grpc,
            hugging_face,
            models,
            preload,
        } = config;

        let mut models = models
            .into_iter()
            .map(|(k, v)| match v {
                ModelConfig::HuggingFace(_) => ("hugging_face", (k, v)),
                ModelConfig::Fastembed(_) => ("fastembed", (k, v)),
                ModelConfig::Grpc(_) => ("grpc", (k, v)),
            })
            .into_group_map();

        if let Some(fastembed) = fastembed {
            let model_configs = models.remove("fastembed").unwrap_or_default();
            let model_configs: HashMap<String, fe::FastEmbedModelRepoConfig> = model_configs
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        match v {
                            ModelConfig::Fastembed(v) => v,
                            _ => unreachable!(),
                        },
                    )
                })
                .collect();

            for model_name in model_configs.keys() {
                service
                    .models_repo
                    .insert(model_name.clone(), Repo::FastEmbed);
            }

            service.fastembed_repo = Some(fe::FastEmbedRepo::new(fastembed, model_configs));
        }

        if let Some(hugging_face) = hugging_face {
            let model_configs = models.remove("hugging_face").unwrap_or_default();
            let model_configs: HashMap<String, hf::HuggingFaceModelRepoConfig> = model_configs
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        match v {
                            ModelConfig::HuggingFace(v) => v,
                            _ => unreachable!(),
                        },
                    )
                })
                .collect();

            for model_name in model_configs.keys() {
                service
                    .models_repo
                    .insert(model_name.clone(), Repo::HuggingFace);
            }

            service.hugging_face_repo = Some(hf::HuggingFaceRepo::new(hugging_face, model_configs));
        }

        if let Some(grpc) = grpc {
            let model_configs = models.remove("grpc").unwrap_or_default();
            let model_configs: HashMap<String, grpc::GrpcModelConfig> = model_configs
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        match v {
                            ModelConfig::Grpc(v) => v,
                            _ => unreachable!(),
                        },
                    )
                })
                .collect();

            for model_name in model_configs.keys() {
                service.models_repo.insert(model_name.clone(), Repo::Grpc);
            }

            service.grpc_repo = Some(grpc::GrpcRepo::new(grpc, model_configs));
        }

        debug_assert!(models.is_empty(), "Models should be empty");

        for model_name in preload {
            service.load_model(model_name).await?;
        }

        Ok(service)
    }

    pub async fn get_model(&self, model_name: String) -> Result<Arc<LoadedModel>> {
        match self.loaded_models.entry(model_name.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => Ok(entry.get().clone()),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                println!("Loading model: {}", model_name);
                let model = self.load_model(model_name).await?;
                println!("Model loaded");
                entry.insert(model.clone());
                Ok(model)
            }
        }
    }

    async fn load_model(&self, model_name: String) -> Result<Arc<LoadedModel>> {
        let repo = match self.models_repo.get(&model_name) {
            None => {
                return Err(anyhow!("Model not found: {}", model_name));
            }
            Some(repo) => repo,
        };

        let loaded_model = match repo {
            Repo::FastEmbed => {
                let repo = self
                    .fastembed_repo
                    .as_ref()
                    .ok_or_else(|| anyhow!("FastEmbedRepo is missing"))?;
                let model = repo.load_model(model_name.clone()).await?;

                LoadedModel::Fastembed(model)
            }
            Repo::HuggingFace => {
                let repo = self
                    .hugging_face_repo
                    .as_ref()
                    .ok_or_else(|| anyhow!("HuggingFaceRepo is missing"))?;
                let model = repo.load_model(model_name.clone()).await?;

                LoadedModel::HuggingFace(model)
            }
            Repo::Grpc => {
                let repo = self
                    .grpc_repo
                    .as_ref()
                    .ok_or_else(|| anyhow!("GrpcRepo is missing"))?;

                let model = repo.load_model(model_name.clone()).await?;

                LoadedModel::Grpc(model)
            }
        };

        let loaded_model = Arc::new(loaded_model);

        Ok(loaded_model)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_embedding_deserialize_config() {
        let models = r#"
    {
        "small-model": {
            "type": "hugging_face",
            "max_input_tokens": 512,
            "dimensions": 384,
            "real_model_name": "Xenova/gte-small",
            "files": {
                "onnx_model": "onnx/model_quantized.onnx",
                "special_tokens_map": "special_tokens_map.json",
                "tokenizer": "tokenizer.json",
                "tokenizer_config": "tokenizer_config.json",
                "config": "config.json"
            }
        },

        "foo-model": {
            "type": "grpc",
            "real_model_name": "BGESmall",
            "dimensions": 384
        },

        "gte-small": {
            "type": "fastembed",
            "real_model_name": "Xenova/bge-small-en-v1.5",
            "dimensions": 384
        }
    }
        "#;

        let models =
            serde_json::from_str::<std::collections::HashMap<String, super::ModelConfig>>(models)
                .unwrap();

        assert_eq!(models.len(), 3);
    }

    /*
    #[tokio::test]
    async fn test_embeddings_calculate_embedding() {
        let _ = tracing_subscriber::fmt::try_init();

        // We don't want to download the model every time we run the test
        // So we use a standard temp directory, without any cleanup logic
        let temp_dir = std::env::temp_dir();

        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            cache_path: temp_dir.clone(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        })
        .await
        .expect("Failed to initialize the EmbeddingService");

        let model = embedding_service.get_model(OramaModel::Fastembed(OramaFastembedModel::GTESmall))
            .await
            .expect("Failed to get model");

        let output = model
            .embed(
                vec!["Hello, world!".to_string()],
                Some(1),
            )
            .expect("Failed to embed text");

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), OramaFastembedModel::GTESmall.dimensions());

        assert_eq!(embedding_service.loaded_models.len(), 1);
    }
    */
}
