use anyhow::Result;
use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display, EnumIter};
use tonic::{transport::Channel, Request, Response};

pub mod orama_ai_service {
    tonic::include_proto!("orama_ai_service");
}

use orama_ai_service::{
    EmbeddingRequest, EmbeddingResponse, HealthCheckRequest, HealthCheckResponse, OramaIntent,
    OramaModel,
};

use crate::ai_client::client::orama_ai_service::llm_service_client::LlmServiceClient;

#[derive(Debug, Clone, Copy)]
pub enum Intent {
    Query,
    Passage,
}

impl From<Intent> for OramaIntent {
    fn from(intent: Intent) -> Self {
        match intent {
            Intent::Query => Self::Query,
            Intent::Passage => Self::Passage,
        }
    }
}

#[derive(Debug, Default)]
pub struct AIServiceBackendConfig {
    pub host: Option<String>,
    pub port: Option<String>,
    pub api_key: Option<String>,
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, EnumIter, Display, AsRefStr,
)]
pub enum LlmType {
    #[serde(rename = "content_expansion")]
    #[strum(serialize = "content_expansion")]
    ContentExpansion,
    #[serde(rename = "google_query_translator")]
    #[strum(serialize = "google_query_translator")]
    GoogleQueryTranslator,
    #[serde(rename = "vision")]
    #[strum(serialize = "vision")]
    Vision,
}

impl From<LlmType> for i32 {
    fn from(llm_type: LlmType) -> Self {
        match llm_type {
            LlmType::ContentExpansion => 0,
            LlmType::GoogleQueryTranslator => 1,
            LlmType::Vision => 2,
        }
    }
}

#[derive(Debug)]
pub struct AIServiceBackend {
    llm_client: LlmServiceClient<Channel>,
}

impl AIServiceBackend {
    const DEFAULT_HOST: &'static str = "localhost";
    const DEFAULT_PORT: &'static str = "50051";

    pub async fn try_new(config: AIServiceBackendConfig) -> Result<Self> {
        let addr = format!(
            "http://{}:{}",
            config.host.as_deref().unwrap_or(Self::DEFAULT_HOST),
            config.port.as_deref().unwrap_or(Self::DEFAULT_PORT),
        );

        let llm_client = LlmServiceClient::connect(addr.clone()).await?;

        Ok(Self { llm_client })
    }

    pub async fn health_check(&mut self) -> Result<Response<HealthCheckResponse>> {
        let request = Request::new(HealthCheckRequest {
            service: "HealthCheck".to_string(),
        });

        Ok(self.llm_client.check_health(request).await?)
    }

    pub async fn generate_embeddings(
        &mut self,
        input: Vec<String>,
        model: Model,
        intent: Intent,
    ) -> Result<Response<EmbeddingResponse>> {
        let request = Request::new(EmbeddingRequest {
            input: input.clone(),
            model: model.into(),
            intent: intent.into(),
        });

        Ok(self.llm_client.get_embedding(request).await?)
    }
}

impl Model {
    pub const fn dimensions(&self) -> usize {
        match self {
            Self::BgeSmall | Self::MultilingualE5Small => 384,
            Self::BgeBase | Self::MultilingualE5Base => 768,
            Self::BgeLarge | Self::MultilingualE5Large => 1024,
        }
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, EnumIter, Display, AsRefStr,
)]
pub enum Model {
    #[serde(rename = "bge-small")]
    #[strum(serialize = "bge-small")]
    BgeSmall,
    #[serde(rename = "bge-base")]
    #[strum(serialize = "bge-base")]
    BgeBase,
    #[serde(rename = "bge-large")]
    #[strum(serialize = "bge-large")]
    BgeLarge,
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

impl From<Model> for i32 {
    fn from(model: Model) -> Self {
        OramaModel::from(model) as i32
    }
}

impl From<Model> for OramaModel {
    fn from(model: Model) -> Self {
        match model {
            Model::BgeSmall => Self::BgeSmall,
            Model::BgeBase => Self::BgeBase,
            Model::BgeLarge => Self::BgeLarge,
            Model::MultilingualE5Small => Self::MultilingualE5Small,
            Model::MultilingualE5Base => Self::MultilingualE5Base,
            Model::MultilingualE5Large => Self::MultilingualE5Large,
        }
    }
}

impl From<Intent> for i32 {
    fn from(intent: Intent) -> Self {
        OramaIntent::from(intent) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_sync() {
        fn test_send_sync<T: Send + Sync>() {}

        test_send_sync::<Model>();
        test_send_sync::<AIServiceBackend>();
    }
}
