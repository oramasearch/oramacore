use std::pin::Pin;
use std::{collections::HashMap, net::IpAddr};

use http::uri::Scheme;
use tonic::{Request, Response, Streaming};

use crate::ai::llm_service_client::LlmServiceClient;
use crate::ai::{
    ChatRequest, ChatResponse, Conversation, EmbeddingRequest, HealthCheckRequest, LlmType,
    OramaIntent, OramaModel,
};
use anyhow::{anyhow, Context, Result};
use futures::Stream;
use mobc::{async_trait, Manager, Pool};
use serde::Deserialize;
use tracing::info;

use super::ChatStreamResponse;

pub type ChatStreamResult = Pin<Box<dyn Stream<Item = Result<ChatStreamResponse>> + Send>>;

struct GrpcConnection {
    client: LlmServiceClient<tonic::transport::Channel>,
}

impl GrpcConnection {
    async fn check(&mut self) -> Result<()> {
        let health_check_request = Request::new(HealthCheckRequest {
            service: "HealthCheck".to_string(),
        });

        let response = self.client.check_health(health_check_request).await?;
        let response = response.into_inner();

        if response.status != "OK" {
            return Err(anyhow!("Invalid status: {}", response.status));
        }

        Ok(())
    }
}

struct GrpcManager {
    config: GrpcRepoConfig,
}

impl GrpcManager {
    pub fn new(config: GrpcRepoConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Manager for GrpcManager {
    type Connection = GrpcConnection;
    type Error = anyhow::Error;

    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        use http::Uri;

        let uri = Uri::builder()
            .scheme(Scheme::HTTP)
            .authority(format!("{}:{}", self.config.host, self.config.port))
            .path_and_query("/")
            .build()
            .unwrap();

        info!("Connecting to gRPC");
        let endpoint = tonic::transport::Endpoint::new(uri)?.connect().await?;
        info!("Connected to gRPC");

        let client: LlmServiceClient<tonic::transport::Channel> = LlmServiceClient::new(endpoint);

        Ok(GrpcConnection { client })
    }

    async fn check(&self, mut conn: Self::Connection) -> Result<Self::Connection, Self::Error> {
        conn.check().await?;

        Ok(conn)
    }
}

#[derive(Debug)]
pub struct GrpcLLM {
    manager: Pool<GrpcManager>,
}

impl GrpcLLM {
    pub async fn chat(
        &self,
        llm_type: LlmType,
        prompt: String,
        conversation: Conversation,
        context: Option<String>,
    ) -> Result<ChatResponse> {
        let mut conn = self.manager.get().await.context("Cannot get connection")?;

        let request = Request::new(ChatRequest {
            conversation: Some(conversation),
            prompt,
            model: llm_type as i32,
            context,
        });

        let response = conn
            .client
            .chat(request)
            .await
            .map(|response| response.into_inner())
            .context("Cannot perform chat request")?;

        Ok(response)
    }

    pub async fn chat_stream(
        &self,
        llm_type: LlmType,
        prompt: String,
        conversation: Conversation,
        context: Option<String>,
    ) -> Result<Streaming<ChatStreamResponse>> {
        let mut conn = self.manager.get().await.context("Cannot get connection")?;

        let request = Request::new(ChatRequest {
            conversation: Some(conversation),
            prompt,
            model: llm_type as i32,
            context,
        });

        let response: Response<Streaming<ChatStreamResponse>> = conn
            .client
            .chat_stream(request)
            .await
            .context("Cannot initiate chat stream request")?;

        Ok(response.into_inner())
    }
}

#[derive(Debug)]
pub struct GrpcEmbeddingModel {
    model_name: String,
    model_id: i32,
    manager: Pool<GrpcManager>,
    dimensions: usize,
}
impl GrpcEmbeddingModel {
    pub fn model_name(&self) -> String {
        self.model_name.clone()
    }
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, input: Vec<&String>, intent: OramaIntent) -> Result<Vec<Vec<f32>>> {
        let mut conn = self.manager.get().await.context("Cannot get connection")?;

        // We cloned the input because tonic requires it even if it shouldn't
        // In fact, tonic will copy the strings into a TCP socket, so it shouldn't be own values
        // Anyway we have to.

        let request = Request::new(EmbeddingRequest {
            input: input.into_iter().cloned().collect(),
            model: self.model_id,
            intent: intent.into(),
        });

        let v = conn
            .client
            .get_embedding(request)
            .await
            .map(|response| response.into_inner())
            .context("Cannot get embeddings")?;

        let v = v
            .embeddings_result
            .into_iter()
            .map(|embedding| embedding.embeddings)
            .collect();

        Ok(v)
    }

    pub async fn embed_query(&self, input: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        self.embed(input, OramaIntent::Query).await
    }

    pub async fn embed_passage(&self, input: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        self.embed(input, OramaIntent::Passage).await
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GrpcModelConfig {
    pub real_model_name: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GrpcRepoConfig {
    pub host: IpAddr,
    pub port: u16,
    pub api_key: Option<String>,
}

#[derive(Debug)]
pub struct GrpcRepo {
    grpc_config: GrpcRepoConfig,
    model_configs: HashMap<String, GrpcModelConfig>,
}

impl GrpcRepo {
    pub fn new(
        grpc_config: GrpcRepoConfig,
        model_configs: HashMap<String, GrpcModelConfig>,
    ) -> Self {
        Self {
            grpc_config,
            model_configs,
        }
    }

    #[tracing::instrument]
    pub async fn load_llm(&self) -> Result<GrpcLLM> {
        info!("Creating pool");
        let pool = Pool::builder()
            .max_open(15)
            .build(GrpcManager::new(GrpcRepoConfig {
                host: self.grpc_config.host,
                port: self.grpc_config.port,
                api_key: self.grpc_config.api_key.clone(),
            }));

        let model = GrpcLLM { manager: pool };

        Ok(model)
    }

    #[tracing::instrument]
    pub async fn load_model(&self, model_name: String) -> Result<GrpcEmbeddingModel> {
        info!("Loading GRPC model");

        let model_config = match self.model_configs.get(&model_name) {
            Some(config) => config,
            None => {
                return Err(anyhow!("Model not found: {}", model_name));
            }
        };

        let model = match OramaModel::from_str_name(&model_config.real_model_name) {
            Some(model) => model,
            None => {
                return Err(anyhow!(
                    "Unknown model name: {}",
                    model_config.real_model_name
                ));
            }
        };

        info!("Creating pool");
        let pool = Pool::builder()
            .max_open(15)
            .build(GrpcManager::new(GrpcRepoConfig {
                host: self.grpc_config.host,
                port: self.grpc_config.port,
                api_key: self.grpc_config.api_key.clone(),
            }));

        let model = GrpcEmbeddingModel {
            model_name: model_name.clone(),
            model_id: model.into(),
            manager: pool,
            dimensions: model_config.dimensions,
        };

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "test-python")]
    #[tokio::test]
    async fn test_embedding_run_grpc() -> anyhow::Result<()> {
        use super::*;

        let _ = tracing_subscriber::fmt::try_init();

        let grpc_config = GrpcRepoConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 50051,
            api_key: None,
        };

        let model = GrpcModelConfig {
            real_model_name: "BGESmall".to_string(),
            dimensions: 384,
        };

        let rebranded_name = "my-model".to_string();
        let repo = GrpcRepo::new(
            grpc_config,
            HashMap::from_iter([(rebranded_name.clone(), model)]),
        );
        let model = repo
            .load_model(rebranded_name)
            .await
            .expect("Failed to cache model");

        let output = model
            .embed(vec![&"foo".to_string()], OramaIntent::Passage)
            .await?;

        assert_eq!(output[0].len(), 384);

        Ok(())
    }
}
