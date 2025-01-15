use std::{collections::HashMap, net::IpAddr};

use crate::ai::{
    calculate_embeddings_service_client::CalculateEmbeddingsServiceClient, EmbeddingRequest,
    OramaIntent, OramaModel,
};
use http::uri::Scheme;
use tonic::Request;

use anyhow::{anyhow, Context, Result};
use mobc::{async_trait, Manager, Pool};
use serde::Deserialize;
use tracing::info;

struct GrpcConnection {
    client: CalculateEmbeddingsServiceClient<tonic::transport::Channel>,
}

impl GrpcConnection {
    async fn check(&self) -> Result<()> {
        // self.client.
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

        let client: CalculateEmbeddingsServiceClient<tonic::transport::Channel> =
            CalculateEmbeddingsServiceClient::new(endpoint);

        Ok(GrpcConnection { client })
    }

    async fn check(&self, conn: Self::Connection) -> Result<Self::Connection, Self::Error> {
        conn.check().await?;

        Ok(conn)
    }
}

#[derive(Debug)]
pub struct GrpcModel {
    model_name: String,
    model_id: i32,
    manager: Pool<GrpcManager>,
    dimensions: usize,
}
impl GrpcModel {
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
    pub async fn load_model(&self, model_name: String) -> Result<GrpcModel> {
        info!("Loading model");

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

        let model = GrpcModel {
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

        let output = model.embed_query(vec![&"foo".to_string()]).await?;

        assert_eq!(output[0].len(), 384);

        Ok(())
    }
}
