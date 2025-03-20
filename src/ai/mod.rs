use std::time::Duration;

use backoff::ExponentialBackoff;
use http::uri::Scheme;
use llm_service_client::LlmServiceClient;
use mobc::{async_trait, Manager, Pool};
use serde::Deserialize;

use anyhow::{anyhow, Context, Result};
use tonic::{transport::Channel, Request};
use tracing::{debug, info, trace};

use crate::metrics::{
    ai::{EMBEDDING_CALCULATION_PARALLEL_COUNT, EMBEDDING_CALCULATION_TIME},
    EmbeddingCalculationLabels,
};

pub mod context_evaluator;
pub mod party_planner;
pub mod vllm;

tonic::include_proto!("orama_ai_service");

impl OramaModel {
    pub fn dimensions(&self) -> usize {
        match self {
            OramaModel::BgeSmall => 384,
            OramaModel::BgeBase => 768,
            OramaModel::BgeLarge => 1024,
            OramaModel::MultilingualE5Small => 384,
            OramaModel::MultilingualE5Base => 768,
            OramaModel::MultilingualE5Large => 1024,
            OramaModel::JinaEmbeddingsV2BaseCode => 768,
            OramaModel::MultilingualMiniLml12v2 => 768,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AIServiceLLMConfig {
    pub port: u16,
    pub host: String,
    pub model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AIServiceConfig {
    #[serde(deserialize_with = "deserialize_scheme")]
    #[serde(default = "default_scheme")]
    pub scheme: Scheme,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    #[serde(default = "default_max_connections")]
    pub max_connections: u64,
    pub llm: AIServiceLLMConfig,
}

#[derive(Debug)]
pub struct AIService {
    pool: Pool<GrpcManager>,
}

impl AIService {
    pub fn new(config: AIServiceConfig) -> Self {
        let pool = Pool::builder().max_open(15).build(GrpcManager { config });

        Self { pool }
    }

    pub async fn wait_ready(&self) -> Result<()> {
        loop {
            match self.pool.get().await {
                Ok(_) => {
                    info!("AI service is ready");
                    break Ok(());
                }
                Err(err) => {
                    info!("AI service is not ready: {}. Waiting again...", err);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    async fn embed(
        &self,
        model: OramaModel,
        input: Vec<&String>,
        intent: OramaIntent,
    ) -> Result<Vec<Vec<f32>>> {
        let mut conn = self.pool.get().await.context("Cannot get connection")?;

        let time_metric = EMBEDDING_CALCULATION_TIME.create(EmbeddingCalculationLabels {
            model: model.as_str_name().into(),
        });
        EMBEDDING_CALCULATION_PARALLEL_COUNT.track_usize(
            EmbeddingCalculationLabels {
                model: model.as_str_name().into(),
            },
            input.len(),
        );

        trace!("Requesting embeddings");
        let request = Request::new(EmbeddingRequest {
            input: input.iter().map(|s| s.to_string()).collect(),
            model: model.into(),
            intent: intent.into(),
        });
        let v = conn
            .get_embedding(request)
            .await
            .map(|response| response.into_inner())
            .context("Cannot get embeddings")?;

        drop(time_metric);

        trace!("Received embeddings");

        let v = v
            .embeddings_result
            .into_iter()
            .map(|embedding| embedding.embeddings)
            .collect();

        Ok(v)
    }

    pub async fn embed_query(
        &self,
        model: OramaModel,
        input: Vec<&String>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed(model, input, OramaIntent::Query).await
    }

    pub async fn embed_passage(
        &self,
        model: OramaModel,
        input: Vec<&String>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed(model, input, OramaIntent::Passage).await
    }
}

#[derive(Debug)]
struct GrpcManager {
    config: AIServiceConfig,
}

#[async_trait]
impl Manager for GrpcManager {
    type Connection = LlmServiceClient<Channel>;
    type Error = anyhow::Error;

    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        use http::Uri;

        let uri = Uri::builder()
            .scheme(Scheme::HTTP)
            .authority(format!("{}:{}", self.config.host, self.config.port))
            .path_and_query("/")
            .build()
            .context("Cannot build URI")?;

        info!("Connecting to gRPC");
        use backoff::{future::retry, Error};
        use tonic::transport::{Channel, Endpoint};

        let uri = &uri;
        let channel: std::result::Result<Channel, backoff::Error<anyhow::Error>> =
            retry(ExponentialBackoff::default(), || async {
                debug!("Trying to connect to {:?}", uri);
                let endpoint: Endpoint = Endpoint::new(uri.clone())
                    .context("Cannot create endpoint")
                    .map_err(Error::permanent)?;

                let channel: Channel = endpoint
                    .connect()
                    .await
                    .with_context(move || format!("Cannot connect to {:?}", uri))
                    .map_err(Error::transient)?;

                Ok(channel)
            })
            .await;

        let channel = match channel {
            Ok(channel) => channel,
            Err(err) => return Err(anyhow!("Cannot connect to gRPC: {}", err)),
        };

        let client: LlmServiceClient<tonic::transport::Channel> = LlmServiceClient::new(channel)
            .max_decoding_message_size(16_777_216) // 16MB
            .max_encoding_message_size(16_777_216); // 16MB

        Ok(client)
    }

    async fn check(&self, mut conn: Self::Connection) -> Result<Self::Connection, Self::Error> {
        let health_check_request = Request::new(HealthCheckRequest {
            service: "HealthCheck".to_string(),
        });

        let response = conn.check_health(health_check_request).await?;
        let response = response.into_inner();

        if response.status != "OK" {
            return Err(anyhow!("Invalid status: {}", response.status));
        }

        Ok(conn)
    }
}

fn deserialize_scheme<'de, D>(deserializer: D) -> Result<Scheme, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    struct SchemeVisitor;

    impl serde::de::Visitor<'_> for SchemeVisitor {
        type Value = Scheme;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string containing json data")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            match Scheme::try_from(v) {
                Ok(scheme) => Ok(scheme),
                Err(_) => Err(E::custom("Invalid scheme")),
            }
        }
    }

    // use our visitor to deserialize an `ActualValue`
    deserializer.deserialize_any(SchemeVisitor)
}

fn default_max_connections() -> u64 {
    15
}
fn default_scheme() -> Scheme {
    Scheme::HTTP
}
