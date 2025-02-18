use std::{net::IpAddr, pin::Pin, str::FromStr, task::Poll};

use futures::Stream;
use http::uri::Scheme;
use llm_service_client::LlmServiceClient;
use mobc::{async_trait, Manager, Pool};
use pin_project_lite::pin_project;
use serde::Deserialize;

use anyhow::{anyhow, Context, Result};
use tonic::{metadata::MetadataValue, transport::Channel, Request, Response, Status, Streaming};
use tracing::{info, trace};

pub mod segments;

use crate::{
    collection_manager::dto::{ApiKey, InteractionMessage},
    metrics::{
        ai::{
            CHAT_CALCULATION_TIME, EMBEDDING_CALCULATION_PARALLEL_COUNT,
            EMBEDDING_CALCULATION_TIME, STREAM_CHAT_CALCULATION_TIME,
        },
        ChatCalculationLabels, EmbeddingCalculationLabels,
    },
};

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
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AIServiceConfig {
    #[serde(deserialize_with = "deserialize_scheme")]
    #[serde(default = "default_scheme")]
    pub scheme: Scheme,
    pub host: IpAddr,
    pub port: u16,
    pub api_key: Option<String>,
    #[serde(default = "default_max_connections")]
    pub max_connections: u64,
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
            input: input.into_iter().cloned().collect(),
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

    pub async fn chat(
        &self,
        llm_type: LlmType,
        prompt: String,
        conversation: Option<Vec<InteractionMessage>>,
        context: Option<String>,
    ) -> Result<ChatResponse> {
        let mut conn = self.pool.get().await.context("Cannot get connection")?;

        let time_metric = CHAT_CALCULATION_TIME.create(ChatCalculationLabels {
            model: llm_type.as_str_name(),
        });

        let conversation = self.get_grpc_conversation(conversation);
        let request = Request::new(ChatRequest {
            conversation: Some(conversation),
            prompt,
            model: llm_type as i32,
            context,
        });

        let response = conn
            .chat(request)
            .await
            .map(|response| response.into_inner())
            .context("Cannot perform chat request")?;

        drop(time_metric);

        Ok(response)
    }

    pub async fn chat_stream(
        &self,
        llm_type: LlmType,
        prompt: String,
        conversation: Option<Vec<InteractionMessage>>,
        context: Option<String>,
    ) -> Result<ChatStream> {
        let mut conn = self.pool.get().await.context("Cannot get connection")?;

        let time_metric = STREAM_CHAT_CALCULATION_TIME.create(ChatCalculationLabels {
            model: llm_type.as_str_name(),
        });

        let conversation = self.get_grpc_conversation(conversation);
        let request = Request::new(ChatRequest {
            conversation: Some(conversation),
            prompt,
            model: llm_type as i32,
            context,
        });

        let response: Response<Streaming<ChatStreamResponse>> = conn
            .chat_stream(request)
            .await
            .context("Cannot initiate chat stream request")?;

        let chat_stream = ChatStream {
            inner: response.into_inner(),
            end_cb: std::sync::RwLock::new(Some(Box::new(move || {
                drop(time_metric);
            }))),
        };

        Ok(chat_stream)
    }

    pub async fn planned_answer_stream(
        &self,
        input: String,
        collection_id: String,
        conversation: Option<Vec<InteractionMessage>>,
        api_key: ApiKey,
    ) -> Result<Streaming<PlannedAnswerResponse>> {
        let mut conn = self.pool.get().await.context("Cannot get connection")?;

        let conversation = self.get_grpc_conversation(conversation);

        // let mut metadata = MetadataMap::new();
        let api_key_value =
            MetadataValue::from_str(api_key.0.expose_secret()).context("Invalid API key")?;

        // metadata.insert("x-api-key", api_key_value.clone());

        let mut request = Request::new(PlannedAnswerRequest {
            input,
            collection_id,
            conversation: Some(conversation),
        });

        request.metadata_mut().append("x-api-key", api_key_value);

        let response: Response<Streaming<PlannedAnswerResponse>> = conn
            .planned_answer(request)
            .await
            .context("Cannot initiate chat stream request")?;

        Ok(response.into_inner())
    }

    fn get_grpc_conversation(&self, interactions: Option<Vec<InteractionMessage>>) -> Conversation {
        use crate::collection_manager::dto::Role as DtoRole;

        if let Some(interactions) = interactions {
            let messages = interactions
                .iter()
                .map(|message| {
                    let role = match message.role {
                        DtoRole::User => 0,
                        DtoRole::Assistant => 1,
                        DtoRole::System => 2,
                    };

                    ConversationMessage {
                        role,
                        content: message.content.clone(),
                    }
                })
                .collect();

            Conversation { messages }
        } else {
            Conversation { messages: vec![] }
        }
    }
}

pin_project! {
    pub struct ChatStream {
        #[pin]
        inner: Streaming<ChatStreamResponse>,
        end_cb: std::sync::RwLock<Option<Box<dyn FnOnce() + Send>>>,
    }
}
impl Stream for ChatStream {
    type Item = Result<ChatStreamResponse, Status>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let a = this.inner.poll_next(cx);

        match a {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                let end_cb = this.end_cb.get_mut().unwrap();
                if let Some(end_cb) = end_cb.take() {
                    end_cb();
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(a))) => Poll::Ready(Some(Ok(a))),
            Poll::Ready(Some(Err(e))) => {
                let end_cb = this.end_cb.get_mut().unwrap();
                if let Some(end_cb) = end_cb.take() {
                    end_cb();
                }
                Poll::Ready(Some(Err(e)))
            }
        }
    }
}

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
            .unwrap();

        info!("Connecting to gRPC");
        let endpoint: tonic::transport::Channel = tonic::transport::Endpoint::new(uri.clone())?
            .connect()
            .await
            .with_context(move || format!("Cannot connect to {:?}", uri))?;
        info!("Connected to gRPC");

        let client: LlmServiceClient<tonic::transport::Channel> = LlmServiceClient::new(endpoint);

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
