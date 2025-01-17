use tokio::sync::RwLock;

use anyhow::Result;
use grpc::{ChatStreamResult, GrpcEmbeddingModel, GrpcLLM};
use tonic::Streaming;

use crate::ai::grpc::GrpcRepo;

pub mod grpc;

tonic::include_proto!("orama_ai_service");

#[derive(Debug)]
pub struct AiService {
    grpc: GrpcRepo,
    llm_cache: RwLock<Option<GrpcLLM>>,
}

impl AiService {
    pub fn new(grpc: GrpcRepo) -> Self {
        Self {
            grpc,
            llm_cache: Default::default(),
        }
    }

    pub async fn load_model(&self, model_name: String) -> Result<GrpcEmbeddingModel> {
        self.grpc.load_model(model_name).await
    }

    pub async fn chat(
        &self,
        llm_type: LlmType,
        prompt: String,
        conversation: Conversation,
        context: Option<String>,
    ) -> Result<ChatResponse> {
        let lock = self.llm_cache.read().await;

        if let Some(llm) = &*lock {
            return llm.chat(llm_type, prompt, conversation, context).await;
        }

        drop(lock);

        let mut cache_w = self.llm_cache.write().await;
        *cache_w = Some(self.grpc.load_llm().await?);

        drop(cache_w);

        let lock = self.llm_cache.read().await;

        let llm = lock.as_ref();

        llm.unwrap()
            .chat(llm_type, prompt, conversation, context)
            .await
    }

    pub async fn chat_stream(
        &self,
        llm_type: LlmType,
        prompt: String,
        conversation: Conversation,
        context: Option<String>,
    ) -> Result<Streaming<ChatStreamResponse>> {
        let lock = self.llm_cache.read().await;

        if let Some(llm) = &*lock {
            return llm
                .chat_stream(llm_type, prompt, conversation, context)
                .await;
        }

        drop(lock);

        let mut cache_w = self.llm_cache.write().await;
        *cache_w = Some(self.grpc.load_llm().await?);

        drop(cache_w);

        let lock = self.llm_cache.read().await;

        let llm = lock.as_ref();

        llm.unwrap()
            .chat_stream(llm_type, prompt, conversation, context)
            .await
    }
}
