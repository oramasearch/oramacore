use std::sync::Arc;

use oramacore_lib::nlp::NLPService;
use tokio::sync::mpsc::Sender;

use crate::{
    ai::{automatic_embeddings_selector::AutomaticEmbeddingsSelector, llms::LLMService, AIService},
    collection_manager::sides::{
        write::embedding::MultiEmbeddingCalculationRequest, OperationSender,
    },
};

#[derive(Clone)]
pub struct WriteSideContext {
    pub ai_service: Arc<AIService>,
    pub embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
    pub op_sender: OperationSender,
    pub nlp_service: Arc<NLPService>,
    pub automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
    pub llm_service: Arc<LLMService>,
}
