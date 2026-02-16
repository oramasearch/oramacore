use std::sync::Arc;

use oramacore_lib::nlp::NLPService;
use tokio::sync::mpsc::Sender;

use crate::{
    ai::{automatic_embeddings_selector::AutomaticEmbeddingsSelector, llms::LLMService},
    collection_manager::sides::{
        write::{document_storage::DocumentStorage, embedding::MultiEmbeddingCalculationRequest},
        OperationSender,
    },
    embeddings::EmbeddingsService,
    HooksConfig,
};

#[derive(Clone)]
pub struct WriteSideContext {
    pub embeddings_service: Arc<EmbeddingsService>,
    pub embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
    pub op_sender: OperationSender,
    pub nlp_service: Arc<NLPService>,
    pub automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
    pub llm_service: Arc<LLMService>,
    pub global_document_storage: Arc<DocumentStorage>,
    pub hooks_config: Arc<HooksConfig>,
}
