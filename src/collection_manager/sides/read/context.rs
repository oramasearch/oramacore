use std::sync::Arc;

use oramacore_lib::nlp::NLPService;

use crate::{
    ai::{llms::LLMService, AIService},
    collection_manager::sides::read::notify::Notifier,
    python::embeddings::EmbeddingsService,
};

#[derive(Clone)]
pub struct ReadSideContext {
    pub embeddings_service: Arc<EmbeddingsService>,
    pub nlp_service: Arc<NLPService>,
    pub llm_service: Arc<LLMService>,
    pub notifier: Option<Arc<Notifier>>,
}
