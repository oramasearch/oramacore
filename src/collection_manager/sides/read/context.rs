use std::sync::Arc;

use oramacore_lib::nlp::NLPService;

use crate::{
    ai::llms::LLMService,
    collection_manager::sides::read::{document_storage::DocumentStorage, notify::Notifier},
    embeddings::EmbeddingsService,
    HooksConfig,
};

#[derive(Clone)]
pub struct ReadSideContext {
    pub embeddings_service: Arc<EmbeddingsService>,
    pub nlp_service: Arc<NLPService>,
    pub llm_service: Arc<LLMService>,
    pub notifier: Option<Arc<Notifier>>,
    pub global_document_storage: Arc<DocumentStorage>,
    pub hooks_config: Arc<HooksConfig>,
}
