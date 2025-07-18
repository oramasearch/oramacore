use std::sync::Arc;

use nlp::NLPService;

use crate::{
    ai::{llms::LLMService, AIService},
    collection_manager::sides::read::notify::Notifier,
};

#[derive(Clone)]
pub struct ReadSideContext {
    pub ai_service: Arc<AIService>,
    pub nlp_service: Arc<NLPService>,
    pub llm_service: Arc<LLMService>,
    pub notifier: Option<Arc<Notifier>>,
}
