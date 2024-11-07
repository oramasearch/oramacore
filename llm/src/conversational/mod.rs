use crate::conversational::prompts::default_system_prompt;
use crate::LocalLLM;
use anyhow::Result;
use mistralrs::{Model, TextMessageRole, TextMessages};
use serde_json::Value;
use std::sync::Arc;

mod prompts;

#[derive(Clone)]
pub struct Segment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub goal: String,
}

#[derive(Clone)]
pub struct Trigger {
    pub id: String,
    pub name: String,
    pub description: String,
    pub response: String,
}

#[derive(Clone)]
pub struct GenerativeTextOptions {
    pub segment: Option<Segment>,
    pub trigger: Option<Trigger>,
    pub user_context: Option<Value>,
}

pub struct AnswerSession {
    pub options: GenerativeTextOptions,
    pub system_prompt: String,
    pub messages: TextMessages,
    pub llm: Arc<Model>,
}

impl AnswerSession {
    pub async fn try_new(options: GenerativeTextOptions) -> Result<Self> {
        let llm = LocalLLM::Phi3_5MiniInstruct.try_new().await?;
        let system_prompt = default_system_prompt(options.clone());

        Ok(AnswerSession {
            llm,
            options,
            system_prompt: system_prompt.clone(),
            messages: TextMessages::new().add_message(TextMessageRole::System, system_prompt),
        })
    }
}
