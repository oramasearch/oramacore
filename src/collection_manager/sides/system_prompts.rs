use anyhow::{Context, Result};
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    ai::llms::{KnownPrompts, LLMService},
    collection_manager::sides::write::WriteError,
    types::{CollectionId, InteractionLLMConfig, SystemPromptUsageMode},
};

use super::generic_kv::{format_key, KV};
use llm_json::repair_json;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPrompt {
    pub id: String,
    pub name: String,
    pub usage_mode: SystemPromptUsageMode,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptValidationSecurity {
    pub valid: bool,
    pub reason: String,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptValidationTechnical {
    valid: bool,
    reason: String,
    instruction_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptValidationOverall {
    valid: bool,
    summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptValidationResponse {
    pub security: SystemPromptValidationSecurity,
    pub technical: SystemPromptValidationTechnical,
    pub overall_assessment: SystemPromptValidationOverall,
}

#[derive(Clone)]
pub struct SystemPromptInterface {
    kv: Arc<KV>,
    llm_service: Arc<LLMService>,
}

impl SystemPromptInterface {
    pub fn new(kv: Arc<KV>, llm_service: Arc<LLMService>) -> Self {
        Self { kv, llm_service }
    }

    pub async fn validate_system_prompt(
        &self,
        system_prompt: SystemPrompt,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<SystemPromptValidationResponse, WriteError> {
        let r = self.validate_prompt(system_prompt, llm_config).await?;

        Ok(r)
    }

    pub async fn insert_system_prompt(
        &self,
        collection_id: CollectionId,
        system_prompt: SystemPrompt,
    ) -> Result<(), WriteError> {
        self.insert(collection_id, system_prompt.clone())
            .await
            .context("Cannot insert system prompt")?;

        Ok(())
    }

    pub async fn delete_system_prompt(
        &self,
        collection_id: CollectionId,
        system_prompt_id: String,
    ) -> Result<Option<SystemPrompt>, WriteError> {
        let r = self
            .delete(collection_id, system_prompt_id.clone())
            .await
            .context("Cannot delete system prompt")?;
        Ok(r)
    }

    pub async fn update_system_prompt(
        &self,
        collection_id: CollectionId,
        system_prompt: SystemPrompt,
    ) -> Result<(), WriteError> {
        self.delete(collection_id, system_prompt.id.clone())
            .await
            .context("Cannot delete system prompt")?;
        self.insert(collection_id, system_prompt)
            .await
            .context("Cannot insert system prompt")?;

        Ok(())
    }

    pub async fn get(
        &self,
        collection_id: CollectionId,
        system_prompt_id: String,
    ) -> Result<Option<SystemPrompt>> {
        let system_prompt_key = format!("system_prompt:{system_prompt_id}");
        let key = format_key(collection_id, &system_prompt_key);

        match self.kv.get(&key).await {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(system_prompt)) => Ok(Some(system_prompt)),
        }
    }

    async fn validate_prompt(
        &self,
        system_prompt: SystemPrompt,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<SystemPromptValidationResponse> {
        let variables = vec![("input".to_string(), system_prompt.prompt)];
        let response = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::ValidateSystemPrompt,
                vec![],
                variables,
                None,
                llm_config,
            )
            .await?;

        let repaired = repair_json(&response, &Default::default())?;
        let deserialized: SystemPromptValidationResponse = serde_json::from_str(&repaired)?;

        Ok(deserialized)
    }

    async fn insert(&self, collection_id: CollectionId, system_prompt: SystemPrompt) -> Result<()> {
        let key = format_key(
            collection_id,
            &format!("system_prompt:{}", system_prompt.id),
        );
        self.kv.insert(key, system_prompt).await?;
        Ok(())
    }

    async fn delete(
        &self,
        collection_id: CollectionId,
        system_prompt_id: String,
    ) -> Result<Option<SystemPrompt>> {
        let system_prompt_key = format!("system_prompt:{system_prompt_id}");
        let key = format_key(collection_id, &system_prompt_key);

        match self.kv.remove_and_get(&key).await? {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(system_prompt)) => Ok(Some(system_prompt)),
        }
    }

    pub async fn has_system_prompts(&self, collection_id: CollectionId) -> Result<bool> {
        let system_prompts = self.list_by_collection(collection_id).await?;

        Ok(!system_prompts.is_empty())
    }

    pub async fn list_by_collection(
        &self,
        collection_id: CollectionId,
    ) -> Result<Vec<SystemPrompt>> {
        let prefix = format!("{}:system_prompt:", collection_id.as_str());

        let system_prompts = self.kv.prefix_scan(&prefix).await.context(format!(
            "Cannot scan system prompts for collection {}",
            collection_id.as_str()
        ))?;

        Ok(system_prompts)
    }

    pub async fn perform_system_prompt_selection(
        &self,
        collection_id: CollectionId,
    ) -> Result<Option<SystemPrompt>> {
        let system_prompts = self.list_by_collection(collection_id).await?;

        if system_prompts.is_empty() {
            return Ok(None);
        }

        let mut rng = rand::rng();

        let chosen = system_prompts.choose(&mut rng);

        Ok(chosen.cloned())
    }
}

pub struct CollectionSystemPromptsInterface {
    interface: SystemPromptInterface,
    collection_id: CollectionId,
}

impl CollectionSystemPromptsInterface {
    pub fn new(interface: SystemPromptInterface, collection_id: CollectionId) -> Self {
        Self {
            interface,
            collection_id,
        }
    }

    pub async fn validate_system_prompt(
        &self,
        system_prompt: SystemPrompt,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<SystemPromptValidationResponse, WriteError> {
        self.interface
            .validate_system_prompt(system_prompt, llm_config)
            .await
    }

    pub async fn insert_system_prompt(
        &self,
        system_prompt: SystemPrompt,
    ) -> Result<(), WriteError> {
        self.interface
            .insert_system_prompt(self.collection_id, system_prompt)
            .await
    }

    pub async fn delete_system_prompt(
        &self,
        system_prompt_id: String,
    ) -> Result<Option<SystemPrompt>, WriteError> {
        self.interface
            .delete_system_prompt(self.collection_id, system_prompt_id)
            .await
    }

    pub async fn update_system_prompt(
        &self,
        system_prompt: SystemPrompt,
    ) -> Result<(), WriteError> {
        self.interface
            .update_system_prompt(self.collection_id, system_prompt)
            .await
    }

    pub async fn get(&self, system_prompt_id: String) -> Result<Option<SystemPrompt>> {
        self.interface
            .get(self.collection_id, system_prompt_id)
            .await
    }

    pub async fn has_system_prompts(&self) -> Result<bool> {
        self.interface.has_system_prompts(self.collection_id).await
    }

    pub async fn list_by_collection(&self) -> Result<Vec<SystemPrompt>> {
        self.interface.list_by_collection(self.collection_id).await
    }

    pub async fn perform_system_prompt_selection(&self) -> Result<Option<SystemPrompt>> {
        self.interface
            .perform_system_prompt_selection(self.collection_id)
            .await
    }
}
