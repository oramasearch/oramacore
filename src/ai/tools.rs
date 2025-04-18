use std::sync::Arc;

use anyhow::{Context, Result};
use async_openai::types::{
    ChatCompletionRequestMessage, FunctionCall, FunctionObject, FunctionObjectArgs,
};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::generic_kv::KV,
    types::{CollectionId, InteractionLLMConfig, InteractionMessage},
};

use super::llms::LLMService;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tool {
    pub id: String,
    pub name: String,
    pub description: String,
    pub parameters: String, // @todo: check if we want to have a serde_json::Value here
}

impl Tool {
    pub fn new(id: String, name: String, description: String, parameters: String) -> Self {
        Tool {
            id,
            name,
            description,
            parameters,
        }
    }

    pub fn to_openai_tool(&self) -> Result<FunctionObject> {
        let function_params = serde_json::to_value(&self.parameters).context(format!(
            "Cannot parameters for tool {}. Not a valid JSON.",
            &self.id
        ))?;

        Ok(FunctionObjectArgs::default()
            .name(&self.name)
            .description(&self.description)
            .parameters(function_params)
            .build()?)
    }
}

pub struct ToolsInterface {
    pub kv: Arc<KV>,
    pub llm_service: Arc<LLMService>,
}

impl ToolsInterface {
    pub fn new(kv: Arc<KV>, llm_service: Arc<LLMService>) -> Self {
        ToolsInterface { kv, llm_service }
    }

    pub async fn insert(&self, collection_id: CollectionId, tool: Tool) -> Result<()> {
        let key = self.format_key(collection_id, &tool.id);

        self.kv.insert(key, tool).await?;

        Ok(())
    }

    pub async fn get(&self, collection_id: CollectionId, tool_id: String) -> Result<Option<Tool>> {
        let key = self.format_key(collection_id, &tool_id);

        match self.kv.get::<Tool>(&key).await {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(tool)) => Ok(Some(tool)),
        }
    }

    pub async fn delete(&self, collection_id: CollectionId, tool_id: String) -> Result<()> {
        let key = self.format_key(collection_id, &tool_id);

        self.kv.remove(&key).await?;

        Ok(())
    }

    pub async fn has_tools(&self, collection_id: CollectionId) -> Result<bool> {
        let tools = self.list_by_collection(collection_id).await?;

        Ok(!tools.is_empty())
    }

    pub async fn list_by_collection(&self, collection_id: CollectionId) -> Result<Vec<Tool>> {
        let prefix = format!("{}:tool:", collection_id.as_str());

        let segments = self.kv.prefix_scan(&prefix).await.context(format!(
            "Cannot scan tools for collection {}",
            collection_id.as_str()
        ))?;

        Ok(segments)
    }

    pub async fn execute_tools(
        &self,
        collection_id: CollectionId,
        messages: Vec<InteractionMessage>,
        tool_ids: Option<Vec<String>>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<Vec<FunctionCall>>> {
        let tools = match tool_ids {
            Some(tool_ids) => {
                let mut tools = Vec::new();
                for tool_id in tool_ids {
                    match self.get(collection_id.clone(), tool_id.clone()).await? {
                        Some(tool) => tools.push(tool),
                        None => anyhow::bail!("Tool with id {} not found", tool_id),
                    }
                }
                tools
            }
            None => {
                let all_tools = self.list_by_collection(collection_id).await?;

                if all_tools.is_empty() {
                    anyhow::bail!("No tools found for collection {}", collection_id.as_str());
                }

                all_tools
            }
        };

        let tools_to_function_objects: Vec<FunctionObject> = tools
            .iter()
            .map(|tool| tool.to_openai_tool())
            .collect::<Result<Vec<_>>>()?;

        let conversation: Vec<ChatCompletionRequestMessage> = messages
            .iter()
            .map(|m| m.to_async_openai_message())
            .collect();

        let chosen_tools = self
            .llm_service
            .execute_tools(conversation, tools_to_function_objects, llm_config)
            .await?;

        Ok(chosen_tools)
    }

    fn format_key(&self, collection_id: CollectionId, tool_id: &str) -> String {
        format!("{}:tool:{}", collection_id, tool_id)
    }
}
