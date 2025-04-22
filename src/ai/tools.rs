use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result};
use async_openai::types::{ChatCompletionRequestMessage, FunctionObject, FunctionObjectArgs};
use orama_js_pool::{JSExecutorConfig, JSExecutorPoolConfig, OramaJSPool, OramaJSPoolConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    collection_manager::sides::generic_kv::KV,
    types::{CollectionId, InteractionLLMConfig, InteractionMessage},
};

use super::llms::LLMService;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolExecutionResult {
    pub tool_id: String,
    pub result: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ToolExecutionReturnType {
    FunctionParameters(ToolExecutionResult),
    FunctionResult(ToolExecutionResult),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tool {
    pub id: String,
    pub description: String,
    pub parameters: String, // @todo: check if we want to have a serde_json::Value here
    pub code: Option<String>, // Option here since we may just want to return the parameters, not necessarily execute the code on the server-side.
}

impl Tool {
    pub fn new(id: String, description: String, parameters: String, code: Option<String>) -> Self {
        Tool {
            id,
            description,
            parameters,
            code,
        }
    }

    pub fn to_openai_tool(&self) -> Result<FunctionObject> {
        let function_params: serde_json::Value =
            serde_json::from_str(&self.parameters).context(format!(
                "Cannot deserialize parameters for tool {}. Not a valid JSON.",
                &self.id
            ))?;

        Ok(FunctionObjectArgs::default()
            .name(&self.id)
            .description(&self.description)
            .parameters(function_params)
            .build()?)
    }
}

pub struct ToolsRuntime {
    pub kv: Arc<KV>,
    pub llm_service: Arc<LLMService>,
    pub tools_js_runtime: OramaJSPool<Value, Value>,
}

impl ToolsRuntime {
    pub fn new(kv: Arc<KV>, llm_service: Arc<LLMService>) -> Self {
        let tools_js_runtime = OramaJSPool::new(OramaJSPoolConfig {
            check_interval: Duration::from_secs(5), // @todo: make it configurable
            max_idle_time: Duration::from_secs(5),  // @todo: make it configurable
            pool_config: JSExecutorPoolConfig {
                instances_count_per_code: 3usize, // @todo: make it configurable
                queue_capacity: 10usize,          // @todo: make it configurable
                executor_config: JSExecutorConfig {
                    allowed_hosts: vec![],
                    function_name: "oramacore_tools".to_string(),
                    is_async: true,
                    max_execution_time: Duration::from_secs(5), // @todo: make it configurable
                    max_startup_time: Duration::from_millis(500), // @todo: make it configurable
                },
            },
        });

        ToolsRuntime {
            kv,
            llm_service,
            tools_js_runtime,
        }
    }

    pub async fn insert(&self, collection_id: CollectionId, tool: Tool) -> Result<()> {
        let key = self.format_key(collection_id, &tool.id);

        // Since we use function names as keys, it may be easier to unintentionally overwrite a tool.
        // Users should delete the tool first and then insert it again (or update an existing one).
        if let Some(_existing_tool) = self.get(collection_id.clone(), tool.id.clone()).await? {
            anyhow::bail!("Tool with id {} already exists", tool.id);
        }

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
    ) -> Result<Option<Vec<ToolExecutionReturnType>>> {
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

        match chosen_tools {
            // There might be the case when the LLM doesn't choose any tool, in that case we return None
            // to indicate that no tool was executed.
            Some(chosen_tools) => {
                // We can have two kinds of tools:
                // 1. Tools that have code and we need to execute it on the server-side
                // 2. Tools that don't have code and we just need to return the parameters
                // In the first case we need to execute the code and return the result
                // In the second case we just need to return the parameters
                // We can have multiple tools chosen, so we need to iterate over them
                // and execute the code for each one of them
                let mut results: Vec<ToolExecutionReturnType> = Vec::new();

                for tool in chosen_tools.iter() {
                    let full_tool = self.get(collection_id, tool.name.clone()).await?;

                    // Since we got the tools from the KV in the first place, this should never be None.
                    // But I don't trust myself enough to use `unwrap` here. Tommaso you should be proud.
                    let full_tool = match full_tool {
                        Some(tool) => tool,
                        None => anyhow::bail!("Tool with id {} not found", tool.name),
                    };

                    // As said before, some tools may have some code to execute, while others may just have parameters.
                    // If there is some code, we will execute it and push the result to the results vector.
                    if let Some(code) = full_tool.code {
                        // LLMs will typically return the arguments as a JSON string, so we should be fine with deserializing it.
                        // But as a general rule, I'd prefer to use something like `json-repair` to ensure the JSON integrity.
                        // @todo: check if we want to use json-repair here
                        let arguments_as_json_value = serde_json::from_str(&tool.arguments)
                            .context(format!(
                                "Cannot deserialize arguments for tool {}. Not a valid JSON.",
                                &tool.name
                            ))?;

                        // Let's call the function with the deserialized arguments.
                        // We should assume that it'll always return a JSON object or a valid JSON string.
                        let function_call = self
                            .tools_js_runtime
                            .execute(&code, arguments_as_json_value)
                            .await;

                        // If the function call was successful, we push the result to the results vector.
                        // For now, we'll bail if the function call fails.
                        // @todo: handle this error more gracefully.
                        if let Ok(result) = function_call {
                            results.push(ToolExecutionReturnType::FunctionResult(
                                ToolExecutionResult {
                                    tool_id: tool.name.clone(),
                                    result: result.to_string(),
                                },
                            ));
                        } else {
                            anyhow::bail!(
                                "Error executing tool {}: {}",
                                &tool.name,
                                function_call.unwrap_err()
                            );
                        }
                    } else {
                        // Case when we just need to return the parameters.
                        results.push(ToolExecutionReturnType::FunctionParameters(
                            ToolExecutionResult {
                                tool_id: tool.name.clone(),
                                result: tool.arguments.clone(),
                            },
                        ));
                    }
                }
                Ok(Some(results))
            }
            None => Ok(None),
        }
    }

    fn format_key(&self, collection_id: CollectionId, tool_id: &str) -> String {
        format!("{}:tool:{}", collection_id, tool_id)
    }
}
