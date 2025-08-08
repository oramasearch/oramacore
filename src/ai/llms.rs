use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolType,
        CreateChatCompletionRequestArgs, FunctionCall, FunctionObject,
    },
};
use futures::{Stream, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

use crate::types::{InteractionLLMConfig, RelatedRequest, SuggestionsRequest};
use crate::{collection_manager::sides::system_prompts::SystemPrompt, types::RelatedQueriesFormat};

use super::{AIServiceLLMConfig, RemoteLLMProvider, RemoteLLMsConfig};
use crate::ai::gpu::LocalGPUManager;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnownPrompts {
    Answer,
    Autoquery,
    AdvancedAutoqueryQueryAnalyzer,
    AdvancedAutoQueryPropertiesSelector,
    AdvancedAutoQueryQueryComposer,
    AutomaticEmbeddingsSelector,
    OptimizeQuery,
    ValidateSystemPrompt,
    Followup,
    Suggestions,
    GenerateRelatedQueries,
    DetermineQueryStrategy,
}

#[derive(Debug, Clone)]
pub struct KnownPrompt {
    pub system: String,
    pub user: String,
}

impl TryFrom<&str> for KnownPrompts {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "ANSWER" => Ok(Self::Answer),
            "AUTOQUERY" => Ok(Self::Autoquery),
            "ADVANCED_AUTOQUERY_QUERY_ANALYZER" => Ok(Self::AdvancedAutoqueryQueryAnalyzer),
            "ADVANCED_AUTOQUERY_PROPERTIES_SELECTOR" => {
                Ok(Self::AdvancedAutoQueryPropertiesSelector)
            }
            "ADVANCED_AUTOQUERY_QUERY_COMPOSER" => Ok(Self::AdvancedAutoQueryQueryComposer),
            "AUTOMATIC_EMBEDDINGS_SELECTOR" => Ok(Self::AutomaticEmbeddingsSelector),
            "OPTIMIZE_QUERY" => Ok(Self::OptimizeQuery),
            "VALIDATE_SYSTEM_PROMPT" => Ok(Self::ValidateSystemPrompt),
            "FOLLOWUP" => Ok(Self::Followup),
            "GENERATE_RELATED_QUERIES" => Ok(Self::GenerateRelatedQueries),
            "SUGGESTIONS" => Ok(Self::Suggestions),
            "DETERMINE_QUERY_STRATEGY" => Ok(Self::DetermineQueryStrategy),
            _ => Err(format!("Unknown prompt type: {s}")),
        }
    }
}

impl KnownPrompts {
    pub fn get_prompts(&self) -> KnownPrompt {
        match self {
            KnownPrompts::Answer => KnownPrompt {
                system: include_str!("../prompts/v1/answer/system.md").to_string(),
                user: include_str!("../prompts/v1/answer/user.md").to_string(),
            },
            KnownPrompts::Autoquery => KnownPrompt {
                system: include_str!("../prompts/v1/autoquery/system.md").to_string(),
                user: include_str!("../prompts/v1/autoquery/user.md").to_string(),
            },
            KnownPrompts::OptimizeQuery => KnownPrompt {
                system: include_str!("../prompts/v1/optimize_query/system.md").to_string(),
                user: include_str!("../prompts/v1/optimize_query/user.md").to_string(),
            },
            KnownPrompts::ValidateSystemPrompt => KnownPrompt {
                system: include_str!("../prompts/v1/custom_system_prompt_validator/system.md")
                    .to_string(),
                user: include_str!("../prompts/v1/custom_system_prompt_validator/user.md")
                    .to_string(),
            },
            KnownPrompts::Followup => KnownPrompt {
                system: include_str!("../prompts/v1/party_planner/actions/ask_followup_system.md")
                    .to_string(),
                user: include_str!("../prompts/v1/party_planner/actions/ask_followup_user.md")
                    .to_string(),
            },
            KnownPrompts::GenerateRelatedQueries => KnownPrompt {
                system: include_str!("../prompts/v1/related_queries/system.md").to_string(),
                user: include_str!("../prompts/v1/related_queries/user.md").to_string(),
            },
            KnownPrompts::AutomaticEmbeddingsSelector => KnownPrompt {
                system: include_str!("../prompts/v1/automatic_embeddings_selector/system.md")
                    .to_string(),
                user: include_str!("../prompts/v1/automatic_embeddings_selector/user.md")
                    .to_string(),
            },
            KnownPrompts::AdvancedAutoqueryQueryAnalyzer => KnownPrompt {
                system: include_str!("../prompts/v1/advanced_autoquery/query_analyzer/system.md")
                    .to_string(),
                user: include_str!("../prompts/v1/advanced_autoquery/query_analyzer/user.md")
                    .to_string(),
            },
            KnownPrompts::AdvancedAutoQueryPropertiesSelector => KnownPrompt {
                system: include_str!(
                    "../prompts/v1/advanced_autoquery/properties_selector/system.md"
                )
                .to_string(),
                user: include_str!("../prompts/v1/advanced_autoquery/properties_selector/user.md")
                    .to_string(),
            },
            KnownPrompts::AdvancedAutoQueryQueryComposer => KnownPrompt {
                system: include_str!("../prompts/v1/advanced_autoquery/query_composer/system.md")
                    .to_string(),
                user: include_str!("../prompts/v1/advanced_autoquery/query_composer/user.md")
                    .to_string(),
            },
            KnownPrompts::DetermineQueryStrategy => KnownPrompt {
                system: include_str!("../prompts/v1/determine_query_strategy/system.md")
                    .to_string(),
                user: include_str!("../prompts/v1/determine_query_strategy/user.md").to_string(),
            },
            KnownPrompts::Suggestions => KnownPrompt {
                system: include_str!("../prompts/v1/suggestions/system.md").to_string(),
                user: include_str!("../prompts/v1/suggestions/user.md").to_string(),
            },
        }
    }
}

pub fn format_prompt(prompt: String, variables: HashMap<String, String>) -> String {
    let mut result = prompt.to_string();
    for (key, value) in variables {
        result = result.replace(&format!("{{{key}}}"), &value);
    }
    result
}

#[derive(Debug)]
pub struct LLMService {
    pub local_vllm_client: Option<async_openai::Client<OpenAIConfig>>,
    pub unified_remote_client: Option<async_openai::Client<OpenAIConfig>>,
    pub remote_clients: Option<HashMap<RemoteLLMProvider, async_openai::Client<OpenAIConfig>>>,
    pub model: String,
    pub default_remote_models: Option<HashMap<RemoteLLMProvider, String>>,
    pub local_gpu_manager: Arc<LocalGPUManager>,
    pub is_unified_remote: bool,
}

impl LLMService {
    pub fn try_new(
        llm_config: AIServiceLLMConfig,
        remote_llm_config: Option<Vec<RemoteLLMsConfig>>,
        local_gpu_manager: Arc<LocalGPUManager>,
    ) -> Result<Self> {
        let is_unified_remote = !llm_config.local;

        let (local_vllm_client, unified_remote_client) = if is_unified_remote {
            // Remote configuration - use host as the API base URL
            info!(
                "Configuring unified remote LLM: {} with model {}",
                llm_config.host, llm_config.model
            );
            let mut config = OpenAIConfig::new().with_api_base(&llm_config.host);
            if !llm_config.api_key.is_empty() {
                config = config.with_api_key(&llm_config.api_key);
            }
            (None, Some(async_openai::Client::with_config(config)))
        } else {
            // Local configuration - construct URL with port
            let local_vllm_provider_url = format!(
                "http://{}:{}/v1",
                llm_config.host,
                llm_config.port.unwrap_or(8000)
            );
            info!(
                "Configuring local LLM: {} with model {}",
                local_vllm_provider_url, llm_config.model
            );
            let mut config = OpenAIConfig::new().with_api_base(&local_vllm_provider_url);
            if !llm_config.api_key.is_empty() {
                config = config.with_api_key(&llm_config.api_key);
            }
            (Some(async_openai::Client::with_config(config)), None)
        };

        let mut remote_llm_providers: HashMap<
            RemoteLLMProvider,
            async_openai::Client<OpenAIConfig>,
        > = HashMap::new();

        let mut default_remote_models: HashMap<RemoteLLMProvider, String> = HashMap::new();

        if let Some(remote_config) = remote_llm_config {
            for conf in remote_config {
                match conf.provider {
                    RemoteLLMProvider::OpenAI => {
                        info!("Found OpenAI remote LLM provider");

                        match conf.default_model.as_str() {
                            "" => {
                                return Err(anyhow::Error::msg(
                                    "Default model is required for OpenAI provider",
                                ));
                            }
                            _ => {
                                default_remote_models
                                    .insert(RemoteLLMProvider::OpenAI, conf.default_model.clone());
                            }
                        }

                        remote_llm_providers.insert(
                            RemoteLLMProvider::OpenAI,
                            async_openai::Client::with_config(
                                OpenAIConfig::new()
                                    .with_api_key(&conf.api_key)
                                    .with_api_base(conf.url.unwrap_or_else(|| {
                                        "https://api.openai.com/v1".to_string()
                                    })),
                            ),
                        );
                    }
                    RemoteLLMProvider::Groq => {
                        info!("Found Groq remote LLM provider");

                        match conf.default_model.as_str() {
                            "" => {
                                return Err(anyhow::Error::msg(
                                    "Default model is required for Groq provider",
                                ));
                            }
                            _ => {
                                default_remote_models
                                    .insert(RemoteLLMProvider::Groq, conf.default_model.clone());
                            }
                        }

                        remote_llm_providers.insert(
                            RemoteLLMProvider::Groq,
                            async_openai::Client::with_config(
                                OpenAIConfig::new()
                                    .with_api_key(&conf.api_key)
                                    .with_api_base(conf.url.unwrap_or_else(|| {
                                        "https://api.groq.com/openai/v1".to_string()
                                    })),
                            ),
                        );
                    }
                    RemoteLLMProvider::Anthropic => {
                        info!("Found Anthropic remote LLM provider");

                        match conf.default_model.as_str() {
                            "" => {
                                return Err(anyhow::Error::msg(
                                    "Default model is required for Anthropic provider",
                                ));
                            }
                            _ => {
                                default_remote_models.insert(
                                    RemoteLLMProvider::Anthropic,
                                    conf.default_model.clone(),
                                );
                            }
                        }

                        remote_llm_providers.insert(
                            RemoteLLMProvider::Anthropic,
                            async_openai::Client::with_config(
                                OpenAIConfig::new()
                                    .with_api_key(&conf.api_key)
                                    .with_api_base(conf.url.unwrap_or_else(|| {
                                        "https://api.anthropic.com/v1/".to_string()
                                    })),
                            ),
                        );
                    }
                    RemoteLLMProvider::Fireworks => {
                        info!("Found Fireworks remote LLM provider");

                        match conf.default_model.as_str() {
                            "" => {
                                return Err(anyhow::Error::msg(
                                    "Default model is required for Fireworks provider",
                                ));
                            }
                            _ => {
                                default_remote_models.insert(
                                    RemoteLLMProvider::Fireworks,
                                    conf.default_model.clone(),
                                );
                            }
                        }

                        remote_llm_providers.insert(
                            RemoteLLMProvider::Fireworks,
                            async_openai::Client::with_config(
                                OpenAIConfig::new()
                                    .with_api_key(&conf.api_key)
                                    .with_api_base(conf.url.unwrap_or_else(|| {
                                        "https://api.fireworks.ai/inference/v1".to_string()
                                    })),
                            ),
                        );
                    }
                    RemoteLLMProvider::Together => {
                        info!("Found Together remote LLM provider");

                        match conf.default_model.as_str() {
                            "" => {
                                return Err(anyhow::Error::msg(
                                    "Default model is required for Together provider",
                                ));
                            }
                            _ => {
                                default_remote_models.insert(
                                    RemoteLLMProvider::Together,
                                    conf.default_model.clone(),
                                );
                            }
                        }

                        remote_llm_providers.insert(
                            RemoteLLMProvider::Together,
                            async_openai::Client::with_config(
                                OpenAIConfig::new()
                                    .with_api_key(&conf.api_key)
                                    .with_api_base(conf.url.unwrap_or_else(|| {
                                        "https://api.together.xyz/v1".to_string()
                                    })),
                            ),
                        );
                    }
                    RemoteLLMProvider::GoogleVertex => {
                        info!("Found Google Vertex remote LLM provider");

                        match conf.default_model.as_str() {
                            "" => {
                                return Err(anyhow::Error::msg(
                                    "Default model is required for Google Vertex provider",
                                ));
                            }
                            _ => {
                                default_remote_models.insert(
                                    RemoteLLMProvider::GoogleVertex,
                                    conf.default_model.clone(),
                                );
                            }
                        }

                        remote_llm_providers.insert(
                            RemoteLLMProvider::GoogleVertex,
                            async_openai::Client::with_config(
                                OpenAIConfig::new()
                                    .with_api_key(&conf.api_key)
                                    .with_api_base(conf.url.unwrap_or_else(|| {
                                        "https://generativelanguage.googleapis.com/v1beta/openai"
                                            .to_string()
                                    })),
                            ),
                        );
                    }
                    #[allow(unreachable_patterns)]
                    _ => {
                        return Err(anyhow::Error::msg(format!(
                            "Unsupported remote LLM provider: {}",
                            conf.provider
                        )));
                    }
                }
            }
        }

        let remote_clients = match remote_llm_providers.len() {
            0 => None,
            _ => Some(remote_llm_providers),
        };

        let default_remote_models = match default_remote_models.len() {
            0 => None,
            _ => Some(default_remote_models),
        };

        Ok(Self {
            local_vllm_client,
            unified_remote_client,
            default_remote_models,
            remote_clients,
            model: llm_config.model,
            local_gpu_manager,
            is_unified_remote,
        })
    }

    pub async fn execute_tools(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: Vec<FunctionObject>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<Vec<FunctionCall>>> {
        let chosen_model = self.get_chosen_model(llm_config.clone());
        let llm_client = self.get_chosen_llm_client(llm_config);

        let all_tools: Vec<ChatCompletionTool> = tools
            .iter()
            .map(|tool| ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: tool.clone(),
            })
            .collect();

        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(2048u32)
            .temperature(0.0)
            .model(chosen_model)
            .messages(messages)
            .tools(all_tools)
            .build()?;

        let response_message = llm_client
            .chat()
            .create(request)
            .await?
            .choices
            .first()
            .unwrap()
            .message
            .clone();

        match response_message.tool_calls {
            Some(calls) => Ok(Some(
                calls.iter().map(|call| call.function.clone()).collect(),
            )),
            None => Ok(None),
        }
    }

    pub async fn run_known_prompt(
        &self,
        prompt: KnownPrompts,
        variables: Vec<(String, String)>,
        custom_system_prompt: Option<SystemPrompt>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<String> {
        let mut acc = String::new();
        let mut stream = self
            .run_known_prompt_stream(prompt, variables, custom_system_prompt, llm_config)
            .await
            .context("An error occurred while initializing the stream from remote LLM instance")?;

        while let Some(msg) = stream.next().await {
            match msg {
                Ok(m) => {
                    acc += &m;
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok(acc)
    }

    pub async fn run_known_prompt_stream(
        &self,
        prompt: KnownPrompts,
        variables: Vec<(String, String)>,
        custom_system_prompt: Option<SystemPrompt>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<impl Stream<Item = Result<String>>> {
        let mut prompts = prompt.get_prompts();
        let variables_map: HashMap<String, String> = HashMap::from_iter(variables);

        if let Some(system_propmpt) = custom_system_prompt {
            prompts.system += format!(
                "\n\n# Additional Information - FOLLOW STRICTLY\n\n{}",
                system_propmpt.prompt
            )
            .as_str();
        }

        let chosen_model = self.get_chosen_model(llm_config.clone());
        let llm_client = self.get_chosen_llm_client(llm_config);

        let request = CreateChatCompletionRequestArgs::default()
            .model(chosen_model)
            .stream(true)
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(prompts.system)
                    .build()
                    .unwrap()
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(format_prompt(prompts.user, variables_map))
                    .build()
                    .unwrap()
                    .into(),
            ])
            .build()
            .context("Unable to build KnownPrompt LLM request body")?;

        let mut response_stream =
            llm_client.chat().create_stream(request).await.context(
                "An error occurred while initializing the stream from remote LLM instance",
            )?;

        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(result) = response_stream.next().await {
                match result {
                    Ok(response) => {
                        let empty_str = &String::new();
                        let chunk = response
                            .choices
                            .first()
                            .unwrap()
                            .delta
                            .content
                            .as_ref()
                            .unwrap_or(empty_str);

                        tx.send(Ok(chunk.to_string())).await.unwrap();
                    }
                    Err(e) => {
                        let error_message = format!("An error occurred while processing prompt '{prompt:?}'. Response from the remote LLM instance: {e:?}");
                        tx.send(Err(anyhow::Error::msg(error_message)))
                            .await
                            .unwrap();
                    }
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }

    pub fn get_chosen_llm_client(
        &self,
        config: Option<InteractionLLMConfig>,
    ) -> async_openai::Client<OpenAIConfig> {
        // If config specifies a provider, use that remote client
        if let Some(config) = config {
            if let Some(remote_clients) = &self.remote_clients {
                if let Some(client) = remote_clients.get(&config.provider) {
                    return client.clone();
                }
            }
        }

        // If we're in unified remote mode, use the unified remote client
        if self.is_unified_remote {
            if let Some(unified_client) = &self.unified_remote_client {
                return unified_client.clone();
            }
        }

        // Otherwise use the local client
        if let Some(local_client) = &self.local_vllm_client {
            return local_client.clone();
        }

        // This should never happen if configuration is valid
        panic!("No LLM client available - configuration error")
    }

    pub fn get_chosen_model(&self, config: Option<InteractionLLMConfig>) -> String {
        if let Some(config) = config {
            return config.model;
        }

        self.model.clone()
    }

    pub fn get_suggestions_params(
        &self,
        suggestion_request: SuggestionsRequest,
    ) -> Vec<(String, String)> {
        vec![
            ("conversation".to_string(), "".to_string()),
            ("context".to_string(), "".to_string()),
            ("query".to_string(), suggestion_request.query),
            (
                "maxSuggestions".to_string(),
                suggestion_request.max_suggestions.unwrap_or(3).to_string(),
            ),
        ]
    }

    pub fn get_related_questions_params(
        &self,
        related: Option<RelatedRequest>,
    ) -> Vec<(String, String)> {
        if let Some(related_config) = related {
            if let Some(enabled) = related_config.enabled {
                if enabled {
                    let number = related_config.size.unwrap_or(3);
                    let question =
                        match related_config.format.unwrap_or(RelatedQueriesFormat::Query) {
                            RelatedQueriesFormat::Query => false,
                            RelatedQueriesFormat::Question => true,
                        };

                    return vec![
                        ("number".to_string(), number.to_string()),
                        ("question".to_string(), question.to_string()),
                    ];
                }
            }
        }

        Vec::new()
    }

    pub fn is_gpu_overloaded(&self) -> bool {
        // If we're using unified remote configuration, GPU overload is not relevant
        if self.is_unified_remote {
            return false;
        }

        match self.local_gpu_manager.is_overloaded() {
            Ok(overloaded) => overloaded,
            Err(e) => {
                error!(error = ?e, "Cannot check if GPU is overloaded. This may be due to GPU malfunction. Forcing inference on remote LLMs for safety.");
                true
            }
        }
    }

    pub fn get_available_remote_llm_services(&self) -> Option<HashMap<RemoteLLMProvider, String>> {
        self.default_remote_models.clone()
    }

    pub fn select_random_remote_llm_service(&self) -> Option<(RemoteLLMProvider, String)> {
        match self.get_available_remote_llm_services() {
            Some(services) => {
                let mut rng = rand::rng();
                let random_index = rand::Rng::random_range(&mut rng, 0..services.len());
                services.into_iter().nth(random_index)
            }
            None => {
                error!("No remote LLM services available. Unable to select a random one for handling a offloading request.");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::gpu::LocalGPUManager;
    use std::sync::Arc;

    #[test]
    fn test_local_configuration_detection() {
        let local_config = AIServiceLLMConfig {
            local: true,
            port: Some(8000),
            host: "localhost".to_string(),
            model: "test-model".to_string(),
            api_key: "".to_string(),
        };

        let gpu_manager = Arc::new(LocalGPUManager::new());
        let service = LLMService::try_new(local_config, None, gpu_manager).unwrap();

        assert!(!service.is_unified_remote);
        assert!(service.local_vllm_client.is_some());
        assert!(service.unified_remote_client.is_none());
    }

    #[test]
    fn test_remote_configuration_detection() {
        let remote_config = AIServiceLLMConfig {
            local: false,
            port: None,
            host: "https://api.groq.com/openai/v1".to_string(),
            model: "test-model".to_string(),
            api_key: "test-key".to_string(),
        };

        let gpu_manager = Arc::new(LocalGPUManager::new());
        let service = LLMService::try_new(remote_config, None, gpu_manager).unwrap();

        assert!(service.is_unified_remote);
        assert!(service.local_vllm_client.is_none());
        assert!(service.unified_remote_client.is_some());
    }

    #[test]
    fn test_gpu_overload_check_with_remote_config() {
        let remote_config = AIServiceLLMConfig {
            local: false,
            port: None,
            host: "https://api.groq.com/openai/v1".to_string(),
            model: "test-model".to_string(),
            api_key: "test-key".to_string(),
        };

        let gpu_manager = Arc::new(LocalGPUManager::new());
        let service = LLMService::try_new(remote_config, None, gpu_manager).unwrap();

        // GPU overload should always return false for remote configurations
        assert!(!service.is_gpu_overloaded());
    }
}
