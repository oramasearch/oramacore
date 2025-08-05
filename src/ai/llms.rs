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
    TrainingSetsQueriesGenerator,
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
            KnownPrompts::TrainingSetsQueriesGenerator => KnownPrompt {
                system: include_str!(
                    "../prompts/v1/training_sets/query_optimizer/queries_generator/system.md"
                )
                .to_string(),
                user: include_str!(
                    "../prompts/v1/training_sets/query_optimizer/queries_generator/user.md"
                )
                .to_string(),
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
    pub local_vllm_client: async_openai::Client<OpenAIConfig>,
    pub remote_clients: Option<HashMap<RemoteLLMProvider, async_openai::Client<OpenAIConfig>>>,
    pub model: String,
    pub default_remote_models: Option<HashMap<RemoteLLMProvider, String>>,
    pub local_gpu_manager: Arc<LocalGPUManager>,
}

impl LLMService {
    pub fn try_new(
        local_vllm_config: AIServiceLLMConfig,
        remote_llm_config: Option<Vec<RemoteLLMsConfig>>,
        local_gpu_manager: Arc<LocalGPUManager>,
    ) -> Result<Self> {
        let local_vllm_provider_url = format!(
            "http://{}:{}/v1",
            local_vllm_config.host, local_vllm_config.port
        );

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
            local_vllm_client: async_openai::Client::with_config(
                OpenAIConfig::new().with_api_base(&local_vllm_provider_url),
            ),
            default_remote_models,
            remote_clients,
            model: local_vllm_config.model,
            local_gpu_manager,
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
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<String> {
        let mut acc = String::new();
        let mut stream = self
            .run_known_prompt_stream(prompt, variables, None, llm_config)
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
        if let Some(config) = config {
            if let Some(remote_clients) = &self.remote_clients {
                if let Some(client) = remote_clients.get(&config.provider) {
                    return client.clone();
                }
            }
        }

        self.local_vllm_client.clone()
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
        return vec![
            ("conversation".to_string(), "".to_string()),
            ("context".to_string(), "".to_string()),
            ("query".to_string(), suggestion_request.query),
            (
                "maxSuggestions".to_string(),
                suggestion_request.max_suggestions.unwrap_or(3).to_string(),
            ),
        ];
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
