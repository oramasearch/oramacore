use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolArgs,
        ChatCompletionToolType, CreateChatCompletionRequestArgs, FunctionObject,
    },
};
use futures::{Stream, StreamExt};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::types::{InteractionLLMConfig, InteractionMessage, RelatedRequest};
use crate::{
    collection_manager::sides::system_prompts::SystemPrompt,
    types::{RelatedQueriesFormat, Role},
};

use super::{
    party_planner::Step, tools::Tool, AIServiceLLMConfig, RemoteLLMProvider, RemoteLLMsConfig,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnownPrompts {
    Answer,
    Autoquery,
    AutomaticEmbeddingsSelector,
    OptimizeQuery,
    PartyPlanner,
    Segmenter,
    Trigger,
    ValidateSystemPrompt,
    Followup,
    GenerateRelatedQueries,
}

#[derive(Debug, Clone)]
pub struct KnownPrompt {
    pub system: String,
    pub user: String,
}

pub enum PartyPlannerPrompt {
    AskFollowup,
    OptimizeQuery,
    GenerateQueries,
    DescribeInputCode,
    ImproveInput,
    CreateCode,
    GiveReply,
}

impl TryFrom<&str> for PartyPlannerPrompt {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "ASK_FOLLOWUP" => Ok(Self::AskFollowup),
            "OPTIMIZE_QUERY" => Ok(Self::OptimizeQuery),
            "GENERATE_QUERIES" => Ok(Self::GenerateQueries),
            "DESCRIBE_INPUT_CODE" => Ok(Self::DescribeInputCode),
            "IMPROVE_INPUT" => Ok(Self::ImproveInput),
            "CREATE_CODE" => Ok(Self::CreateCode),
            "GIVE_REPLY" => Ok(Self::GiveReply),
            _ => Err(format!("Unknown prompt type: {}", s)),
        }
    }
}

pub struct PartyPlannerPromptHyperParams {
    pub system_prompt: String,
    pub user_prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
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
            KnownPrompts::PartyPlanner => KnownPrompt {
                system: include_str!("../prompts/v1/party_planner/system.md").to_string(),
                user: include_str!("../prompts/v1/party_planner/user.md").to_string(),
            },
            KnownPrompts::Segmenter => KnownPrompt {
                system: include_str!("../prompts/v1/segmenter/system.md").to_string(),
                user: include_str!("../prompts/v1/segmenter/user.md").to_string(),
            },
            KnownPrompts::Trigger => KnownPrompt {
                system: include_str!("../prompts/v1/trigger/system.md").to_string(),
                user: include_str!("../prompts/v1/trigger/user.md").to_string(),
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
        }
    }
}

impl PartyPlannerPrompt {
    pub fn get_hyperparameters(
        &self,
        variables: Vec<(String, String)>,
    ) -> PartyPlannerPromptHyperParams {
        let variables_map: HashMap<String, String> = HashMap::from_iter(variables);

        match self {
            PartyPlannerPrompt::AskFollowup => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/ask_followup_system.md")
                        .to_string();
                let user = include_str!("../prompts/v1/party_planner/actions/ask_followup_user.md")
                    .to_string();

                PartyPlannerPromptHyperParams {
                    system_prompt: system,
                    user_prompt: format_prompt(user, variables_map),
                    max_tokens: 256,
                    temperature: 0.3,
                }
            }
            PartyPlannerPrompt::OptimizeQuery => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/optimize_query_system.md")
                        .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/optimize_query_user.md")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    system_prompt: system,
                    user_prompt: format_prompt(user, variables_map),
                    max_tokens: 256,
                    temperature: 0.3,
                }
            }
            PartyPlannerPrompt::GenerateQueries => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/generate_queries_system.md")
                        .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/generate_queries_user.md")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    system_prompt: system,
                    user_prompt: format_prompt(user, variables_map),
                    max_tokens: 512,
                    temperature: 0.3,
                }
            }
            PartyPlannerPrompt::DescribeInputCode => {
                let system = include_str!(
                    "../prompts/v1/party_planner/actions/describe_input_code_system.md"
                )
                .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/describe_input_code_user.md")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    system_prompt: system,
                    user_prompt: format_prompt(user, variables_map),
                    max_tokens: 512,
                    temperature: 0.1,
                }
            }
            PartyPlannerPrompt::ImproveInput => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/improve_input_system.md")
                        .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/improve_input_user.md")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    system_prompt: system,
                    user_prompt: format_prompt(user, variables_map),
                    max_tokens: 1024,
                    temperature: 0.2,
                }
            }
            PartyPlannerPrompt::CreateCode => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/create_code_system.md")
                        .to_string();
                let user = include_str!("../prompts/v1/party_planner/actions/create_code_user.md")
                    .to_string();

                PartyPlannerPromptHyperParams {
                    system_prompt: system,
                    user_prompt: format_prompt(user, variables_map),
                    max_tokens: 2048,
                    temperature: 0.1,
                }
            }
            PartyPlannerPrompt::GiveReply => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/give_reply_system.md")
                        .to_string();
                let user = include_str!("../prompts/v1/party_planner/actions/give_reply_user.md")
                    .to_string();

                PartyPlannerPromptHyperParams {
                    system_prompt: system,
                    user_prompt: format_prompt(user, variables_map),
                    max_tokens: 4096,
                    temperature: 0.1,
                }
            }
        }
    }
}

pub fn format_prompt(prompt: String, variables: HashMap<String, String>) -> String {
    let mut result = prompt.to_string();
    for (key, value) in variables {
        result = result.replace(&format!("{{{}}}", key), &value);
    }
    result
}

#[derive(Debug)]
pub struct LLMService {
    pub local_vllm_client: async_openai::Client<OpenAIConfig>,
    pub remote_clients: Option<HashMap<RemoteLLMProvider, async_openai::Client<OpenAIConfig>>>,
    pub model: String,
    pub default_remote_models: Option<HashMap<RemoteLLMProvider, String>>,
}

impl LLMService {
    pub fn try_new(
        local_vllm_config: AIServiceLLMConfig,
        remote_llm_config: Option<Vec<RemoteLLMsConfig>>,
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
        })
    }

    pub async fn execute_tools(
        &self,
        message: String,
        tools: Vec<FunctionObject>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<Vec<ChatCompletionMessageToolCall>>> {
        let chosen_model = self.get_chosen_model(llm_config.clone());
        let llm_client = self.get_chosen_llm_client(llm_config);

        // @todo: accept a list of chat messages
        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(message)
            .build()
            .unwrap()
            .into();

        let all_tools: Vec<ChatCompletionTool> = tools
            .iter()
            .map(|tool| ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: tool.clone(),
            })
            .collect();

        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(1024u32)
            .temperature(0.0)
            .model(chosen_model)
            .messages([user_message])
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

        Ok(response_message.tool_calls)
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
            .await;

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
    ) -> impl Stream<Item = Result<String>> {
        let mut prompts = prompt.get_prompts();
        let variables_map: HashMap<String, String> = HashMap::from_iter(variables);

        if let Some(system_propmpt) = custom_system_prompt {
            prompts.system += format!(
                "\n\n# Additional Information - FOLLOW STRICTLY\n\n{}",
                system_propmpt.prompt
            )
            .as_str();
        }

        // Only for Answer prompts, add the trigger to the user prompt
        if prompt == KnownPrompts::Answer {
            if variables_map.contains_key("segment") {
                prompts.user += "### Persona\n\n";
                prompts.user += variables_map
                    .get("segment")
                    .context("Unable to retrieve segment from variable map")
                    .unwrap();
                prompts.user += "\n\n";
            }

            if variables_map.contains_key("trigger") {
                prompts.user += "### Instructions\n\n";
                prompts.user += variables_map
                    .get("trigger")
                    .context("Unable to retrieve trigger from variable map")
                    .unwrap();
                prompts.user += "\n\n";
            }
        }

        let chosen_model = self.get_chosen_model(llm_config.clone());
        let llm_client = self.get_chosen_llm_client(llm_config);

        let request = CreateChatCompletionRequestArgs::default()
            .model(chosen_model)
            .max_tokens(512u32)
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
            .context("Unable to build KnownPrompt LLM request body")
            .unwrap();

        let mut response_stream = llm_client
            .chat()
            .create_stream(request)
            .await
            .context("An error occurred while initializing the stream from remote LLM instance")
            .unwrap();

        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(result) = response_stream.next().await {
                match result {
                    Ok(response) => {
                        let chunk = response
                            .choices
                            .first()
                            .unwrap()
                            .delta
                            .content
                            .as_ref()
                            .unwrap();

                        tx.send(Ok(chunk.to_string())).await.unwrap();
                    }
                    Err(e) => {
                        let error_message = format!("An error occurred while processing the response from the remote LLM instance: {:?}", e);
                        tx.send(Err(anyhow::Error::msg(error_message)))
                            .await
                            .unwrap();
                    }
                }
            }
        });

        ReceiverStream::new(rx)
    }

    pub async fn run_party_planner_prompt_stream(
        &self,
        step: Step,
        input: &str,
        history: &Vec<InteractionMessage>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<impl Stream<Item = Result<String>>> {
        let step_name = step.name.as_str();

        let variables = match step_name {
            "GIVE_REPLY" => vec![
                ("question".to_string(), input.to_owned()),
                ("context".to_string(), step.description),
            ],
            _ => vec![
                ("input".to_string(), input.to_owned()),
                ("description".to_string(), step.description),
            ],
        };

        let party_planner_prompt = PartyPlannerPrompt::try_from(step_name).unwrap();
        let hyperparameters = party_planner_prompt.get_hyperparameters(variables);

        let mut full_history: Vec<ChatCompletionRequestMessage> =
            vec![ChatCompletionRequestSystemMessageArgs::default()
                .content(hyperparameters.system_prompt)
                .build()
                .unwrap()
                .into()];

        for message in history {
            match message.role {
                Role::System => {
                    // @todo: make sure there are no multiple system messages in the history
                    // return Err(anyhow::Error::msg(
                    //     "Found multiple system messages in Party Planner chat history",
                    // ));
                }
                Role::User => {
                    full_history.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(message.content.clone())
                            .build()
                            .unwrap()
                            .into(),
                    );
                }
                Role::Assistant => {
                    full_history.push(
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(message.content.clone())
                            .build()
                            .unwrap()
                            .into(),
                    );
                }
            }
        }

        full_history.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(hyperparameters.user_prompt)
                .build()
                .unwrap()
                .into(),
        );

        let chosen_model = self.get_chosen_model(llm_config.clone());
        let llm_client = self.get_chosen_llm_client(llm_config);

        let request = CreateChatCompletionRequestArgs::default()
            .model(chosen_model)
            .max_tokens(hyperparameters.max_tokens)
            .stream(true)
            .messages(full_history)
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
                        let error_message = format!("An error occurred while processing the response from the remote Party Planner LLM instance: {:?}", e);
                        tx.send(Err(anyhow::Error::msg(error_message)))
                            .await
                            .unwrap();
                    }
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }

    pub async fn run_party_planner_prompt(
        &self,
        step: Step,
        input: &str,
        history: &Vec<InteractionMessage>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<String> {
        let mut acc = String::new();
        let mut stream = self
            .run_party_planner_prompt_stream(step, input, history, llm_config)
            .await?;

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
}
