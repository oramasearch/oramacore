use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use futures::{Stream, StreamExt};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::collection_manager::dto::InteractionMessage;

use super::party_planner::Step;

pub enum KnownPrompts {
    Answer,
    Autoquery,
    OptimizeQuery,
    PartyPlanner,
    Segmenter,
    Trigger,
}

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

pub fn get_openai_client() -> async_openai::Client<OpenAIConfig> {
    async_openai::Client::with_config(OpenAIConfig::new().with_api_base("http://localhost:8000/v1"))
}

pub async fn run_known_prompt(
    prompt: KnownPrompts,
    variables: Vec<(String, String)>,
) -> Result<String> {
    let mut acc = String::new();
    let mut stream = run_known_prompt_stream(prompt, variables).await;

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
    prompt: KnownPrompts,
    variables: Vec<(String, String)>,
) -> impl Stream<Item = Result<String>> {
    let client = get_openai_client();

    let prompts = prompt.get_prompts();
    let variables_map: HashMap<String, String> = HashMap::from_iter(variables);

    let request = CreateChatCompletionRequestArgs::default()
        .model("Qwen/Qwen2.5-3B-Instruct")
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

    let mut response_stream = client
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
    step: Step,
    input: &String,
    history: &Vec<InteractionMessage>,
) -> Result<impl Stream<Item = Result<String>>> {
    let client = get_openai_client();
    let step_name = step.name.as_str();

    let variables = match step_name {
        "GIVE_REPLY" => vec![
            ("question".to_string(), input.clone()),
            ("context".to_string(), step.description),
        ],
        _ => vec![
            ("input".to_string(), input.clone()),
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
            crate::collection_manager::dto::Role::System => {
                // @todo: make sure there are no multiple system messages in the history
                // return Err(anyhow::Error::msg(
                //     "Found multiple system messages in Party Planner chat history",
                // ));
            }
            crate::collection_manager::dto::Role::User => {
                full_history.push(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(message.content.clone())
                        .build()
                        .unwrap()
                        .into(),
                );
            }
            crate::collection_manager::dto::Role::Assistant => {
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

    let request = CreateChatCompletionRequestArgs::default()
        .model("Qwen/Qwen2.5-3B-Instruct")
        .max_tokens(hyperparameters.max_tokens)
        .stream(true)
        .messages(full_history)
        .build()
        .context("Unable to build KnownPrompt LLM request body")?;

    let mut response_stream = client
        .chat()
        .create_stream(request)
        .await
        .context("An error occurred while initializing the stream from remote LLM instance")?;

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
    step: Step,
    input: &String,
    history: &Vec<InteractionMessage>,
) -> Result<String> {
    let mut acc = String::new();
    let mut stream = run_party_planner_prompt_stream(step, &input, &history).await?;

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
