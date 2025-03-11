use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
        CreateChatCompletionStreamResponse,
    },
};
use futures::{Stream, StreamExt, TryStreamExt};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use openai_api_rust::{
    chat::{ChatApi, ChatBody},
    Auth, Message, OpenAI, Role,
};

pub enum KnownPrompts {
    Answer,
    Autoquery,
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
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<i32>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub n: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub top_p: Option<f32>,
    pub user: Option<String>,
}

impl KnownPrompts {
    pub fn get_prompts(&self) -> KnownPrompt {
        match self {
            KnownPrompts::Answer => KnownPrompt {
                system: include_str!("../prompts/v1/answer/system.txt").to_string(),
                user: include_str!("../prompts/v1/answer/user.txt").to_string(),
            },
            KnownPrompts::Autoquery => KnownPrompt {
                system: include_str!("../prompts/v1/autoquery/system.txt").to_string(),
                user: include_str!("../prompts/v1/autoquery/user.txt").to_string(),
            },
            KnownPrompts::PartyPlanner => KnownPrompt {
                system: include_str!("../prompts/v1/party_planner/system.txt").to_string(),
                user: include_str!("../prompts/v1/party_planner/user.txt").to_string(),
            },
            KnownPrompts::Segmenter => KnownPrompt {
                system: include_str!("../prompts/v1/segmenter/system.txt").to_string(),
                user: include_str!("../prompts/v1/segmenter/user.txt").to_string(),
            },
            KnownPrompts::Trigger => KnownPrompt {
                system: include_str!("../prompts/v1/trigger/system.txt").to_string(),
                user: include_str!("../prompts/v1/trigger/user.txt").to_string(),
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
                    include_str!("../prompts/v1/party_planner/actions/ask_followup_system.txt")
                        .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/ask_followup_user.txt")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: system,
                        },
                        Message {
                            role: Role::User,
                            content: format_prompt(user, variables_map),
                        },
                    ],
                    max_tokens: Some(256),
                    stream: Some(true),
                    temperature: Some(0.3),
                    frequency_penalty: None,
                    logit_bias: None,
                    n: None,
                    presence_penalty: None,
                    stop: None,
                    top_p: None,
                    user: None,
                }
            }
            PartyPlannerPrompt::OptimizeQuery => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/optimize_query_system.txt")
                        .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/optimize_query_user.txt")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: system,
                        },
                        Message {
                            role: Role::User,
                            content: format_prompt(user, variables_map),
                        },
                    ],
                    max_tokens: Some(256),
                    stream: Some(false),
                    temperature: Some(0.3),
                    frequency_penalty: None,
                    logit_bias: None,
                    n: None,
                    presence_penalty: None,
                    stop: None,
                    top_p: None,
                    user: None,
                }
            }
            PartyPlannerPrompt::GenerateQueries => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/generate_queries_system.txt")
                        .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/generate_queries_user.txt")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: system,
                        },
                        Message {
                            role: Role::User,
                            content: format_prompt(user, variables_map),
                        },
                    ],
                    max_tokens: Some(512),
                    stream: Some(false),
                    temperature: Some(0.3),
                    frequency_penalty: None,
                    logit_bias: None,
                    n: None,
                    presence_penalty: None,
                    stop: None,
                    top_p: None,
                    user: None,
                }
            }
            PartyPlannerPrompt::DescribeInputCode => {
                let system = include_str!(
                    "../prompts/v1/party_planner/actions/describe_input_code_system.txt"
                )
                .to_string();
                let user = include_str!(
                    "../prompts/v1/party_planner/actions/describe_input_code_user.txt"
                )
                .to_string();

                PartyPlannerPromptHyperParams {
                    model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: system,
                        },
                        Message {
                            role: Role::User,
                            content: format_prompt(user, variables_map),
                        },
                    ],
                    max_tokens: Some(512),
                    stream: Some(true),
                    temperature: Some(0.1),
                    frequency_penalty: None,
                    logit_bias: None,
                    n: None,
                    presence_penalty: None,
                    stop: None,
                    top_p: None,
                    user: None,
                }
            }
            PartyPlannerPrompt::ImproveInput => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/improve_input_system.txt")
                        .to_string();
                let user =
                    include_str!("../prompts/v1/party_planner/actions/improve_input_user.txt")
                        .to_string();

                PartyPlannerPromptHyperParams {
                    model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: system,
                        },
                        Message {
                            role: Role::User,
                            content: format_prompt(user, variables_map),
                        },
                    ],
                    max_tokens: Some(1024),
                    stream: Some(true),
                    temperature: Some(0.2),
                    frequency_penalty: None,
                    logit_bias: None,
                    n: None,
                    presence_penalty: None,
                    stop: None,
                    top_p: None,
                    user: None,
                }
            }
            PartyPlannerPrompt::CreateCode => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/create_code_system.txt")
                        .to_string();
                let user = include_str!("../prompts/v1/party_planner/actions/create_code_user.txt")
                    .to_string();

                PartyPlannerPromptHyperParams {
                    model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: system,
                        },
                        Message {
                            role: Role::User,
                            content: format_prompt(user, variables_map),
                        },
                    ],
                    max_tokens: Some(2048),
                    stream: Some(true),
                    temperature: Some(0.1),
                    frequency_penalty: None,
                    logit_bias: None,
                    n: None,
                    presence_penalty: None,
                    stop: None,
                    top_p: None,
                    user: None,
                }
            }
            PartyPlannerPrompt::GiveReply => {
                let system =
                    include_str!("../prompts/v1/party_planner/actions/give_reply_system.txt")
                        .to_string();
                let user = include_str!("../prompts/v1/party_planner/actions/give_reply_user.txt")
                    .to_string();

                PartyPlannerPromptHyperParams {
                    model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: system,
                        },
                        Message {
                            role: Role::User,
                            content: format_prompt(user, variables_map),
                        },
                    ],
                    max_tokens: Some(4096),
                    stream: Some(true),
                    temperature: Some(0.1),
                    frequency_penalty: None,
                    logit_bias: None,
                    n: None,
                    presence_penalty: None,
                    stop: None,
                    top_p: None,
                    user: None,
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
    async_openai::Client::with_config(
        OpenAIConfig::new().with_api_base("http://localhost:8000/v1/"),
    )
}

pub async fn run_known_prompt(prompt: KnownPrompts, variables: Vec<(String, String)>) -> String {
    let mut acc = String::new();
    let mut stream = run_known_prompt_stream(prompt, variables).await;

    while let Some(msg) = stream.next().await {
        acc += &msg;
    }

    acc
}

pub async fn run_known_prompt_stream(
    prompt: KnownPrompts,
    variables: Vec<(String, String)>,
) -> impl Stream<Item = String> {
    let client = get_openai_client();

    let prompts = prompt.get_prompts();
    let variables_map: HashMap<String, String> = HashMap::from_iter(variables);

    let request = CreateChatCompletionRequestArgs::default()
        .model("Qwen/Qwen2.5-3B-Instruct")
        .max_tokens(512 as u32)
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

                    tx.send(chunk.to_string()).await.unwrap();
                }
                Err(e) => {
                    // @todo: implement better error handling
                    todo!();
                }
            }
        }
    });

    ReceiverStream::new(rx)
}

pub fn run_party_planner_prompt(
    step_name: String,
    variables: Vec<(String, String)>,
) -> impl Stream<Item = Result<String>> {
    let client = OpenAI::new(Auth::new(""), "http://localhost:8000/v1/");
    let party_planner_prompt = PartyPlannerPrompt::try_from(step_name.as_str()).unwrap();
    let hyperparameters = party_planner_prompt.get_hyperparameters(variables);

    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        let response = client
            .chat_completion_create(&ChatBody {
                model: hyperparameters.model,
                messages: hyperparameters.messages,
                max_tokens: hyperparameters.max_tokens,
                stream: hyperparameters.stream,
                temperature: hyperparameters.temperature,
                frequency_penalty: hyperparameters.frequency_penalty,
                logit_bias: None,
                n: hyperparameters.n,
                presence_penalty: hyperparameters.presence_penalty,
                stop: hyperparameters.stop,
                top_p: hyperparameters.top_p,
                user: hyperparameters.user,
            })
            .expect("Failed to get response");

        match response.choices.first() {
            Some(choice) => match &choice.message {
                Some(message) => {
                    tx.send(Ok(message.content.clone())).await.unwrap();
                }
                None => {
                    tx.send(Ok("{}".to_string())).await.unwrap();
                }
            },
            None => {
                tx.send(Ok("{}".to_string())).await.unwrap();
            }
        }
    });

    ReceiverStream::new(rx)
}
