use anyhow::Result;
use std::collections::HashMap;

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

pub fn format_prompt(prompt: String, variables: HashMap<String, String>) -> String {
    let mut result = prompt.to_string();
    for (key, value) in variables {
        result = result.replace(&format!("{{{}}}", key), &value);
    }
    result
}

pub fn run_known_prompt(prompt: KnownPrompts, variables: Vec<(String, String)>) -> Result<String> {
    let client = OpenAI::new(Auth::new(""), "http://localhost:8000/v1/");

    let prompts = prompt.get_prompts();
    let variables_map: HashMap<String, String> = HashMap::from_iter(variables);

    let messages = vec![
        Message {
            role: Role::Assistant,
            content: prompts.system,
        },
        Message {
            role: Role::User,
            content: format_prompt(prompts.user, variables_map),
        },
    ];

    let response = client
        .chat_completion_create(&ChatBody {
            model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
            messages,
            max_tokens: Some(512),
            stream: Some(false),
            temperature: Some(0.1),
            frequency_penalty: None,
            logit_bias: None,
            n: None,
            presence_penalty: None,
            stop: None,
            top_p: None,
            user: None,
        })
        .expect("Failed to get response");

    match response.choices.first() {
        Some(choice) => match &choice.message {
            Some(message) => Ok(message.content.clone()),
            None => Ok("{}".to_string()),
        },
        None => Ok("{}".to_string()),
    }
}
