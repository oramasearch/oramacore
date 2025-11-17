use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::str::FromStr;

use crate::types::InteractionLLMConfig;
use anyhow::{anyhow, Result};
use strum_macros::Display;

pub mod advanced_autoquery;
pub mod answer;
pub mod automatic_embeddings_selector;
pub mod gpu;
pub mod llms;
pub mod ragat;
mod run_hooks;
pub mod state_machines;
pub mod tools;
pub mod training_sets;
#[derive(Debug, Deserialize, Clone)]
pub struct AIServiceLLMConfig {
    #[serde(default = "default_local")]
    pub local: bool,
    pub port: Option<u16>,
    pub host: String,
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

#[derive(Debug, Serialize, Clone, Hash, PartialEq, Eq, Display, Copy, JsonSchema)]
pub enum RemoteLLMProvider {
    OramaCore,
    OpenAI,
    Fireworks,
    Together,
    GoogleVertex,
    Groq,
    Anthropic,
}

impl FromStr for RemoteLLMProvider {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(RemoteLLMProvider::OpenAI),
            "google" => Ok(RemoteLLMProvider::GoogleVertex),
            "google_vertex" => Ok(RemoteLLMProvider::GoogleVertex),
            "googlevertex" => Ok(RemoteLLMProvider::GoogleVertex),
            "vertex" => Ok(RemoteLLMProvider::GoogleVertex),
            "groq" => Ok(RemoteLLMProvider::Groq),
            "anthropic" => Ok(RemoteLLMProvider::Anthropic),
            _ => Err(anyhow!("Invalid remote LLM provider: {s}")),
        }
    }
}

impl<'de> Deserialize<'de> for RemoteLLMProvider {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        FromStr::from_str(&s).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct RemoteLLMsConfig {
    pub provider: RemoteLLMProvider,
    pub api_key: String,
    pub url: Option<String>,
    pub default_model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AIServiceEmbeddingsConfig {
    pub automatic_embeddings_selector: Option<InteractionLLMConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AIServiceConfig {
    pub llm: AIServiceLLMConfig,
    pub remote_llms: Option<Vec<RemoteLLMsConfig>>,
    pub embeddings: Option<AIServiceEmbeddingsConfig>,
}

#[derive(Debug)]
pub struct AIService {}

impl AIService {
    pub fn new(_config: AIServiceConfig) -> Self {
        Self {}
    }
}

fn default_local() -> bool {
    true
}
