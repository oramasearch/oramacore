use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::{fmt::Display, str::FromStr};

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
    #[serde(default = "default_default_model_group")]
    pub default_model_group: String,
    #[serde(default = "dynamically_load_models")]
    pub dynamically_load_models: bool,

    #[serde(default = "default_execution_providers")]
    pub execution_providers: Vec<String>,

    #[serde(default = "default_embedding_total_threads")]
    pub total_threads: u8,

    pub automatic_embeddings_selector: Option<InteractionLLMConfig>,

    #[serde(default = "default_logging_level")]
    pub level: PythonLoggingLevel,
}

#[derive(Debug, Deserialize, Clone)]
pub enum PythonLoggingLevel {
    #[serde(rename_all = "UPPERCASE")]
    Debug,
    #[serde(rename_all = "UPPERCASE")]
    Info,
    #[serde(rename_all = "UPPERCASE")]
    Warning,
    #[serde(rename_all = "UPPERCASE")]
    Error,
    #[serde(rename_all = "UPPERCASE")]
    Critical,
}

impl Display for PythonLoggingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level_str = match self {
            PythonLoggingLevel::Debug => "DEBUG",
            PythonLoggingLevel::Info => "INFO",
            PythonLoggingLevel::Warning => "WARNING",
            PythonLoggingLevel::Error => "ERROR",
            PythonLoggingLevel::Critical => "CRITICAL",
        };
        write!(f, "{level_str}")
    }
}

fn default_logging_level() -> PythonLoggingLevel {
    PythonLoggingLevel::Info
}

fn dynamically_load_models() -> bool {
    true
}

fn default_embedding_total_threads() -> u8 {
    8
}

fn default_execution_providers() -> Vec<String> {
    vec!["CUDAExecutionProvider".to_string()]
}

fn default_default_model_group() -> String {
    "all".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct AIServiceConfig {
    pub llm: AIServiceLLMConfig,
    pub remote_llms: Option<Vec<RemoteLLMsConfig>>,
    pub embeddings: Option<AIServiceEmbeddingsConfig>,
    #[serde(default = "default_models_cache_dir")]
    pub models_cache_dir: String,
    #[serde(default = "default_total_threads")]
    pub total_threads: u8,
}

fn default_total_threads() -> u8 {
    12
}

fn default_models_cache_dir() -> String {
    "/tmp/fastembed_cache".to_string()
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
