use crate::types::InteractionLLMConfig;
use anyhow::{Context, Result};
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

use super::llms::LLMService;

pub type JSONDocument = Map<String, Value>;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, ToSchema)]
pub struct ChosenProperties {
    pub properties: Vec<String>,
    #[serde(rename = "includeKeys")]
    pub include_keys: Vec<String>,
    pub rename: HashMap<String, String>,
}

impl ChosenProperties {
    pub fn format(&self, document: &JSONDocument) -> String {
        let mut formatted_parts = Vec::new();

        // Process each property in order specified by properties array
        for property in &self.properties {
            if let Some(value) = document.get(property) {
                // Get the display name (either from rename map or original key)
                let raw_display_name = self.rename.get(property).unwrap_or(property);

                // Convert the key name to human-readable format
                let display_name = Self::humanize_key(raw_display_name);

                let formatted_value = match value {
                    Value::String(s) => {
                        if s.is_empty() {
                            continue; // Skip empty strings
                        }

                        if self.include_keys.contains(property) {
                            format!("{} {}", display_name, s)
                        } else {
                            s.clone()
                        }
                    }
                    Value::Number(n) => {
                        if self.include_keys.contains(property) {
                            format!("{} {}", display_name, n)
                        } else {
                            n.to_string()
                        }
                    }
                    Value::Array(arr) => {
                        if arr.is_empty() {
                            continue; // Skip empty arrays
                        }

                        // Format array elements as strings
                        let values: Vec<String> = arr
                            .iter()
                            .filter_map(|v| match v {
                                Value::String(s) => Some(s.clone()),
                                Value::Number(n) => Some(n.to_string()),
                                _ => None, // Skip other types in arrays
                            })
                            .collect();

                        if values.is_empty() {
                            continue;
                        }

                        if self.include_keys.contains(property) {
                            format!("{} {}", display_name, values.join(", "))
                        } else {
                            values.join(", ")
                        }
                    }
                    // Skip other types
                    _ => continue,
                };

                formatted_parts.push(formatted_value);
            }
        }

        println!("\n\n\n");
        dbg!(&formatted_parts.join(". "));
        println!("\n\n\n");

        // Join all parts with periods and spaces
        formatted_parts.join(". ")
    }

    fn humanize_key(key: &str) -> String {
        if key.is_empty() {
            return String::new();
        }

        // First, handle snake_case and kebab-case by replacing separators with spaces
        let with_spaces = key.replace('_', " ").replace('-', " ");

        // Buffer to build our humanized string
        let mut result = String::with_capacity(with_spaces.len());

        // Track if we need to insert a space
        let mut prev_was_lower = false;
        let mut prev_was_upper = false;

        for (i, c) in with_spaces.chars().enumerate() {
            if c == ' ' {
                // Keep existing spaces
                result.push(' ');
                prev_was_lower = false;
                prev_was_upper = false;
                continue;
            }

            let is_upper = c.is_uppercase();

            if i > 0 {
                // Add space when transitioning from lowercase to uppercase (camelCase)
                // Or when going from uppercase to uppercase followed by lowercase (ABCdef)
                if (prev_was_lower && is_upper)
                    || (prev_was_upper
                        && is_upper
                        && with_spaces
                            .chars()
                            .nth(i + 1)
                            .map_or(false, |next| next.is_lowercase()))
                {
                    result.push(' ');
                }
            }

            // For the first character or after a space, keep the case as is
            // For others, preserve the original case
            if i == 0 || with_spaces.chars().nth(i - 1) == Some(' ') {
                result.extend(c.to_uppercase());
            } else {
                result.push(c);
            }

            prev_was_lower = c.is_lowercase();
            prev_was_upper = c.is_uppercase();
        }

        result
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChosenPropertiesError {
    pub error: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChosenPropertiesResult {
    Properties(ChosenProperties),
    Error(ChosenPropertiesError),
}

pub struct AutomaticEmbeddingsSelector {
    pub llm_service: Arc<LLMService>,
    pub llm_config: Option<InteractionLLMConfig>,
}

impl AutomaticEmbeddingsSelector {
    pub fn new(llm_service: Arc<LLMService>, llm_config: Option<InteractionLLMConfig>) -> Self {
        Self {
            llm_service,
            llm_config,
        }
    }

    pub async fn choose_properties(&self, document: &JSONDocument) -> Result<ChosenProperties> {
        let documents_as_json = serde_json::to_string(document)?;
        let variables = vec![("document".to_string(), documents_as_json)];

        let result = self
            .llm_service
            .run_known_prompt(
                super::llms::KnownPrompts::AutomaticEmbeddingsSelector,
                variables,
                self.llm_config.clone(), // @todo: avoid this cloning when possible
            )
            .await
            .with_context(|| "Unable to determine which properties to use for embeddings")?;

        println!("\n\n\n");
        dbg!(&result);
        println!("\n\n\n");

        match serde_json::from_str(&result)? {
            ChosenPropertiesResult::Properties(properties) => Ok(properties),
            ChosenPropertiesResult::Error(error) => Err(anyhow::anyhow!(error.error)),
        }
    }
}
