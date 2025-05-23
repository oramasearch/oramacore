use anyhow::Result;
use std::sync::Arc;

use super::llms::LLMService;
use crate::types::{CollectionId, InteractionLLMConfig, InteractionMessage};

pub enum AdvancedAutoQuerySteps {
    Init,
    AnalyzeInput,
    GetAllProps,
    SelectProps,
    GetPropValues,
    GenerateQueries,
    RunQueries,
}

pub struct AdvancedAutoQueryStepResult {
    pub step: AdvancedAutoQuerySteps,
    pub result: String,
}

pub struct AdvancedAutoQuery {
    pub llm_service: Arc<LLMService>,
    pub llm_config: Option<InteractionLLMConfig>,
    pub collection_id: CollectionId,
    pub step: AdvancedAutoQuerySteps,
    pub step_results: Vec<AdvancedAutoQueryStepResult>,
}

impl AdvancedAutoQuery {
    pub fn new(
        collection_id: CollectionId,
        llm_service: Arc<LLMService>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Self {
        AdvancedAutoQuery {
            llm_service,
            collection_id,
            step: AdvancedAutoQuerySteps::Init,
            step_results: Vec::new(),
            llm_config,
        }
    }

    pub async fn analyze_input(
        &self,
        conversation: Vec<InteractionMessage>,
    ) -> Result<Vec<String>> {
        let conversation_to_json = serde_json::to_string(&conversation)?;
        let variables = vec![("conversation".to_string(), conversation_to_json)];

        let result = self
            .llm_service
            .run_known_prompt(
                super::llms::KnownPrompts::AdvancedAutoqueryQueryAnalyzer,
                variables,
                self.llm_config.clone(),
            )
            .await?;

        let repaired = self.strip_markdown_wrapper(&result);

        Ok(serde_json::from_str::<Vec<String>>(&repaired)?) // Using "Ok" here to coerce type into Anyhow::Result
    }

    // @todo: replace this with a more robust solution
    fn strip_markdown_wrapper(&self, input: &str) -> String {
        let trimmed = input.trim();
        if trimmed.starts_with("```") {
            let lines: Vec<&str> = trimmed.lines().collect();

            if lines.len() >= 3 {
                let start_idx = if lines[0] == "```" || lines[0].starts_with("```json") {
                    1
                } else {
                    0
                };

                let end_idx = if lines.last() == Some(&"```") {
                    lines.len() - 1
                } else {
                    lines.len()
                };

                if start_idx < end_idx {
                    return lines[start_idx..end_idx].join("\n");
                }
            }
        }

        trimmed.to_string()
    }
}
