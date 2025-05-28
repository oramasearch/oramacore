use anyhow::Result;
use axum::extract::Query;
use regex::Regex;
use serde::Serialize;
use serde::{self, Deserialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::llms::LLMService;
use crate::types::{SearchOffset, SearchParams};
use crate::{
    collection_manager::sides::CollectionStats,
    types::{CollectionId, InteractionLLMConfig, InteractionMessage},
};

#[derive(Serialize, Debug)]
struct FieldStats {
    field_path: String,
    field_type: String,
}

#[derive(Serialize)]
struct IndexesStats {
    pub id: String,
    fields_stats: Vec<FieldStats>,
}

#[derive(Serialize)]
struct PropertiesSelectorInput {
    pub indexes_stats: Vec<IndexesStats>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct SelectedProperty {
    pub property: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub get: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelectedProperties {
    pub query: String,
    pub properties: Vec<SelectedProperty>,
}

#[derive(Deserialize, Debug)]
struct LLMPropertiesResponse {
    #[serde(flatten)]
    collections: HashMap<String, CollectionProperties>,
}

#[derive(Deserialize, Debug)]
struct CollectionProperties {
    selected_properties: Vec<SelectedProperty>,
}

#[derive(Serialize, Debug)]
pub struct CollectionSelectedProperties {
    pub selected_properties: Vec<SelectedProperty>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum DeserializedStat {
    #[serde(rename = "uncommitted_bool")]
    UncommittedBoolean {
        false_count: usize,
        true_count: usize,
    },

    #[serde(rename = "committed_bool")]
    CommittedBoolean {
        false_count: usize,
        true_count: usize,
    },

    #[serde(rename = "uncommitted_number")]
    UncommittedNumber { min: f64, max: f64, count: usize },

    #[serde(rename = "committed_number")]
    CommittedNumber { min: f64, max: f64 },

    #[serde(rename = "uncommitted_string_filter")]
    UncommittedStringFilter {
        key_count: usize,
        document_count: usize,
    },

    #[serde(rename = "committed_string_filter")]
    CommittedStringFilter {
        key_count: usize,
        document_count: usize,
    },

    #[serde(rename = "uncommitted_string")]
    UncommittedString {
        key_count: usize,
        global_info: GlobalInfo,
    },

    #[serde(rename = "committed_string")]
    CommittedString {
        key_count: usize,
        global_info: GlobalInfo,
    },

    #[serde(rename = "uncommitted_vector")]
    UncommittedVector {
        document_count: usize,
        vector_count: usize,
    },

    #[serde(rename = "committed_vector")]
    CommittedVector {
        dimensions: usize,
        vector_count: usize,
    },
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}

impl DeserializedStat {
    // Helper method to extract document count from any variant
    fn get_document_count(&self) -> usize {
        match self {
            DeserializedStat::UncommittedBoolean {
                false_count,
                true_count,
            } => false_count + true_count,
            DeserializedStat::CommittedBoolean {
                false_count,
                true_count,
            } => false_count + true_count,
            DeserializedStat::UncommittedNumber { count, .. } => *count,
            DeserializedStat::CommittedNumber { .. } => 0, // Committed numbers don't have count
            DeserializedStat::UncommittedStringFilter { document_count, .. } => *document_count,
            DeserializedStat::CommittedStringFilter { document_count, .. } => *document_count,
            DeserializedStat::UncommittedString { global_info, .. } => global_info.total_documents,
            DeserializedStat::CommittedString { global_info, .. } => global_info.total_documents,
            DeserializedStat::UncommittedVector { document_count, .. } => *document_count,
            DeserializedStat::CommittedVector { .. } => 0, // Vector count isn't document count
        }
    }
}

#[derive(Serialize, Debug)]
pub struct QueryAndProperties {
    pub query: String,
    pub properties: HashMap<String, CollectionSelectedProperties>,
}

pub enum AdvancedAutoQuerySteps {
    Init,
    AnalyzeInput,
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
    pub collection_stats: CollectionStats,
    pub step: AdvancedAutoQuerySteps,
    pub step_results: Vec<AdvancedAutoQueryStepResult>,
}

impl AdvancedAutoQuery {
    pub fn new(
        collection_id: CollectionId,
        collection_stats: CollectionStats,
        llm_service: Arc<LLMService>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Self {
        AdvancedAutoQuery {
            llm_service,
            collection_stats,
            collection_id,
            step: AdvancedAutoQuerySteps::Init,
            step_results: Vec::new(),
            llm_config,
        }
    }

    pub async fn run(
        &mut self,
        conversation: Vec<InteractionMessage>,
    ) -> Result<Vec<QueryAndProperties>> {
        let conversation_to_json = serde_json::to_string(&conversation)?;
        let optimized_queries = self.analyze_input(conversation_to_json).await?;
        dbg!(&optimized_queries);
        let selected_properties = self.select_properties(optimized_queries.clone()).await?;
        dbg!(&selected_properties);

        let queries_and_properties: Vec<QueryAndProperties> = optimized_queries
            .into_iter()
            .zip(selected_properties)
            .map(|(query, properties)| QueryAndProperties { query, properties })
            .collect();

        Ok(queries_and_properties)
    }

    pub async fn analyze_input(&self, conversation_as_json: String) -> Result<Vec<String>> {
        let current_datetime = chrono::Utc::now().timestamp();
        let formatted_datetime =
            chrono::DateTime::<chrono::Utc>::from_timestamp(current_datetime, 0)
                .unwrap()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();

        let variables = vec![
            ("conversation".to_string(), conversation_as_json),
            ("timestamp".to_string(), formatted_datetime),
            ("timestamp_unix".to_string(), current_datetime.to_string()),
        ];

        let result = self
            .llm_service
            .run_known_prompt(
                super::llms::KnownPrompts::AdvancedAutoqueryQueryAnalyzer,
                variables,
                self.llm_config.clone(),
            )
            .await?;

        let repaired = self.strip_markdown_wrapper(&result);

        Ok(serde_json::from_str::<Vec<String>>(&repaired)?)
    }

    pub async fn select_properties(
        &mut self,
        queries: Vec<String>,
    ) -> Result<Vec<HashMap<String, CollectionSelectedProperties>>> {
        let selectable_props = serde_json::to_string(&self.format_collection_stats())?;

        let futures = queries
            .iter()
            .map(|query| {
                let variables = vec![
                    ("query".to_string(), query.clone()),
                    ("properties_list".to_string(), selectable_props.clone()),
                ];
                self.llm_service.run_known_prompt(
                    super::llms::KnownPrompts::AdvancedAutoQueryPropertiesSelector,
                    variables,
                    self.llm_config.clone(),
                )
            })
            .collect::<Vec<_>>();

        let results = futures::future::join_all(futures).await;
        dbg!(&results);

        let mut properties_list = Vec::new();

        for (index, result) in results.into_iter().enumerate() {
            let repaired = self.strip_markdown_wrapper(&result?);

            match serde_json::from_str::<LLMPropertiesResponse>(&repaired) {
                Ok(llm_response) => {
                    // Convert the LLM response to the expected output format
                    let mut collection_properties = HashMap::new();

                    for (collection_name, props) in llm_response.collections {
                        collection_properties.insert(
                            collection_name,
                            CollectionSelectedProperties {
                                selected_properties: props.selected_properties,
                            },
                        );
                    }

                    properties_list.push(collection_properties);
                }
                Err(e) => {
                    eprintln!("Failed to parse LLM response at index {}: {}", index, e);
                    eprintln!("Raw response: {}", repaired);

                    // Create empty result as fallback
                    properties_list.push(HashMap::new());
                }
            }
        }

        Ok(properties_list)
    }

    // async fn get_search_query(
    //     &mut self,
    //     query_plan: Vec<QueryAndProperties>,
    // ) -> Result<SearchParams> {
    //     let mut results = Vec::new();

    //     for SelectedProperties { query, properties } in query_plan.iter() {
    //         let properties_values_to_retrieve: Vec<String> = properties
    //             .iter()
    //             .filter(|p| p.get)
    //             .map(|p| p.property.clone())
    //             .collect();
    //     }

    //     unimplemented!()
    // }

    fn format_collection_stats(&mut self) -> PropertiesSelectorInput {
        let prefix_re = Regex::new(r"^(committed|uncommitted)_").unwrap();

        let mut indexes_stats: Vec<IndexesStats> = Vec::new();
        let mut seen: HashMap<String, bool> = HashMap::new();

        for index in self.collection_stats.indexes_stats.iter() {
            let mut fields_stats: Vec<FieldStats> = Vec::new();
            if !index.is_temp {
                let index_id = index.id;

                for stat in index.fields_stats.iter() {
                    let mut field_path = stat.field_path.clone();
                    field_path = prefix_re.replace(&field_path, "").to_string();

                    // Skip the auto embedding index
                    if field_path == "___orama_auto_embedding" {
                        continue;
                    }

                    let field_stat = serde_json::to_string(&stat.stats).unwrap(); // @todo: handle error

                    match serde_json::from_str::<DeserializedStat>(&field_stat) {
                        Ok(deserialized_stat) => {
                            let total_docs = deserialized_stat.get_document_count();
                            let is_unfiltrable_string = matches!(
                                deserialized_stat,
                                DeserializedStat::UncommittedString { .. }
                                    | DeserializedStat::CommittedString { .. }
                            );

                            if is_unfiltrable_string {
                                // Skip unfiltrable strings
                                continue;
                            }

                            if total_docs > 0 {
                                if seen.contains_key(&field_path) {
                                    continue;
                                }

                                seen.insert(field_path.clone(), true);

                                fields_stats.push(FieldStats {
                                    field_path: field_path.clone(),
                                    field_type: match deserialized_stat {
                                        DeserializedStat::UncommittedBoolean { .. } => {
                                            "boolean".to_string()
                                        }
                                        DeserializedStat::CommittedBoolean { .. } => {
                                            "boolean".to_string()
                                        }
                                        DeserializedStat::UncommittedNumber { .. } => {
                                            "number".to_string()
                                        }
                                        DeserializedStat::CommittedNumber { .. } => {
                                            "number".to_string()
                                        }
                                        DeserializedStat::UncommittedStringFilter { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::CommittedStringFilter { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::UncommittedString { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::CommittedString { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::UncommittedVector { .. } => {
                                            "vector".to_string()
                                        }
                                        DeserializedStat::CommittedVector { .. } => {
                                            "vector".to_string()
                                        }
                                    },
                                });
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to deserialize field stat for {}: {}", field_path, e);
                            continue;
                        }
                    }
                }

                if !fields_stats.is_empty() {
                    indexes_stats.push(IndexesStats {
                        id: index_id.as_str().to_string(),
                        fields_stats,
                    });
                }
            }
        }

        self.step_results.push(AdvancedAutoQueryStepResult {
            step: AdvancedAutoQuerySteps::SelectProps,
            result: serde_json::to_string(&indexes_stats).unwrap(),
        });

        PropertiesSelectorInput { indexes_stats }
    }

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
